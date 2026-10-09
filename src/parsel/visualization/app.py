
import os
import numpy as np
import sympy as sp
import open3d as o3d
import torch as th
import json
import glob
import time
import copy
import pickle
from pathlib import Path

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ..data.loading import load_shape, local_to_cube, global_to_local
from ..shape_system.constants import PART_ACTIVE, PART_INACTIVE
from ..edit_dsl.edits import MAIN_VAR, load_edits
from ..shape_system.shape import Part

from ..edit_propagation.scoring import get_vert_face_map, clean_up_motion, DEFORM_THRESHOLD
from ..edit_propagation.deformation import Deformer
from ..shape_system.relations import RotationSymmetry, TranslationSymmetry, HeightContact

MIN_MOTION_THRESHOLD = 0.1
INIT_RANGE = 0.85
# DEFORM_THRESHOLD = 20

from .helpers import create_geom_map, create_switch_board, update_switch_board, update_model
from .settings import Settings
from .constants import (
    W,
    H,
    BLK_SIZE,
    MESH_ITEM,
    OBB_ITEM,
    LABEL_ITEM,
    PC_ITEM,
    UNLIT_MATERIAL,
)

# check if cuda is available
if th.cuda.is_available():
    DEVICE = th.device("cuda")
else:
    DEVICE = th.device("cpu")

POINT_LIMIT = 1024 * 8

class ParselApp:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(
        self,
        selected_obj,
        data_dir,
        output_dir,
        dataset,
        *,
        program_file=None,
        regenerate=False,
        width=W,
        height=H,
    ):
        self.selected_obj = str(Path(selected_obj))
        self.shape_id = Path(selected_obj).stem
        self.output_dir = str(Path(output_dir))
        self.method_marker = "manual"
        self.data_dir = str(Path(data_dir))
        self.dataset = dataset
        self.dataset_index = 0
        self.program_file = (
            Path(program_file).expanduser().resolve()
            if program_file is not None
            else None
        )

        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/default"

        self.edit_callbacks = {}

        processed_data, symbolic_data = load_shape(
            selected_obj,
            self.data_dir,
            dataset=self.dataset,
            regenerate=regenerate,
            add_ground=False,
        )

        (
            self.part_dict,
            self.relation_dict,
            self.intersection_dict,
            _,
        ) = processed_data

        self.name_to_geom, self.label_dict = create_geom_map(*processed_data)
        self.switch_board = create_switch_board(self.name_to_geom, self.label_dict)

        self.switch_board = update_switch_board(
            self.name_to_geom,
            self.label_dict,
            symbolic_data[0],
        )

        self.symbolic_data = symbolic_data
        self.ground_parts = []
        # base settings
        self.show_mesh = True
        self.show_bbox = False
        self.show_labels = True

        self.window = gui.Application.instance.create_window(
            "ShapeEditor", width, height)

        w = self.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        # The window has following components:
        # A Scene Render
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        # switch on the sun

        self._scene.scene.camera.look_at([0, 0, 0], [0, 1, 2], [0, 1, 0])
        # A Object Description Panel
        self._obj_panel = self.load_object_panel(self.window)
        # A Chat thread
        self._chat_panel = self.load_chat_panel(self.window)
        # A procedural editing block
        self._procedural_panel = self.load_procedural_panel(self.window)
        # Visualization settings
        self._settings_panel = self.add_settings_panel()

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        
        w.add_child(self._obj_panel)
        w.add_child(self._chat_panel)
        w.add_child(self._procedural_panel)
        w.add_child(self._settings_panel)

        self.update_from_switch_board()
        self._apply_settings()

        self.add_ground()
        self.reposition()
        self.range = (-INIT_RANGE, INIT_RANGE)
        # self.get_edit()


    def update_from_switch_board(self):
        for name, state in self.switch_board.items():
            if name in self.name_to_geom:
                item_type = MESH_ITEM
            elif name in self.label_dict:
                item_type = LABEL_ITEM
            else:
                item_type = PC_ITEM
            self._toggle_switch(name, not state, item_type)
            self._toggle_switch(name, state, item_type)

    def _toggle_switch(self, name, state, item_type):
        old_state = self.switch_board[name]
        if old_state == state:
            return
        self.switch_board[name] = state
        # load the object
        if item_type == MESH_ITEM:
            if state:
                geometry = self.name_to_geom[name]
                if isinstance(geometry, rendering.TriangleMeshModel):
                    self._scene.scene.add_model(name, self.name_to_geom[name])
                else:
                    self._scene.scene.add_geometry(
                        name, self.name_to_geom[name], self.settings.material)
            else:
                self._scene.scene.remove_geometry(name)
                # self._scene.scene.remove_model(name)
        elif item_type == OBB_ITEM:
            if state:
                self._scene.scene.add_geometry(
                    name, self.name_to_geom[name], self.settings.material)
            else:
                self._scene.scene.remove_geometry(name)
        elif item_type == LABEL_ITEM:
            # add the label
            if state:
                label, center, _ = self.label_dict[name]
                obj = self._scene.add_3d_label(center, label)
                self.label_dict[name] = (label, center, obj)
            else:
                label, center, obj = self.label_dict[name]
                self._scene.remove_3d_label(obj)
        elif item_type == PC_ITEM:
            if state:
                self._scene.scene.add_geometry(
                    name, self.name_to_geom[name], self.settings.material)
            else:
                self._scene.scene.remove_geometry(name)

    def load_object_panel(self, w):

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        panel = gui.Vert(0.0, gui.Margins(em, em, em, em))
        # panel.add_fixed(separation_height)
        obj_dets = gui.CollapsableVert(
            "Object Details", 0.0 * em, gui.Margins(0, 0, 0, 0))
        # obj_details should contain the followings:
        # 1. File name
        title = gui.Horiz(0.0, gui.Margins(0, 0, 0, 0))
        title.add_stretch()
        title.add_child(gui.Label(f"File Name: {self.selected_obj}"))
        title.add_stretch()
        obj_dets.add_child(title)

        # 2. Visualization settings.
        collapse = gui.CollapsableVert(
            "Visualization Settings", 0.25 * em, gui.Margins(em, 0, 0, 0))
        vis_settings = gui.Horiz()
        cb_1 = gui.Checkbox("Mesh")
        cb_1.set_on_checked(self._toggle_mesh)  # set the callback function
        cb_1.checked = self.show_mesh
        cb_2 = gui.Checkbox("BBox")
        cb_2.set_on_checked(self._toggle_bbox)  # set the callback function
        cb_2.checked = self.show_bbox
        cb_3 = gui.Checkbox("Labels")
        cb_3.set_on_checked(self._toggle_labels)  # set the callback function
        cb_3.checked = self.show_labels

        vis_settings.add_stretch()
        vis_settings.add_child(cb_1)
        vis_settings.add_child(cb_2)
        vis_settings.add_child(cb_3)
        vis_settings.add_stretch()
        # add switch for contact and syms.
        collapse.add_child(vis_settings)
        obj_dets.add_child(collapse)
        obj_dets.add_fixed(separation_height)

        # 3. Dynamic List of object parts
        part_list = gui.CollapsableVert(
            "Parts", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.dynamic_part_list = gui.WidgetProxy()
        part_list.add_child(self.dynamic_part_list)
        obj_dets.add_child(part_list)
        obj_dets.add_fixed(separation_height)

        # 4. Dynamic List of Object Sym Relations
        sym_relations = gui.CollapsableVert(
            "Symmetry Relations", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.dynamic_sym_list = gui.WidgetProxy()
        sym_relations.add_child(self.dynamic_sym_list)
        obj_dets.add_child(sym_relations)
        obj_dets.add_fixed(separation_height)

        # 5. Dynamic List of Object Contact Relations
        contact_relations = gui.CollapsableVert(
            "Contact Relations", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self.dynamic_contact_list = gui.WidgetProxy()
        contact_relations.add_child(self.dynamic_contact_list)
        obj_dets.add_child(contact_relations)

        self.load_object_panel_content()

        panel.add_child(obj_dets)
        return panel

    def load_object_panel_content(self):
        dpl = self.dynamic_part_list
        # dpl.clear_children()
        part_panel = gui.Vert()
        # Add buttons for show all
        btn_horiz = gui.Horiz()
        show_btn = gui.Button("Show All")
        show_btn.horizontal_padding_em = 0.5
        show_btn.set_on_clicked(self.show_all_parts)
        hide_btn = gui.Button("Hide All")
        hide_btn.horizontal_padding_em = 0.5
        hide_btn.set_on_clicked(self.hide_all_parts)

        btn_horiz.add_stretch()
        btn_horiz.add_child(show_btn)
        btn_horiz.add_child(hide_btn)
        btn_horiz.add_stretch()
        part_panel.add_child(btn_horiz)
        for ind, part in self.part_dict.items():
            part_name = f"{ind}_{part['label']}"
            caller = self.get_part_caller(ind)
            state = self.switch_board[f"{ind}_mesh"]
            cb = gui.CheckableTextTreeCell(part_name, state, caller)
            part_panel.add_child(cb)

        dpl.set_widget(part_panel)

        dsl = self.dynamic_sym_list
        # dsl.clear_children()
        sym_panel = gui.Vert()
        for ind, relation in self.relation_dict.items():
            caller = self.get_sym_caller(ind)
            cb = gui.CheckableTextTreeCell(relation['name'], False, caller)
            sym_panel.add_child(cb)
        dsl.set_widget(sym_panel)

        dcl = self.dynamic_contact_list
        # dcl.clear_children()
        contact_panel = gui.Vert()
        for ind, intersection in self.intersection_dict.items():
            caller = self.get_contact_caller(ind)
            cb = gui.CheckableTextTreeCell(intersection['name'], False, caller)
            contact_panel.add_child(cb)
        dcl.set_widget(contact_panel)

    def get_contact_caller(self, ind):
        def caller(state):
            name = f"{ind}_I"
            self._toggle_switch(name, state, item_type=PC_ITEM)
            name = f"{name}_3d_label"
            self._toggle_switch(name, state, item_type=LABEL_ITEM)
        return caller

    def get_sym_caller(self, ind):

        def caller(state):
            name = f"{ind}_R"
            self._toggle_switch(name, state, item_type=MESH_ITEM)
            name = f"{name}_3d_label"
            self._toggle_switch(name, state, item_type=LABEL_ITEM)
        return caller

    def get_part_caller(self, ind):

        def caller(state):
            if self.show_mesh:
                name = f"{ind}_mesh"
                self._toggle_switch(name, state, item_type=MESH_ITEM)
            if self.show_bbox:
                name = f"{ind}_obb"
                self._toggle_switch(name, state, item_type=OBB_ITEM)
            if self.show_labels:
                name = f"{ind}_3d_label"
                self._toggle_switch(name, state, item_type=LABEL_ITEM)
        return caller

    def show_all_parts(self, ):
        if self.show_mesh:
            for ind, part in self.part_dict.items():
                name = f"{ind}_mesh"
                self._toggle_switch(name, True, item_type=MESH_ITEM)
        if self.show_bbox:
            for ind, part in self.part_dict.items():
                name = f"{ind}_obb"
                self._toggle_switch(name, True, item_type=OBB_ITEM)
        if self.show_labels:
            for ind, part in self.part_dict.items():
                name = f"{ind}_3d_label"
                self._toggle_switch(name, True, item_type=LABEL_ITEM)
        # update the gui
        dpl = self.dynamic_part_list
        radios = dpl.get_widget().get_children()[1:]
        for radio in radios:
            radio.checkbox.checked = True

    def hide_all_parts(self, ):
        if self.show_mesh:
            for ind, part in self.part_dict.items():
                name = f"{ind}_mesh"
                self._toggle_switch(name, False, item_type=MESH_ITEM)
        if self.show_bbox:
            for ind, part in self.part_dict.items():
                name = f"{ind}_obb"
                self._toggle_switch(name, False, item_type=OBB_ITEM)
        if self.show_labels:
            for ind, part in self.part_dict.items():
                name = f"{ind}_3d_label"
                self._toggle_switch(name, False, item_type=LABEL_ITEM)
        dpl = self.dynamic_part_list
        radios = dpl.get_widget().get_children()[1:]
        for radio in radios:
            radio.checkbox.checked = False

    def load_chat_panel(self, w):
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        panel = gui.Vert(0.0, gui.Margins(em, em, em, em))

        chat_panel = gui.CollapsableVert(
            "Chat", 0.0 * em, gui.Margins(0, 0, 0, 0))
        # dynamic update the chat panel
        prev_chat = gui.WidgetProxy()
        chat_panel.add_fixed(separation_height)
        chat_panel.add_child(prev_chat)
        chat_panel.add_fixed(separation_height)
        input_box = gui.Horiz(0.0, gui.Margins(0, 0, 0, 0))
        self.chat_input = gui.TextEdit()
        self.edit_button = gui.Button("Load Edit Program")
        self.edit_button.enabled = self.program_file is not None
        self.edit_button.set_on_clicked(self.get_edit)
        input_box.add_child(self.chat_input)
        input_box.add_child(self.edit_button)
        chat_panel.add_child(input_box)

        # Add a text box and a button to send the message
        panel.add_child(chat_panel)
        return panel

    def load_procedural_panel(self, w):
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        panel = gui.Vert(0.0, gui.Margins(em, em, em, em))

        proc_panel = gui.CollapsableVert(
            "Procedural Edit", 0.0 * em, gui.Margins(0, 0, 0, 0))
        # dynamic update the chat panel
        self.proc_inner = gui.WidgetProxy()
        proc_panel.add_fixed(separation_height)
        proc_panel.add_child(self.proc_inner)
        proc_panel.add_fixed(separation_height)

        # Add a text box and a button to send the message
        panel.add_child(proc_panel)
        return panel

    def update_procedural_panel(self):
        if self.edits is None:
            return
        self.symbol_value_dict = {}
        for edit_name, edit_item in self.edits.items():
            edit_list, symbol = edit_item
            self.symbol_value_dict[symbol] = 0.0

        proc_panel = gui.Vert()
        for edit_name, edit_item in self.edits.items():
            edit_list, symbol = edit_item
            title = gui.Horiz()
            title.add_stretch()
            title.add_child(gui.Label(edit_name))
            title.add_stretch()
            proc_panel.add_child(title)
            # symbolic slider.
            sym_value_container = gui.Horiz()
            sym_value_container.add_stretch()
            sym_value = gui.Label(" 0.00")
            # sym_value_container.add_child(sym_value)
            sym_value_container.add_stretch()
            proc_panel.add_child(sym_value_container)
            # now represent mesh in a different way
            # slider for the symbolic value
            symbolic_slider = gui.Slider(gui.Slider.DOUBLE)
            symbolic_slider.set_limits(*self.range)
            edit_callback = self.get_edit_callback(symbol)

            self.edit_callbacks[symbol] = edit_callback
            symbolic_slider.set_on_value_changed(edit_callback)
            proc_panel.add_child(symbolic_slider)
        save_btn = gui.Button("Save")
        save_btn.set_on_clicked(self.save_model)
        btn_holder = gui.Horiz()
        btn_holder.add_stretch()
        btn_holder.add_child(save_btn)
        btn_holder.add_stretch()

        proc_panel.add_child(btn_holder)
        # procedural slider
        self.proc_inner.set_widget(proc_panel)


    def get_edit_callback(self, symbol):

        def edit_callback(value):
            self.symbol_value_dict[symbol] = value
            # sym_value.text = f"{value:.2f}"
            self._edit_execute(symbol)
        return edit_callback

    def _edit_prebake(self, ):
        # Now we need to register the parts which are updated
        self.symbol_to_parts = {}
        shape = self.symbolic_data[0]
        for name, edit_item in self.edits.items():
            edit, symbol = edit_item
            cur_list = []
            for part in shape.partset:
                part_eq = part.primitive.dynamic_expression()
                if symbol in part_eq.free_symbols:
                    cur_list.append(part)
            for relation in shape.all_relations():
                if hasattr(relation, 'dynamic_expression'):
                    relation_eq = relation.dynamic_expression()
                    for eq in relation_eq:
                        if symbol in eq.free_symbols:
                            cur_list.append(
                                relation.parent_part)
                            break
                    # also save the cube coord and other information.

            self.symbol_to_parts[symbol] = list(set(cur_list))
        cube_coord_dict = {}
        
        for part in shape.partset:
            part_index = part.part_index
            corresponding_mesh = self.name_to_geom[f"{part_index}_mesh"]
            obb = self.part_dict[part_index]['obb']
            point_set = [np.asarray(x.mesh.vertices) for x in corresponding_mesh.meshes]
            # points = np.asarray(corresponding_mesh.meshes[0].mesh.vertices)
            points = np.concatenate(point_set, axis=0)
            local_coords = global_to_local(points, obb) 
            cube_coords = local_to_cube(local_coords)
            # cube_coords = np.contiguousarray(cube_coords)
            cube_coords = th.from_numpy(cube_coords).float().to(DEVICE)
            cube_coord_dict[part_index] = cube_coords
        self.cube_coord_dict = cube_coord_dict

    def _edit_execute(self, symbol):
        # only update the ones in the symbol_to_parts
        relevant_parts = self.symbol_to_parts[symbol]
        for part in relevant_parts:

            if len(part.partset) > 0:
                # see which kind of execution is required.
                core_relation = part.core_relation
                output_bboxes = core_relation.execute_relation(self.symbol_value_dict)
                n_boxes = len(output_bboxes)
                if n_boxes == 1:
                    # this means take the parent mesh, and apply the edit
                    part_index = part.part_index
                    mesh_name = f"{part_index}_mesh"
                    part_eq = output_bboxes[0]
                    self._update_part_mesh(part.part_index, part_eq)

                else:
                    # this means we have to do the
                    part_id = part.part_index
                    primitive = part.core_relation.primitives[0]
                    true_id = primitive.part.part_index
                    mesh_name = f"{true_id}_mesh"
                    cur_cube_coords = self.cube_coord_dict[true_id]
                    primitive_mesh = self.name_to_geom[mesh_name]
                    # transfer bboxes at once
                    output_bboxes = np.stack([np.asarray(x) for x in output_bboxes], axis=0).astype(np.float32)
                    output_bboxes = th.from_numpy(output_bboxes).to(DEVICE)
                    new_verts = []
                    for index in range(n_boxes):
                        real_th = output_bboxes[index]
                        vertices = cur_cube_coords[..., None] * real_th[None, ...]
                        vertices = vertices.sum(dim=1)
                        new_verts.append(vertices)
                    new_verts = th.stack(new_verts, dim=0)
                    new_verts = new_verts.cpu().numpy()
                    vertices = new_verts.astype(np.float64)

                    meshes_to_copy = primitive_mesh.meshes
                    n_meshes = len(meshes_to_copy)

                    global_mesh = o3d.visualization.rendering.TriangleMeshModel()
                    for index in range(n_boxes):
                        cur_verts = vertices[index]
                        n_verts = 0
                        for j, mesh_to_copy in enumerate(meshes_to_copy):
                            new_mesh = copy.deepcopy(mesh_to_copy.mesh)
                            # new_mesh = mesh_to_copy.mesh.clone()
                            cur_n_verts = len(new_mesh.vertices)
                            new_verts = cur_verts[n_verts: n_verts + cur_n_verts]
                            new_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
                            material_idx = mesh_to_copy.material_idx
                            mesh_info = o3d.visualization.rendering.TriangleMeshModel.MeshInfo(
                                new_mesh, f"geometry_{index * n_meshes + j}", material_idx)
                            global_mesh.meshes = global_mesh.meshes + [mesh_info]
                            n_verts += cur_n_verts
                    global_mesh.materials = primitive_mesh.materials
                    mesh_name = f"{part_id}_mesh"
                    # add the color and normals
                    self.name_to_geom[mesh_name] = global_mesh
                    if self.switch_board[mesh_name]:
                        self._scene.scene.remove_geometry(mesh_name)
                        self._scene.scene.add_model(mesh_name, self.name_to_geom[mesh_name])
                    # Do the same for bbox
            else:
                part_eq = part.primitive.dynamic_expression()
                self._update_part_mesh(part.part_index, part_eq)
        self._on_shader("", 0)

    def _update_part_mesh(self, part_id, part_eq):
        mesh_name = f"{part_id}_mesh"
        cur_cube_coords = self.cube_coord_dict[part_id]
        realization = part_eq.subs(self.symbol_value_dict)
        real_np = np.asarray(realization).astype(np.float32)
        real_th = th.from_numpy(real_np).to(DEVICE)
        vertices = cur_cube_coords[..., None] * real_th[None, ...]
        vertices = vertices.sum(dim=1).cpu().numpy().astype(np.float64)
        corresponding_mesh = self.name_to_geom[mesh_name]
        
        vertice_count = 0
        for mesh_info in corresponding_mesh.meshes:
            cur_n_vertices = len(mesh_info.mesh.vertices)
            cur_vertices = vertices[vertice_count: vertice_count + cur_n_vertices]
            creation = o3d.utility.Vector3dVector(cur_vertices)
            mesh_info.mesh.vertices = creation
            vertice_count += cur_n_vertices
        if self.switch_board[mesh_name]:
            self._scene.scene.remove_geometry(mesh_name)
            if isinstance(corresponding_mesh, o3d.geometry.TriangleMesh):
                self._scene.scene.add_geometry(
                    mesh_name, corresponding_mesh, self.settings.material)
            else:
                self._scene.scene.add_model(mesh_name, corresponding_mesh)
        # Do the same for bbox
        # obb_name = f"{part_id}_obb"
        # obb_mesh = self.name_to_geom[obb_name]
        # obb_mesh.vertices = o3d.utility.Vector3dVector(real_np)
        # if self.switch_board[obb_name]:
        #     self._scene.scene.remove_geometry(obb_name)
        #     self._scene.scene.add_geometry(
        #         obb_name, self.name_to_geom[obb_name], UNLIT_MATERIAL)

    def get_edit(self, link_to_single=False):
        if self.program_file is None:
            raise ValueError(
                "A program file is required to load edit controls"
            )

        if hasattr(self, "symbol_to_parts"):
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = 0
                self._edit_execute(symbol)
                
        shape = self.symbolic_data[0]
        shape.clean_up_motion()
        self.switch_board = update_switch_board(
            self.name_to_geom,
            self.label_dict,
            shape,
        )
        self.update_from_switch_board()

        with self.program_file.open("rb") as handle:
            edit_gens = pickle.load(handle)

        if isinstance(edit_gens, dict):
            # Multiple edits
            all_edited_parts = []
            self.edits = {}
            for edit_name, edit_item in edit_gens.items():
                # Update the free variable
                edit_name = str(edit_name)
                free_variable = sp.Symbol(f"e_{edit_name}")
                for cur_edit in edit_item:
                    edit_func = cur_edit[0]
                    edit_func.amount = edit_func.amount.subs({MAIN_VAR: free_variable})
                edits, edited_parts = load_edits(shape, edit_item)

                self.edits[edit_name] = (edits, free_variable)
                all_edited_parts.extend(edited_parts)
            edited_parts = list(set(all_edited_parts))
        else:
            edits, edited_parts = load_edits(shape, edit_gens)
            ### HACK
            # edits = edits[:2]
            # part = shape.get("leg_front_right")
            # X = sp.Symbol("X")
            # new_edit = Translate(part, RightUnitVector(), amount= 0.55 * X)
            # edits.append(new_edit)
            if link_to_single:
                for edit in edits:
                    for free_var in edit.amount.free_symbols:
                        if free_var != MAIN_VAR:
                            edit.amount = edit.amount.subs({free_var: MAIN_VAR})
            self.edits = {
                "Base": (edits, MAIN_VAR)
            }

        # Fix Visualization
        flip_candidates = []
        for part in edited_parts:
            if isinstance(part, Part):
                if part.parent is not None:
                    for child in part.parent.sub_parts:
                        flip_candidates.append(child)
                    flip_candidates.append(part.parent)
                    flip_candidates.append(part)
        flip_candidates = list(set(flip_candidates))
        for part in flip_candidates:
            name = f"{part.part_index}_mesh"
            state = self.switch_board[name]
            self._toggle_switch(name, not state, item_type=MESH_ITEM)
            self._toggle_switch(name, not state, item_type=MESH_ITEM)
        
        for name, edit_item in self.edits.items():
            edit_list, sym = edit_item
            for edit in edit_list:
                edit.propagate()

        for part in shape.partset:
            part_edited = len(part.primitive.edit_sequence) > 0
            if part_edited:
                part.state[0] = PART_ACTIVE
            if len(part.sub_parts) > 0:
                edited_children = [len(child.primitive.edit_sequence) > 0 for child in part.sub_parts]
                if any(edited_children):
                    shape.deactivate_parent(part)
        if not isinstance(edit_gens, dict):
            self.set_range()
        self.update_procedural_panel()
        self._edit_prebake()

    def save_model(self, ):

        import pymeshlab
        import trimesh
        simplify = True
        first_spr_depth = 16
        second_spr_depth = 10
        SAVE_DIRECT = False
        MAX_POINTS = 200_000
        secondary_cleanup = False
        start_time = time.time()
        shape = self.symbolic_data[0]
        save_dir = os.path.join(self.output_dir, "objs", self.method_marker, f"{self.shape_id}_{self.dataset_index}_cur")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        COMPAT = False
        # What is it
        # 1. convert/update the form of the shape. 
        update_model(shape, self.part_dict, self.symbol_to_parts, 
                     self.symbol_value_dict, self.cube_coord_dict, 
                     self.name_to_geom)
        

        if SAVE_DIRECT:
            model_id = self.selected_obj.split("/")[-1].split(".")[0]

            data_dir = os.path.join(self.data_dir, str(model_id), 'objs')
            final_save_dir = os.path.join(save_dir, "objs_def")
            Path(final_save_dir).mkdir(parents=True, exist_ok=True)
            count = 0
            parts = list(shape.partset)
            # back_slat_parts = [x for x in parts if 'back_surface_vertical_bar' in x.label]
            # not_back_slat_parts = [x for x in parts if 'back_surface_vertical_bar' not in x.label]
            # not_back_slat_parts= sorted(not_back_slat_parts, key=lambda x: x.label)
            # back_slat_parts = sorted(back_slat_parts, key=lambda x: x.label)
            # parts = not_back_slat_parts + back_slat_parts
            # count = 0
            for cur_part in parts:
                if cur_part.state[0] == PART_ACTIVE:
                    print(cur_part.label)
                    mesh_name = f"{cur_part.part_index}_mesh"
                    part_geom = self.name_to_geom[mesh_name]
            
                    part_index = cur_part.part_index
                    original_files = self.part_dict[part_index]['objs']

                    # Assume only one file?
                    for ind, obj_file_name in enumerate(original_files):
                        # obj_file = os.path.join(data_dir, f"{obj_file}.obj")
                        if "." not in obj_file_name:
                            obj_file_name = f"{obj_file_name}.obj"
                        obj_file = os.path.join(data_dir, f"{obj_file_name}")
                        # with trimesh
                        if COMPAT:
                            original_glb_mesh = trimesh.load_mesh(obj_file)

                            original_inside_mesh = original_glb_mesh.geometry['geometry_0']
                            deformed_mesh = self.name_to_geom[f"{part_index}_mesh"].meshes[ind].mesh
                            # original_inside_mesh.vertices = np.asarray(deformed_mesh.vertices)

                            # vertices = np.asarray(original_inside_mesh.vertices)
                            # get cube_coords
                            # deform with the GT then
                            # Transfer UVs based on pts from trimesh and original mesh
                            # Or update the Cube Coords? 
                            if len(original_inside_mesh.vertices) != len(deformed_mesh.vertices):
                                print("Different Vertices")
                                print(len(original_inside_mesh.vertices), len(deformed_mesh.vertices))
                                v1 = np.asarray(original_inside_mesh.vertices)
                                v2 = np.asarray(deformed_mesh.vertices)
                                v1 = th.from_numpy(v1).float().to(DEVICE)
                                v2 = th.from_numpy(v2).float().to(DEVICE)
                                # find the top 3 nearest
                                # Load the original undeformed vertices
                                
                                original_vals = {x:y for x, y in self.symbol_value_dict.items()}
                                for symbol, caller in self.symbol_value_dict.items():
                                    self.symbol_value_dict[symbol] = 0
                                    self._edit_execute(symbol)
                                undeformed_mesh = self.name_to_geom[f"{part_index}_mesh"].meshes[ind].mesh
                                v3 = np.asarray(undeformed_mesh.vertices)
                                v3 = th.from_numpy(v3).float().to(DEVICE)
                                for symbol, caller in self.symbol_value_dict.items():
                                    self.symbol_value_dict[symbol] = original_vals[symbol]
                                    self._edit_execute(symbol)

                                
                                distances = th.cdist(v1, v3)
                                min_distances, min_indices = th.topk(-distances, 3, dim=1)
                                min_distances = min_distances + 1e-6
                                weights = 1 / min_distances
                                normalized = weights / weights.sum(dim=1, keepdim=True)
                                new_vertices = (normalized[..., None] * v2[min_indices]).sum(dim=1)

                                print("Nearest Vertices")
                                print(len(original_inside_mesh.vertices), new_vertices.shape[0])
                                original_inside_mesh.vertices = new_vertices.cpu().numpy()
                                # continue
                            else:
                                original_inside_mesh.vertices = np.asarray(deformed_mesh.vertices)
                            # original_inside_mesh.faces = np.asarray(deformed_mesh.triangles)
                            original_glb_mesh.geometry['geometry_0'] = original_inside_mesh
                            # original_glb_mesh.geometry['geometry_0'] = 
                            save_file = os.path.join(final_save_dir, f"{cur_part.part_index}_{ind}.glb")
                            original_glb_mesh.export(save_file,)
                            print('exporting', save_file)
                        else:
                            deformed_mesh = self.name_to_geom[f"{part_index}_mesh"].meshes[ind].mesh
                            save_file = os.path.join(final_save_dir, f"{count}.obj")
                            o3d.io.write_triangle_mesh(save_file, deformed_mesh)
                            count += 1

            
        
        part_count = 0
        for cur_part in shape.partset:
            if cur_part.state[0] == PART_ACTIVE:
                part_count += 1
        per_instance_count = MAX_POINTS // part_count
        n_faces = 0
        all_points = []
        all_face_counts = []
        all_original_geoms = []
        all_edited_parts = []
        for cur_part in shape.partset:
            if cur_part.state[0] == PART_ACTIVE:
                print(cur_part.label)
                all_edited_parts.append(cur_part)
                mesh_name = f"{cur_part.part_index}_mesh"
                part_geom = self.name_to_geom[mesh_name]
                all_original_geoms.append(part_geom)
                # add to processed mesh
                # save mesh as glb
                cur_mesh = o3d.geometry.TriangleMesh()
                for mesh_info in part_geom.meshes:
                    cur_mesh += mesh_info.mesh
                points = cur_mesh.sample_points_poisson_disk(per_instance_count)
                all_points.append(points)
                n_faces += len(cur_mesh.triangles)
                all_face_counts.append(len(cur_mesh.triangles))

        final_pc = o3d.geometry.PointCloud()
        for ind, points in enumerate(all_points):
            final_pc += points

        # write the pc
        save_file = os.path.join(save_dir, f"{shape.label}_pc.ply")
        o3d.io.write_point_cloud(save_file, final_pc)

        ms = pymeshlab.MeshSet(verbose=True)
        ms.load_new_mesh(save_file)
        ms.apply_filter('generate_surface_reconstruction_screened_poisson', 
                        depth=14, samplespernode=1, pointweight=6)
        if simplify:
            ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                            targetfacenum= int(n_faces * 7),
                            autoclean=True,)
        ms.set_current_mesh(1)
        save_file = os.path.join(save_dir, f"{shape.label}_recon.obj")
        ms.save_current_mesh(file_name=save_file, save_vertex_normal=True)
        ms.clear()

        # Splitting done in o3d
        # 1. Load the mesh
        mesh = o3d.io.read_triangle_mesh(save_file)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_normals = np.asarray(mesh.vertex_normals)
        mesh.compute_vertex_normals()
        face_centers = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3
        face_centers = th.from_numpy(face_centers).float().to(DEVICE)
        min_distance = th.ones(face_centers.shape[0]).to(DEVICE) * 1000
        min_index = th.ones(face_centers.shape[0]).to(DEVICE)  * -1
        for ind, points in enumerate(all_points):
            cur_points = th.from_numpy(np.asarray(points.points)).float().to(DEVICE)
            distances = th.cdist(face_centers, cur_points)
            cur_min_distance, _ = th.min(distances, dim=1)
            cond_map = cur_min_distance < min_distance
            min_distance[cond_map] = cur_min_distance[cond_map]
            min_index[cond_map] = ind
        
        min_index = min_index.cpu().numpy()
    
        # Now we have for each part the corresponding faces.
        # Now we need to create the mesh for each part.
        partwise_meshes = []
        for index in range(part_count):
            cur_faces = faces[min_index == index]
            # Create the mesh
            cur_mesh = o3d.geometry.TriangleMesh()
            cur_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            cur_mesh.triangles = o3d.utility.Vector3iVector(cur_faces)
            cur_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
            # clear the vertices
            if not secondary_cleanup:
                partwise_meshes.append(cur_mesh)
            else:
                cur_mesh.remove_duplicated_triangles()
                cur_mesh.remove_duplicated_vertices()
                cur_mesh.remove_unreferenced_vertices()

                triangle_clusters, cluster_n_triangles, cluster_area = (
                    cur_mesh.cluster_connected_triangles())

                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                cluster_area = np.asarray(cluster_area)
                # Option 1: Keep Largest component

                # largest_cluster_idx = cluster_n_triangles.argmax()
                # triangles_to_remove = triangle_clusters != largest_cluster_idx
                # Option 2: Smaller components
                triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
                if len(triangles_to_remove) < len(cur_faces) * 0.75:
                    cur_mesh.remove_triangles_by_mask(triangles_to_remove)
                # save
                # Sample save recon with SPR again
                points = cur_mesh.sample_points_poisson_disk(sample_count=per_instance_count * 2)
                points += all_points[index]

                # save it
                save_file = os.path.join(save_dir, f"{shape.label}_{index}_recon.ply")
                o3d.io.write_point_cloud(save_file, points)

                ms.load_new_mesh(save_file)
                ms.clear()
                ms.load_new_mesh(save_file)
                ms.apply_filter('generate_surface_reconstruction_screened_poisson', 
                                depth=second_spr_depth, samplespernode=1, pointweight=6)
                if simplify:
                    ms.apply_filter("meshing_decimation_quadric_edge_collapse",
                                    targetfacenum= int(all_face_counts[index] * 1.5),
                                    autoclean=True,)
                ms.set_current_mesh(1)

                save_file = os.path.join(save_dir, f"{shape.label}_{index}_recon.obj")
                ms.save_current_mesh(file_name=save_file, save_vertex_normal=True)
                # Now this is the part where 
                cur_mesh = o3d.io.read_triangle_mesh(save_file)
                partwise_meshes.append(cur_mesh)


        model_id = self.selected_obj.split("/")[-1].split(".")[0]
        data_dir = os.path.join(self.data_dir, str(model_id), 'objs')
        final_save_dir = os.path.join(save_dir, "objs")
        Path(final_save_dir).mkdir(parents=True, exist_ok=True)
        # Now transfer uv from the original mesh.
        for index, recon_mesh in enumerate(partwise_meshes):
            part = all_edited_parts[index]
            part_index = part.part_index
            original_files = self.part_dict[part_index]['objs']

            # Assume only one file?
            for obj_file in original_files:
                # obj_file = os.path.join(data_dir, f"{obj_file}.obj")
                if "." not in obj_file:
                    obj_file = f"{obj_file}.obj"
                obj_file = os.path.join(data_dir, f"{obj_file}")
                # with trimesh
                original_glb_mesh = trimesh.load_mesh(obj_file)
            
            original_inside_mesh = original_glb_mesh.geometry['geometry_0']
            deformed_mesh = self.name_to_geom[f"{part_index}_mesh"].meshes[0].mesh

            # print(original_inside_mesh.vertices.shape, np.asarray(deformed_mesh.vertices).shape)
            # original_inside_mesh.vertices = np.asarray(deformed_mesh.vertices)
            # original_inside_mesh.faces = np.asarray(deformed_mesh.triangles)
            # Get the deformed mesh here directly using the mesh calculation.
            # original_glb_mesh.geometry['geometry_0'] = original_inside_mesh
            # save_file = os.path.join(save_dir, f"deformed_{shape.label}_{index}.glb")
            # original_glb_mesh.export(save_file, file_type='glb')

            # recon_trimesh = trimesh.Trimesh(vertices=np.asarray(recon_mesh.vertices), faces=np.asarray(recon_mesh.triangles))
            inside_uvs = original_inside_mesh.visual.uv
            inside_uvs = th.from_numpy(inside_uvs).float().to(DEVICE)
            repositioned_verts = np.asarray(deformed_mesh.vertices)
            new_verts = np.asarray(recon_mesh.vertices)
            # find closest point for each
            new_verts = th.from_numpy(new_verts).float().to(DEVICE)
            repositioned_verts = th.from_numpy(repositioned_verts).float().to(DEVICE)
            distances = th.cdist(new_verts, repositioned_verts)
            # Now make it radial 
            k = 1
            cur_min_distance, uv_indices = th.topk(-distances, dim=1, k=k)
            
            new_uvs = inside_uvs[uv_indices]
            if k == 1:
                new_uvs = new_uvs.squeeze(1)
            else:
                new_uvs = (cur_min_distance[..., None] * new_uvs).sum(-1) / cur_min_distance.sum(-1, keepdim=True)
            new_uvs = new_uvs.cpu().numpy()
            original_inside_mesh.vertices = np.asarray(recon_mesh.vertices)
            original_inside_mesh.faces = np.asarray(recon_mesh.triangles)
            original_inside_mesh.visual.uv = new_uvs
            original_glb_mesh.geometry['geometry_0'] = original_inside_mesh
            # 
            # save the mesh
            save_file = f'obj_{index}.glb'
            self.part_dict[part_index]['new_objs'] = [save_file]

            for obj_file in original_files:
                # obj_file = os.path.join(data_dir, f"{obj_file}.obj")
                if "." not in obj_file:
                    obj_file = f"{obj_file}.obj"
            save_file = os.path.join(final_save_dir, obj_file)
            original_glb_mesh.export(save_file, file_type='glb')

        # Then simply make the glb files based on the original
        save_dicts = []
        for part in shape.partset:
            if part.state[0] == PART_INACTIVE:
                continue
            part_index = part.part_index
            original_dict = self.part_dict[part_index]
            
            cur_part_dict = {
                'name': part.original_label,
                'objs': original_dict['new_objs'],
                'children': [],
            }
            save_dicts.append(cur_part_dict)

        parent_node = {'objs': [], 'name': shape.label, 
                    'children': save_dicts, 'hier': 0}
        
        save_dir = os.path.join(self.output_dir, "objs", self.method_marker, f"{self.shape_id}")
        data_file = os.path.join(save_dir, "result_after_merging.json")
        with open(data_file, "w") as f:
            json.dump([parent_node], f)
        end_time = time.time()
        print(f"DONE! Time taken: {end_time - start_time:.2f}s")



        # Load the mesh, sample face centroids.
        # assign faces to parts based on min distance point.

        # Create mesh per_part with the extracted faces.
        # COnsider tirmesh

        # Transfer UV from the original using nearest neighbor.



    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        self._scene.frame = r

        width = max(BLK_SIZE * 3, r.width // 4)
        desired = self._obj_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        width = min(width, desired.width)
        height = min(r.height, BLK_SIZE * 5)
        height = min(height, desired.height)

        self._obj_panel.frame = gui.Rect(r.get_left(), r.y + em, width, height)

        # similarly set the settings for the chat box
        width = max(BLK_SIZE * 4, r.width // 3)
        desired = self._chat_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        # width = min(width, desired.width)
        height = min(r.height, desired.height)
        self._chat_panel.frame = gui.Rect(
            r.get_left(), r.height - height, width, height)

        # and for the 3D
        width = max(BLK_SIZE * 4, r.width // 3)
        desired = self._procedural_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        # width = min(width, desired.width)
        height = min(r.height, desired.height)
        center = (r.get_right() + r.get_left()) // 2
        # self._procedural_panel.frame = gui.Rect(r.get_right() - width, r.height - height, width, height)
        self._procedural_panel.frame = gui.Rect(
            center - width//2, r.height - height, width, height)
        # and for the 3D

        width = max(BLK_SIZE * 3, r.width // 4)
        desired = self._settings_panel.calc_preferred_size(
            layout_context, gui.Widget.Constraints())
        width = min(width, desired.width)
        height = min(r.height, BLK_SIZE * 5)
        height = min(height, desired.height)

        self._settings_panel.frame = gui.Rect(r.get_right()- width, r.y + em, width, height)
    # Action functions

    def _toggle_mesh(self, state):
        for ind, part in self.part_dict.items():
            name = f"{ind}_mesh"
            self._toggle_switch(name, state, item_type=MESH_ITEM)
        self.show_mesh = state

    def _toggle_bbox(self, state):
        for ind, part in self.part_dict.items():
            name = f"{ind}_obb"
            self._toggle_switch(name, state, item_type=OBB_ITEM)
        self.show_bbox = state

    def _toggle_labels(self, state):
        for ind, part in self.part_dict.items():
            name = f"{ind}_3d_label"
            self._toggle_switch(name, state, item_type=LABEL_ITEM)
        self.show_labels = state


    def add_settings_panel(self):

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        w = self.window
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._fly_button = gui.Button("Fly")
        self._fly_button.horizontal_padding_em = 0.5
        self._fly_button.vertical_padding_em = 0
        self._fly_button.set_on_clicked(self._set_mouse_mode_fly)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        self._sun_button = gui.Button("Sun")
        self._sun_button.horizontal_padding_em = 0.5
        self._sun_button.vertical_padding_em = 0
        self._sun_button.set_on_clicked(self._set_mouse_mode_sun)
        self._ibl_button = gui.Button("Environment")
        self._ibl_button.horizontal_padding_em = 0.5
        self._ibl_button.vertical_padding_em = 0
        self._ibl_button.set_on_clicked(self._set_mouse_mode_ibl)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._fly_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(self._sun_button)
        h.add_child(self._ibl_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(gui.Label("Lighting profiles"))
        view_ctrls.add_child(self._profiles)
        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(view_ctrls)

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):

            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = ParselApp.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(advanced)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        self._shader.add_item(ParselApp.MATERIAL_NAMES[0])
        self._shader.add_item(ParselApp.MATERIAL_NAMES[1])
        self._shader.add_item(ParselApp.MATERIAL_NAMES[2])
        self._shader.add_item(ParselApp.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        settings_panel.add_fixed(separation_height)
        settings_panel.add_child(material_settings)
        return settings_panel

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_fly(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _set_mouse_mode_sun(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _set_mouse_mode_ibl(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_IBL)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(ParselApp.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()


    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size



    def reposition(self):
        shape = self.symbolic_data[0]
        mega_mesh = o3d.geometry.TriangleMesh()
        for part in shape.partset:
            if part.label == "ground":
                continue
            mesh_name = f"{part.part_index}_mesh"
            mesh = self.name_to_geom[mesh_name]
            for mesh_info in mesh.meshes:
                mega_mesh += mesh_info.mesh
        aabb = mega_mesh.get_axis_aligned_bounding_box()
        min_y = aabb.get_min_bound()[1]
        # move the ground down
        self.ground_plane.translate((0, min_y, 0))
        self._scene.scene.add_geometry("ground_plane", self.ground_plane, self.settings._materials['defaultLit'])
        # set camera according to the aabb
        # get extent
    def add_ground(self):

        ground_plane = o3d.geometry.TriangleMesh.create_box(
            50.0, 0.1, 50.0, create_uv_map=True, map_texture_to_each_face=True)
        ground_plane.compute_triangle_normals()
        rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi, 0, 0))
        ground_plane.rotate(rotate_180)
        ground_plane.translate((-25.0, -0.1, -25.0))
        ground_plane.paint_uniform_color((0.25, 0.25, 0.25))
        self.ground_plane = o3d.t.geometry.TriangleMesh.from_legacy(ground_plane)


    def set_range(self):
        cur_range = INIT_RANGE
        for _ in range(11):
            limits = (-cur_range, cur_range)
            unsafe = any(
                self.check_flipping(value * 1.25)
                or self.check_distortion(value)
                for value in limits
            )
            if unsafe:
                cur_range = cur_range * 0.75
                continue
            if all(self.check_minimal_motion(value) for value in limits):
                cur_range = cur_range * 4 / 3.0
                continue
            break
        self.range = (-cur_range, cur_range)
        return self.range

    def check_distortion(self, cur_range):
        distortion = []
        for edit_name, edits in self.edits.items():
            edits, symbol = edits
            for edit in edits:
                # edit.propagate()
                
                part_to_edit = edit.operand
                if isinstance(part_to_edit, Part):
                    static = part_to_edit.primitive.static_expression()
                    # A variant where this is driven by 
                    static_np = np.asarray(static).astype(np.float32)
                    v_map, F = get_vert_face_map(part_to_edit)
                    seq_points = static_np[v_map]
                    dynamic_expr = part_to_edit.primitive.dynamic_expression()
                    real_dynamic = dynamic_expr.subs({symbol: cur_range})
                    dynamic_np = np.asarray(real_dynamic).astype(np.float32)
                    dynamic_points = dynamic_np[v_map]
                    deform = Deformer(seq_points, dynamic_points, F)
                    # Assume that the thing is scaled s.t. max length is 1.
                    primitive = part_to_edit.primitive
                    index_0 = primitive.name_to_indices[('back', 'down', 'left')]
                    index_1 = primitive.name_to_indices[('back', 'left', 'up')]
                    index_2 = primitive.name_to_indices[('back', 'down', 'right')]
                    index_3 = primitive.name_to_indices[('down', 'front', 'left')]
                    vec_1 = static_np[index_1, :] - static_np[index_0, :]
                    vec_2 = static_np[index_2, :] - static_np[index_0, :]
                    vec_3 = static_np[index_3, :] - static_np[index_0, :]
                    all_vecs = np.stack([vec_1[0], vec_2[0], vec_3[0]], axis=0)
                    
                    # dy_vec_1 = dynamic_np[index_1, :] - dynamic_np[index_0, :]
                    # dy_vec_2 = dynamic_np[index_2, :] - dynamic_np[index_0, :]
                    # dy_vec_3 = dynamic_np[index_3, :] - dynamic_np[index_0, :]
                    # dy_all_vecs = np.stack([dy_vec_1[0], dy_vec_2[0], dy_vec_3[0]], axis=0)
                    # # norm
                    # all_sizes = np.linalg.norm(all_vecs, axis=1)
                    # dy_all_sizes = np.linalg.norm(dy_all_vecs, axis=1)
                    # ratio_1 = (all_sizes) / (dy_all_sizes + 1e-6)
                    # ratio_2 = (dy_all_sizes) / (all_sizes + 1e-6)
                    # max_ratio = np.maximum(ratio_1, ratio_2) 
                    # if max(max_ratio) > 8:
                    #     high_distortion = True
                    #     break
                    normed_vals = np.linalg.norm(all_vecs, axis=1)
                    scale = max(normed_vals)
                    true_energy = deform.energy / scale
                    distortion.append(true_energy)
                    # clean_up_motion(part_to_edit, edit)
                else:
                    distortion.append(0)
        return bool(distortion) and np.max(distortion) > DEFORM_THRESHOLD

    def check_minimal_motion(self, cur_range):
        deltas = []
        for edit_name, edits in self.edits.items():
            edits, symbol = edits
            for edit in edits:
                part_to_edit = edit.operand
                # edit.propagate()
                if isinstance(edit.operand, Part):
                    static = part_to_edit.primitive.static_expression()
                    # A variant where this is driven by 
                    static_np = np.asarray(static).astype(np.float32)
                    
                    dynamic_expr = part_to_edit.primitive.dynamic_expression()
                    real_dynamic = dynamic_expr.subs({symbol: cur_range})
                    dynamic_np = np.asarray(real_dynamic).astype(np.float32)
                    cur_delta = np.linalg.norm(static_np - dynamic_np, axis=1)
                    deltas.append(np.max(cur_delta))
                else:
                    # For this we need to check if there is change between the parts
                    static_expr = part_to_edit.static_expression()
                    dynamic_expr = part_to_edit.dynamic_expression()
                    dynamic_expr = [x.subs({symbol: cur_range}) for x in dynamic_expr]
                    static_expr = [np.asarray(x).astype(np.float32) for x in static_expr]
                    dynamic_expr = [np.asarray(x).astype(np.float32) for x in dynamic_expr]
                    if isinstance(part_to_edit, RotationSymmetry):
                        static_start, _, _, static_angle, static_count = static_expr
                        dynamic_start, _, _, dynamic_angle, dynamic_count = dynamic_expr
                        delta = np.linalg.norm(static_start - dynamic_start) +\
                              np.linalg.norm(static_angle - dynamic_angle) + \
                                np.linalg.norm(static_count - dynamic_count)
                        deltas.append(delta)
                    elif isinstance(edit.operand, TranslationSymmetry):
                        static_start, static_center, static_end, static_delta, static_count = static_expr
                        dynamic_start, dynamic_center, dynamic_end, dynamic_delta, dynamic_count = dynamic_expr
                        delta = np.linalg.norm(static_start - dynamic_start) +\
                                np.linalg.norm(static_center - dynamic_center) + \
                                np.linalg.norm(static_end - dynamic_end) + \
                                np.linalg.norm(static_delta - dynamic_delta) + \
                                np.linalg.norm(static_count - dynamic_count)
                        deltas.append(delta)
                # clean_up_motion(part_to_edit, edit)
        return bool(deltas) and np.max(deltas) < MIN_MOTION_THRESHOLD
    
    def check_flipping(self, cur_range):
        for edit_name, edits in self.edits.items():
            edits, symbol = edits
            # now we need to see if any side becomes negative at range, if it does then half the range.
            dyn_exprs = []
            indices = []
            init_directions = []
            for edit in edits:
                if isinstance(edit.operand, Part):
                    dyn_expr = edit.operand.primitive.dynamic_expression()
                    dyn_exprs.append(dyn_expr)
                    # Also record the left right and up down
                    primitive = edit.operand.primitive 
                    index_0 = primitive.name_to_indices[('back', 'down', 'left')]
                    index_1 = primitive.name_to_indices[('back', 'left', 'up')]
                    index_2 = primitive.name_to_indices[('back', 'down', 'right')]
                    index_3 = primitive.name_to_indices[('down', 'front', 'left')]
                    indices.append((index_0, index_1, index_2, index_3))
            direction_sets = []
            for ind, expr in enumerate(dyn_exprs):
                static_expr = expr.subs({symbol: 0})
                index_0, index_1, index_2, index_3 = indices[ind]
                direction_1 = (static_expr[index_1, :]- static_expr[index_0, :]).normalized()
                direction_2 = (static_expr[index_2, :]- static_expr[index_0, :]).normalized()             
                direction_3 = (static_expr[index_3, :]- static_expr[index_0, :]).normalized()             
                direction_sets.append((direction_1, direction_2, direction_3))
            
            flips = False
            for ind, expr in enumerate(dyn_exprs):
                # get the 3 sides
                dynamic_expr = expr.subs({symbol: cur_range})
                index_0, index_1, index_2, index_3 = indices[ind]
                direction_1 = (dynamic_expr[index_1, :]- dynamic_expr[index_0, :]).normalized()
                direction_2 = (dynamic_expr[index_2, :]- dynamic_expr[index_0, :]).normalized()
                direction_3 = (dynamic_expr[index_3, :]- dynamic_expr[index_0, :]).normalized()
                # previous
                prev_dir_1, prev_dir_2, prev_dir_3 = direction_sets[ind]
                # check dot prod
                dot_1 = direction_1.dot(prev_dir_1)
                dot_2 = direction_2.dot(prev_dir_2)
                dot_3 = direction_3.dot(prev_dir_3)
                if dot_1 < 0 or dot_2 < 0 or dot_3 < 0:
                    return True
        return False

    def check_under_ground(self, cur_range):
        # how to make it close to flip? 
        under_ground = False
        min_val = 0
        shape = self.symbolic_data[0]
        for part in shape.partset:
            if part.state[0] == PART_ACTIVE:
                static_val = part.primitive.static_expression()
                static_np = np.asarray(static_val).astype(np.float32)
                min_y = np.min(static_np[:, 1])
                if min_y < min_val:
                    min_val = min_y
    


        for edit_name, edits in self.edits.items():
            edits, symbol = edits
            deltas = []
            ground_labels = [x.full_label for x in self.ground_parts] 
            for edit in edits:
                part_to_edit = edit.operand
                # edit.propagate()
                if isinstance(edit.operand, Part):
                    all_rel = edit.operand.all_relations()
                    if any([isinstance(x, HeightContact) for x in all_rel]):
                        continue
                    if part_to_edit.full_label in ground_labels:
                        continue
                    dynamic_expr = part_to_edit.primitive.dynamic_expression()
                    real_dynamic = dynamic_expr.subs({symbol: cur_range})
                    dynamic_np = np.asarray(real_dynamic).astype(np.float32)
                    max_y = np.max(dynamic_np[:, 1])
                    if max_y < min_val:
                        under_ground = True
                        break
        return under_ground
