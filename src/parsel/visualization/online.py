"""Natural-language edit generation for the interactive application."""

import sympy as sp
import open3d.visualization.gui as gui

from ..edit_dsl.edits import load_edits
from .app import ParselApp
from .helpers import update_switch_board


class OnlineParselApp(ParselApp):
    """ParSEL window whose edit button invokes the prompting workflow."""

    def __init__(self, *args, item, repetitions, **kwargs):
        self.item = item
        self.repetitions = repetitions
        super().__init__(*args, **kwargs)

        self.edit_button.text = "Get Edit"
        self.edit_button.enabled = True
        self._obj_panel.get_children()[0].set_is_open(False)

    def get_edit(self, link_to_single=False):
        del link_to_single
        self._reset_current_edits()
        self._initialize_workflow()

        from ..data.loading import load_shape
        from ..edit_dsl.edits import MAIN_VAR
        from ..shape_system.constants import PART_ACTIVE, PART_INACTIVE
        from ..workflow import run_edit_workflow

        _, symbolic_data = load_shape(
            self.selected_obj,
            self.data_dir,
            dataset=self.dataset,
            regenerate=False,
            add_ground=True,
        )
        workflow_shape = symbolic_data[0]
        workflow_shape.clean_up_motion()
        if "category" in self.item:
            workflow_shape.label = self.item["category"]

        all_edits, _, _, _ = run_edit_workflow(
            workflow_shape,
            self.chat_input.text_value,
            self.prompter,
            self.edit_proposer,
        )

        edit_number = len(self.edits)
        free_var = sp.Symbol(f"f_{edit_number}")
        for edit in all_edits:
            edit.amount = edit.amount.subs({MAIN_VAR: free_var})

        shape = self.symbolic_data[0]
        self.switch_board = update_switch_board(
            self.name_to_geom,
            self.label_dict,
            shape,
        )
        self.update_from_switch_board()

        saved_edits = [edit.save_format() for edit in all_edits]
        edits, _ = load_edits(shape, saved_edits)
        self.edits[f"Edit_{edit_number}"] = (edits, free_var)

        shape.clean_up_motion()
        for edit_list, _ in self.edits.values():
            for edit in edit_list:
                edit.propagate()

        for part in shape.partset:
            if not part.partset:
                continue
            if (
                part.core_relation.edit_sequence
                or part.primitive.edit_sequence
            ):
                part.state[0] = PART_ACTIVE
                for subpart in part.partset:
                    subpart.state[0] = PART_INACTIVE
            elif any(
                subpart.primitive.edit_sequence
                for subpart in part.partset
            ):
                part.state[0] = PART_INACTIVE
                for subpart in part.partset:
                    subpart.state[0] = PART_ACTIVE

        for part in shape.partset:
            if part.primitive.edit_sequence:
                part.state[0] = PART_ACTIVE
            if part.sub_parts and any(
                child.primitive.edit_sequence
                for child in part.sub_parts
            ):
                shape.deactivate_parent(part)

        self.update_from_switch_board()
        self.update_procedural_panel()
        self._edit_prebake()
        self.chat_input.text_value = ""

    def _reset_current_edits(self):
        if not hasattr(self, "symbol_to_parts"):
            return
        for symbol in self.symbol_value_dict:
            self.symbol_value_dict[symbol] = 0
            self._edit_execute(symbol)

    def _initialize_workflow(self):
        if hasattr(self, "edits"):
            return

        from ..edit_propagation.parallel import ParallelEditProposer
        from ..prompting import Prompter
        from ..prompting.config import (
            KEY,
            MODEL,
            PROVIDER,
            SEED,
            TEMPERATURE,
        )

        self.edits = {}
        self.prompter = Prompter(
            PROVIDER,
            KEY,
            MODEL,
            TEMPERATURE,
            SEED,
        )
        self.prompter.repetitions = self.repetitions
        self.edit_proposer = ParallelEditProposer()

    def update_procedural_panel(self):
        self.symbol_value_dict = {}
        if self.edits is None:
            return
        for _, symbol in self.edits.values():
            self.symbol_value_dict[symbol] = 0.0

        panel = gui.Vert()
        for edit_name, (_, symbol) in self.edits.items():
            title = gui.Horiz()
            title.add_stretch()
            title.add_child(gui.Label(edit_name))
            title.add_stretch()
            panel.add_child(title)

            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-0.5, 0.5)
            callback = self.get_edit_callback(symbol)
            self.edit_callbacks[symbol] = callback
            slider.set_on_value_changed(callback)
            panel.add_child(slider)

            button_row = gui.Horiz()
            remove_button = gui.Button("Remove")
            remove_button.set_on_clicked(
                self.remove_edit(edit_name, symbol)
            )
            button_row.add_stretch()
            button_row.add_child(remove_button)
            button_row.add_stretch()
            panel.add_child(button_row)

        save_button = gui.Button("Save")
        save_button.set_on_clicked(self.save_model)
        button_row = gui.Horiz()
        button_row.add_stretch()
        button_row.add_child(save_button)
        button_row.add_stretch()
        panel.add_child(button_row)
        self.proc_inner.set_widget(panel)

    def remove_edit(self, edit_name, symbol):
        def callback():
            for current_symbol in self.symbol_value_dict:
                self.symbol_value_dict[current_symbol] = 0
                self._edit_execute(current_symbol)
            self.edits.pop(edit_name)
            self.symbol_value_dict.pop(symbol)
            self.update_from_switch_board()
            self.update_procedural_panel()
            self._edit_prebake()

        return callback
