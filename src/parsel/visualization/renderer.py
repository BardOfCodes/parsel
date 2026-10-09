"""Small offscreen renderer for cached ParSEL shapes."""

from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from .helpers import create_geom_map, update_switch_board


DEFAULT_RESOLUTION = 512


class ShapeRenderer:
    """Render the active parts of one shape to a PNG."""

    def __init__(
        self,
        processed_data,
        symbolic_data,
        output_path,
        resolution=DEFAULT_RESOLUTION,
    ):
        if resolution <= 0:
            raise ValueError("Render resolution must be positive")

        self.output_path = Path(output_path)
        self.shape = symbolic_data[0]
        self.name_to_geom, self.label_dict = create_geom_map(*processed_data)
        self.switch_board = update_switch_board(
            self.name_to_geom,
            self.label_dict,
            self.shape,
        )

        self._scene = rendering.OffscreenRenderer(
            int(resolution),
            int(resolution),
        )
        self._configure_scene()
        bounds = self._shape_bounds()
        self._add_ground(bounds)
        self._add_active_geometry()
        self._position_camera(bounds)

    def render(self):
        """Write the configured scene and return its output path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        image = self._scene.render_to_image()
        if not o3d.io.write_image(str(self.output_path), image, 9):
            raise RuntimeError(f"Could not write render: {self.output_path}")
        return self.output_path

    def _shape_bounds(self):
        combined_mesh = o3d.geometry.TriangleMesh()
        for part in self.shape.partset:
            if part.label == "ground":
                continue
            model = self.name_to_geom.get(f"{part.part_index}_mesh")
            if model is None:
                continue
            for mesh_info in model.meshes:
                combined_mesh += mesh_info.mesh

        if combined_mesh.is_empty():
            raise ValueError("Cannot render a shape without mesh geometry")
        return combined_mesh.get_axis_aligned_bounding_box()

    def _configure_scene(self):
        scene = self._scene.scene
        scene.set_background([0.9, 0.9, 0.9, 1.0])
        scene.show_skybox(False)
        scene.show_axes(False)
        scene.scene.enable_indirect_light(True)
        scene.scene.set_indirect_light_intensity(45_000)
        scene.scene.set_sun_light(
            [0.707, 0.0, 0.707],
            [1.0, 1.0, 1.0],
            750_000,
        )
        scene.scene.enable_sun_light(True)
        scene.set_lighting(
            scene.LightingProfile.DARK_SHADOWS,
            (-0.577, -0.577, -0.077),
        )

    def _add_ground(self, bounds):
        extent = bounds.get_extent()
        ground_size = max(float(extent[0]), float(extent[2]), 1.0) * 20.0
        thickness = max(float(np.linalg.norm(extent)) * 0.005, 0.01)
        minimum = bounds.get_min_bound()
        center = bounds.get_center()

        ground = o3d.geometry.TriangleMesh.create_box(
            ground_size,
            thickness,
            ground_size,
            create_uv_map=True,
            map_texture_to_each_face=True,
        )
        ground.translate(
            (
                center[0] - ground_size / 2,
                minimum[1] - thickness,
                center[2] - ground_size / 2,
            )
        )
        ground.compute_vertex_normals()

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = [0.25, 0.5, 0.25, 1.0]
        self._scene.scene.add_geometry("ground", ground, material)

    def _add_active_geometry(self):
        for name, visible in self.switch_board.items():
            if not visible or name not in self.name_to_geom:
                continue
            geometry = self.name_to_geom[name]
            if isinstance(geometry, rendering.TriangleMeshModel):
                self._scene.scene.add_model(name, geometry)
            else:
                material = rendering.MaterialRecord()
                material.shader = "defaultLit"
                self._scene.scene.add_geometry(name, geometry, material)

    def _position_camera(self, bounds):
        center = bounds.get_center()
        distance = max(float(np.max(bounds.get_extent())), 1e-3) * 1.25
        direction = np.array([1.0, 0.6, 1.0])
        direction /= np.linalg.norm(direction)
        eye = center + direction * distance
        self._scene.scene.camera.look_at(center, eye, [0.0, 1.0, 0.0])
