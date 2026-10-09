import ast
import re

from ..edit_dsl.edits import KeepFixed
from ..edit_dsl.language import (
    AxisScale,
    PlaneScale,
    Rotate,
    Scale,
    Shear,
    SymGroupEdit,
    Translate,
    VolumeScale,
)
from ..shape_system.geometry import (
    BackToFrontLine,
    BackUnitVector,
    DownToUpLine,
    DownUnitVector,
    FrontToBackLine,
    FrontUnitVector,
    LeftToRightLine,
    LeftUnitVector,
    Line,
    Plane,
    RightToLeftLine,
    RightUnitVector,
    UpToDownLine,
    UpUnitVector,
)
from ..shape_system.relations import (
    FaceContact,
    FeatureRelation,
    PartRelation,
    PointContact,
    PrimitiveRelation,
    ReflectionSymmetry,
    RotationSymmetry,
    TranslationSymmetry,
)
from ..shape_system.shape import FaceFeature, Hexahedron, Part, PointFeature
from .errors import PromptSyntaxError


_GEOMETRIC_ITEMS = [
    Plane,
    Line,
    Hexahedron,
    Part,
    PointFeature,
    FaceFeature,
    PartRelation,
    FeatureRelation,
    ReflectionSymmetry,
    RotationSymmetry,
    TranslationSymmetry,
    PointContact,
    FaceContact,
    FrontUnitVector,
    BackUnitVector,
    LeftUnitVector,
    RightUnitVector,
    UpUnitVector,
    DownUnitVector,
    LeftToRightLine,
    FrontToBackLine,
    DownToUpLine,
    RightToLeftLine,
    BackToFrontLine,
    UpToDownLine,
]
GEOMETRIC_ATOMS = {item.__name__: item for item in _GEOMETRIC_ITEMS}

_EDIT_ITEMS = [
    Translate,
    Rotate,
    AxisScale,
    PlaneScale,
    VolumeScale,
    Scale,
    Shear,
    SymGroupEdit,
    KeepFixed,
]
EDIT_ATOMS = {item.__name__: item for item in _EDIT_ITEMS}

_PYTHON_BLOCK = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)
_ALLOWED_ATTRIBUTES = {
    "center",
    "corner",
    "direction",
    "edge",
    "edge_center",
    "face",
    "face_center",
    "get",
    "get_relation",
}
_ALLOWED_NODES = {
    ast.Add,
    ast.Assign,
    ast.Attribute,
    ast.BinOp,
    ast.Call,
    ast.Constant,
    ast.Dict,
    ast.Div,
    ast.Expr,
    ast.keyword,
    ast.List,
    ast.Load,
    ast.MatMult,
    ast.Module,
    ast.Mult,
    ast.Name,
    ast.Store,
    ast.Sub,
    ast.Tuple,
    ast.UAdd,
    ast.UnaryOp,
    ast.USub,
}


def response_to_snippet(response):
    """Join the Python blocks returned by the prompting model."""
    return "\n".join(_PYTHON_BLOCK.findall(response))


class _EditProgramValidator(ast.NodeVisitor):
    def generic_visit(self, node):
        if type(node) not in _ALLOWED_NODES:
            raise PromptSyntaxError(
                f"Unsupported syntax in model output: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Module(self, node):
        if not all(isinstance(statement, ast.Assign) for statement in node.body):
            raise PromptSyntaxError(
                "Model output may contain assignments only."
            )
        self.generic_visit(node)

    def visit_Assign(self, node):
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id.startswith("_")
        ):
            raise PromptSyntaxError(
                "Model output may assign only to ordinary variable names."
            )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr not in _ALLOWED_ATTRIBUTES:
            raise PromptSyntaxError(
                f"Attribute access is not allowed: {node.attr}"
            )
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("_"):
            raise PromptSyntaxError(f"Name is not allowed: {node.id}")
        self.generic_visit(node)


def snippet_to_variable(snippet, shape, variable_name):
    """Evaluate the small assignment-only edit DSL emitted by the model."""
    try:
        tree = ast.parse(snippet, mode="exec")
        _EditProgramValidator().visit(tree)
        scope = {
            "__builtins__": {},
            "shape": shape,
            **GEOMETRIC_ATOMS,
            **EDIT_ATOMS,
        }
        outputs = {}
        exec(compile(tree, "<parsel-prompt>", "exec"), scope, outputs)
    except PromptSyntaxError:
        raise
    except Exception as error:
        raise PromptSyntaxError(str(error)) from error

    if variable_name not in outputs:
        raise PromptSyntaxError(
            f"Model output did not assign `{variable_name}`."
        )
    return outputs[variable_name]


def response_to_variable(response, shape, variable_name):
    return snippet_to_variable(
        response_to_snippet(response),
        shape,
        variable_name,
    )
