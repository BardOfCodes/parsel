from importlib.resources import files


_DOCS = files(__package__) / "prompts" / "api_docs"


def _read(name):
    return (_DOCS / name).read_text(encoding="utf-8")


INIT_EDIT_API = _read("edits_api.md")
SHAPE_API = _read("shape_only.md")
EDIT_API = _read("edits_only.md")
