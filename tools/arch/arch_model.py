"""What the checkout contains, read with `ast` and nothing else.

Nothing here imports the code it describes: a model is built from source text, so it costs
no CUDA, no EGL and no conda env, and a module that cannot even be imported still shows up.

The model is a tree -- repo, package, module, class, function/method -- plus the edges the
source states outright: which module imports which, which scope constructs which class,
which scope calls which module-level function, and which class inherits from which. Those
four are what a reader would follow by hand; nothing is inferred beyond them, so a node or
an edge on the picture is a line of code somebody can open.
"""
import ast
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Directories never walked into: build artefacts, vendored code, and kamerdyner's `data/` --
# real users' memory and logs, which must never end up embedded in a page.
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".idea", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".pyright",
    "node_modules", "build", "dist", "venv", ".venv", "vendor", "data",
}

# Kinds, in the order they nest.
KIND_REPO = "repo"
KIND_PACKAGE = "package"
KIND_MODULE = "module"
KIND_CLASS = "class"
KIND_FUNCTION = "function"
KIND_METHOD = "method"
KIND_FOLDER = "folder"
KIND_FILE = "file"

# Data files worth seeing beside the code: configs, calibrations, label lists.
DATA_EXTENSIONS = (".json", ".csv")
# Past this a file is data rather than configuration, and its shape is not read.
MAX_READ_BYTES = 2 * 1024 * 1024

EDGE_IMPORTS = "imports"
EDGE_CONSTRUCTS = "constructs"
EDGE_CALLS = "calls"
EDGE_INHERITS = "inherits"


@dataclass
class Node:
    """One thing on the picture. `id` is stable across runs, which is what a diff needs."""
    id: str
    kind: str
    name: str
    parent: Optional[str]
    repo: str
    file: Optional[str] = None
    line: int = 0
    end_line: int = 0
    doc: str = ""
    signature: str = ""
    signature_short: str = ""
    signature_typed: str = ""
    bases: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fingerprint: str = ""
    # Annotation name as the signature writes it -> the node it names, for the ones this
    # checkout defines. A type is the one thing in a signature that is also a box on the
    # picture, so the page can let a reader step from `model_config: ModelConfig` to the
    # class itself rather than go looking for it by name.
    types: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id, "kind": self.kind, "name": self.name, "parent": self.parent,
            "repo": self.repo, "file": self.file, "line": self.line,
            "end_line": self.end_line, "doc": self.doc, "signature": self.signature,
            "signature_short": self.signature_short,
            "signature_typed": self.signature_typed, "bases": self.bases,
            "flags": self.flags, "fingerprint": self.fingerprint,
        }
        # Most defs name nothing this checkout defines; an empty map on every one of them
        # is page weight for nothing.
        if self.types:
            payload["types"] = self.types
        return payload


@dataclass
class Edge:
    source: str
    target: str
    kind: str
    line: int

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "target": self.target, "kind": self.kind,
                "line": self.line}


def _is_test_function(name: str, decorators: list[ast.expr]) -> bool:
    """A pytest test or fixture, wherever it sits.

    `pytest.ini` here sets `python_files = *.py`, so tests are not confined to a tests
    folder: `ordered_neighbors.py` and `record_multiple_lidars.py` carry theirs beside the
    code they exercise. A picture of the system has no use for them -- they call
    everything and are called by nothing.
    """
    if name == "test" or name.startswith("test_"):
        return True
    for decorator in decorators:
        written = _dotted_name(decorator)
        if written is None and isinstance(decorator, ast.Call):
            written = _dotted_name(decorator.func)
        if written is not None and ("pytest" in written or "fixture" in written):
            return True
    return False


def _first_doc_line(node: ast.AST) -> str:
    """The docstring's first line, which is the one sentence a node is worth on a picture."""
    docstring = ast.get_docstring(node)
    if docstring is None:
        return ""
    stripped = docstring.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def _fingerprint(source_lines: list[str], node: ast.AST) -> str:
    """A hash of the node's own text, so a diff can tell a moved node from a changed one."""
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    segment = "".join(source_lines[start:end])
    digest = hashlib.sha1(segment.encode("utf-8", errors="replace"))
    return digest.hexdigest()[:12]


def _format_argument(argument: ast.arg, default: Optional[ast.expr]) -> str:
    """One parameter as the source declares it: name, annotation, default."""
    text = argument.arg
    if argument.annotation is not None:
        annotation = ast.unparse(argument.annotation)
        text = f"{text}: {annotation}"
    if default is None:
        return text
    rendered = ast.unparse(default)
    if argument.annotation is None:
        return f"{text}={rendered}"
    return f"{text} = {rendered}"


def _signature(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The def line, which is what a caller has to satisfy and a reader wants first."""
    arguments = statement.args
    positional = list(arguments.posonlyargs) + list(arguments.args)
    missing = len(positional) - len(arguments.defaults)
    defaults: list[Optional[ast.expr]] = [None] * missing
    defaults.extend(arguments.defaults)
    parts: list[str] = []
    for index, argument in enumerate(positional):
        rendered = _format_argument(argument, defaults[index])
        parts.append(rendered)
        if arguments.posonlyargs and index == len(arguments.posonlyargs) - 1:
            parts.append("/")
    if arguments.vararg is not None:
        rendered = _format_argument(arguments.vararg, None)
        parts.append(f"*{rendered}")
    elif arguments.kwonlyargs:
        parts.append("*")
    for index, argument in enumerate(arguments.kwonlyargs):
        rendered = _format_argument(argument, arguments.kw_defaults[index])
        parts.append(rendered)
    if arguments.kwarg is not None:
        rendered = _format_argument(arguments.kwarg, None)
        parts.append(f"**{rendered}")
    joined = ", ".join(parts)
    keyword = "async def" if isinstance(statement, ast.AsyncFunctionDef) else "def"
    head = f"{keyword} {statement.name}({joined})"
    if statement.returns is None:
        return head
    returns = ast.unparse(statement.returns)
    return f"{head} -> {returns}"


def _drawn_signature(statement: ast.FunctionDef | ast.AsyncFunctionDef,
                     with_types: bool) -> str:
    """The signature as a drawn box carries it: defaults dropped, types optional.

    A default is the caller's business and is what the details panel's full signature is
    for. `self` goes too: on a picture the box already says which class it is, and every
    line starting with the same four characters costs width the parameters need. Types are
    the half a reader asks for one function at a time, so the page keeps both renderings
    and swaps them per box rather than choosing once for everything.
    """
    def render(argument: ast.arg) -> str:
        if not with_types or argument.annotation is None:
            return argument.arg
        annotation = ast.unparse(argument.annotation)
        return f"{argument.arg}: {annotation}"

    arguments = statement.args
    positional = list(arguments.posonlyargs) + list(arguments.args)
    parts: list[str] = []
    for index, argument in enumerate(positional):
        if index == 0 and argument.arg in ("self", "cls"):
            continue
        rendered = render(argument)
        parts.append(rendered)
    if arguments.vararg is not None:
        rendered = render(arguments.vararg)
        parts.append(f"*{rendered}")
    for argument in arguments.kwonlyargs:
        rendered = render(argument)
        parts.append(rendered)
    if arguments.kwarg is not None:
        rendered = render(arguments.kwarg)
        parts.append(f"**{rendered}")
    joined = ", ".join(parts)
    head = f"{statement.name}({joined})"
    if statement.returns is None:
        return head
    returns = ast.unparse(statement.returns)
    return f"{head} -> {returns}"


def _names_in(expression: ast.expr) -> list[str]:
    """Every dotted name an annotation expression mentions, outermost first.

    `Optional[ModelConfig]` names two things and only one of them is in this checkout, so
    the whole expression is walked rather than read as a single name. A string annotation
    is a forward reference -- `"TunnelNode"` -- and is parsed rather than skipped.
    """
    written = _dotted_name(expression)
    if written is not None:
        return [written]
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        try:
            parsed = ast.parse(expression.value, mode="eval")
        except SyntaxError:
            return []
        return _names_in(parsed.body)
    found: list[str] = []
    for child in ast.iter_child_nodes(expression):
        if isinstance(child, ast.expr):
            found.extend(_names_in(child))
    return found


def _annotation_names(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The names the def's annotations mention, parameters and return alike, deduplicated."""
    arguments = statement.args
    annotated = list(arguments.posonlyargs) + list(arguments.args) \
        + list(arguments.kwonlyargs)
    if arguments.vararg is not None:
        annotated.append(arguments.vararg)
    if arguments.kwarg is not None:
        annotated.append(arguments.kwarg)
    expressions = [argument.annotation for argument in annotated
                   if argument.annotation is not None]
    if statement.returns is not None:
        expressions.append(statement.returns)
    names: list[str] = []
    for expression in expressions:
        for written in _names_in(expression):
            if written not in names:
                names.append(written)
    return names


def _signature_short(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return _drawn_signature(statement, with_types=False)


def _signature_typed(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return _drawn_signature(statement, with_types=True)


def _size_text(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        kilobytes = size_bytes / 1024
        return f"{kilobytes:.1f} kB"
    megabytes = size_bytes / (1024 * 1024)
    return f"{megabytes:.1f} MB"


def _json_shape(content: Any) -> tuple[str, str]:
    if isinstance(content, dict):
        keys = list(content.keys())
        shown = keys[:14]
        joined = ", ".join(shown)
        tail = ", ..." if len(keys) > len(shown) else ""
        return "{ " + joined + tail + " }", f"{len(keys)} top-level keys"
    if isinstance(content, list):
        return f"[ {len(content)} items ]", "a list"
    return type(content).__name__, ""


def _data_summary(path: str) -> tuple[str, str]:
    """What a data file holds, as far as its first line or its top level says.

    Enough to recognise it from the tree -- which keys a config has, which columns a csv
    carries -- and never the contents themselves: this is a map, not a viewer.
    """
    size_bytes = os.path.getsize(path)
    size = _size_text(size_bytes)
    if size_bytes > MAX_READ_BYTES:
        return "", f"{size} · not read"
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8", errors="replace") as data_file:
            try:
                content = json.load(data_file)
            except json.JSONDecodeError as error:
                return "", f"{size} · unparsed ({error.msg} at line {error.lineno})"
        shape, note = _json_shape(content)
        return shape, f"{size} · {note}" if note else size
    with open(path, "r", encoding="utf-8", errors="replace") as data_file:
        lines = data_file.read().splitlines()
    if not lines:
        return "", f"{size} · empty"
    columns = lines[0].split(",")
    shown = columns[:8]
    joined = ", ".join(shown)
    tail = ", ..." if len(columns) > len(shown) else ""
    return joined + tail, f"{size} · {len(lines) - 1} rows, {len(columns)} columns"


def _dotted_name(node: ast.AST) -> Optional[str]:
    """`a.b.C` for a Name/Attribute chain, None for anything else (a call on a call, ...)."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


class _ModuleReader:
    """One module's nodes, its local name -> fully qualified symbol table, and its edges.

    Edges are recorded against the symbol names the source used and resolved to node ids
    afterwards, once every module has been read: a class may well be defined in a module
    parsed later than the one constructing it.
    """

    def __init__(self, module_fq: str, repo: str, path: str, source: str) -> None:
        self.module_fq = module_fq
        self.repo = repo
        self.path = path
        self.source_lines = source.splitlines(keepends=True)
        self.nodes: list[Node] = []
        self.symbols: dict[str, str] = {}
        # (source node id, target symbol as written, edge kind, line)
        self.pending: list[tuple[str, str, str, int]] = []
        # (def node id, annotation name as the signature writes it)
        self.pending_types: list[tuple[str, str]] = []
        self._module_id = module_id(module_fq)

    def read(self, tree: ast.Module) -> None:
        self._read_symbols(tree)
        self._read_nodes(tree)
        self._read_edges(tree, self._module_id)

    # ---------------------------------------------------------------- symbols
    def _package_fq(self, level: int) -> str:
        """What `from .` / `from ..` means here, in absolute terms."""
        is_package_init = self.path.endswith(os.sep + "__init__.py")
        if is_package_init:
            own_package = self.module_fq
        else:
            own_package = self.module_fq.rpartition(".")[0]
        parts = own_package.split(".") if own_package else []
        upwards = level - 1
        if upwards > 0:
            parts = parts[:-upwards]
        return ".".join(parts)

    def _read_symbols(self, tree: ast.Module) -> None:
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname if alias.asname else alias.name.split(".")[0]
                    target = alias.name if alias.asname else alias.name.split(".")[0]
                    self.symbols[local] = target
                    self.symbols[alias.name] = alias.name
            elif isinstance(statement, ast.ImportFrom):
                base = statement.module if statement.module else ""
                if statement.level > 0:
                    package = self._package_fq(statement.level)
                    base = f"{package}.{base}" if base else package
                for alias in statement.names:
                    local = alias.asname if alias.asname else alias.name
                    self.symbols[local] = f"{base}.{alias.name}" if base else alias.name
            elif isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.symbols[statement.name] = f"{self.module_fq}.{statement.name}"

    # ------------------------------------------------------------------ nodes
    def _read_nodes(self, tree: ast.Module) -> None:
        module_flags = self._module_flags(tree)
        module_node = Node(
            id=self._module_id, kind=KIND_MODULE, name=self.module_fq.rpartition(".")[2],
            parent=None, repo=self.repo, file=self.path, line=1,
            end_line=len(self.source_lines), doc=_first_doc_line(tree), flags=module_flags,
            fingerprint=_fingerprint(self.source_lines, tree),
        )
        self.nodes.append(module_node)
        for statement in tree.body:
            if isinstance(statement, ast.ClassDef):
                self._read_class(statement, module_flags)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._read_function(statement, self._module_id, KIND_FUNCTION, module_flags)

    def _module_flags(self, tree: ast.Module) -> list[str]:
        flags: list[str] = []
        basename = os.path.basename(self.path)
        is_test = (basename.startswith("test_") or basename == "conftest.py"
                   or f"{os.sep}tests{os.sep}" in self.path)
        if is_test:
            flags.append("test")
        if basename.startswith("run_") or basename == "run.py":
            flags.append("entrypoint")
        for statement in tree.body:
            if not isinstance(statement, ast.If):
                continue
            condition = ast.dump(statement.test)
            if "__main__" in condition and "entrypoint" not in flags:
                flags.append("entrypoint")
        return flags

    def _read_class(self, statement: ast.ClassDef, module_flags: list[str]) -> None:
        bases: list[str] = []
        for base in statement.bases:
            name = _dotted_name(base)
            if name is not None:
                bases.append(name)
        flags = list(module_flags)
        for base in bases:
            if base.endswith("Process"):
                flags.append("process")
            if base.endswith("Protocol"):
                flags.append("protocol")
            if base.endswith("Thread"):
                flags.append("thread")
        for decorator in statement.decorator_list:
            name = _dotted_name(decorator)
            if name is not None and "dataclass" in name:
                flags.append("dataclass")
        if statement.name.startswith("Test") and "test" not in flags:
            flags.append("test")
        class_fq = f"{self.module_fq}.{statement.name}"
        declaration = f"class {statement.name}({', '.join(bases)})" if bases \
            else f"class {statement.name}"
        node = Node(
            id=class_id(class_fq), kind=KIND_CLASS, name=statement.name,
            parent=self._module_id, repo=self.repo, file=self.path, line=statement.lineno,
            end_line=statement.end_lineno, doc=_first_doc_line(statement),
            signature=declaration, signature_short=f"class {statement.name}", bases=bases,
            flags=flags, fingerprint=_fingerprint(self.source_lines, statement),
        )
        self.nodes.append(node)
        for base in bases:
            self.pending.append((node.id, base, EDGE_INHERITS, statement.lineno))
        for body_statement in statement.body:
            if isinstance(body_statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._read_function(body_statement, node.id, KIND_METHOD, flags,
                                    owner_fq=class_fq)

    def _read_function(self, statement: ast.FunctionDef | ast.AsyncFunctionDef,
                       parent_id: str, kind: str, inherited_flags: list[str],
                       owner_fq: Optional[str] = None) -> None:
        scope_fq = owner_fq if owner_fq is not None else self.module_fq
        function_fq = f"{scope_fq}.{statement.name}"
        flags = list(inherited_flags)
        is_test = _is_test_function(statement.name, statement.decorator_list)
        if is_test and "test" not in flags:
            flags.append("test")
        declaration = _signature(statement)
        compact = _signature_short(statement)
        typed = _signature_typed(statement)
        node = Node(
            id=function_id(function_fq), kind=kind, name=statement.name, parent=parent_id,
            repo=self.repo, file=self.path, line=statement.lineno,
            end_line=statement.end_lineno, doc=_first_doc_line(statement),
            signature=declaration, signature_short=compact, signature_typed=typed,
            flags=flags, fingerprint=_fingerprint(self.source_lines, statement),
        )
        self.nodes.append(node)
        for written in _annotation_names(statement):
            self.pending_types.append((node.id, written))

    # ------------------------------------------------------------------ edges
    def _read_edges(self, node: ast.AST, scope_id: str) -> None:
        """Walk the tree keeping track of the innermost def/class a call sits in.

        The scope is what an edge starts from, so `TunnelNode.__init__ -> UnifiedSource`
        reads as it does in the file rather than as one module pointing at another.
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                class_fq = f"{self.module_fq}.{child.name}"
                self._read_edges_in_class(child, class_id(class_fq), class_fq)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_fq = f"{self.module_fq}.{child.name}"
                self._read_edges(child, function_id(function_fq))
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                self._read_import_edge(child)
            else:
                if isinstance(child, ast.Call):
                    self._read_call_edge(child, scope_id)
                self._read_edges(child, scope_id)

    def _read_edges_in_class(self, statement: ast.ClassDef, scope_id: str,
                             class_fq: str) -> None:
        for child in statement.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_fq = f"{class_fq}.{child.name}"
                self._read_edges(child, function_id(method_fq))
            else:
                self._read_edges(child, scope_id)

    def _read_import_edge(self, statement: ast.Import | ast.ImportFrom) -> None:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                self.pending.append((self._module_id, alias.name, EDGE_IMPORTS,
                                     statement.lineno))
            return
        base = statement.module if statement.module else ""
        if statement.level > 0:
            package = self._package_fq(statement.level)
            base = f"{package}.{base}" if base else package
        for alias in statement.names:
            target = f"{base}.{alias.name}" if base else alias.name
            self.pending.append((self._module_id, target, EDGE_IMPORTS, statement.lineno))

    def _read_call_edge(self, call: ast.Call, scope_id: str) -> None:
        written = _dotted_name(call.func)
        if written is None:
            return
        self.pending.append((scope_id, written, EDGE_CALLS, call.lineno))


def module_id(module_fq: str) -> str:
    return f"mod:{module_fq}"


def package_id(package_fq: str) -> str:
    return f"pkg:{package_fq}"


def class_id(class_fq: str) -> str:
    return f"cls:{class_fq}"


def function_id(function_fq: str) -> str:
    return f"fn:{function_fq}"


def repo_id(repo: str) -> str:
    return f"repo:{repo}"


def folder_id(repo: str, relative: str) -> str:
    return f"dir:{repo}/{relative}"


def file_id(repo: str, relative: str) -> str:
    return f"file:{repo}/{relative}"


def _python_root(repo_dir: str) -> str:
    """Where this repo's importable names start: `src/` when it has one, else the repo."""
    src_dir = os.path.join(repo_dir, "src")
    if os.path.isdir(src_dir):
        return src_dir
    return repo_dir


def _module_fq_of(python_root: str, path: str) -> str:
    relative = os.path.relpath(path, python_root)
    without_extension = os.path.splitext(relative)[0]
    parts = without_extension.split(os.sep)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _python_files(python_root: str) -> list[str]:
    files: list[str] = []
    for folder, subfolders, names in os.walk(python_root):
        subfolders[:] = [name for name in subfolders if name not in EXCLUDED_DIRS]
        for name in names:
            if not name.endswith(".py"):
                continue
            files.append(os.path.join(folder, name))
    files.sort()
    return files


def _data_files(repo_dir: str) -> list[str]:
    """Every config-shaped file of a repo, wherever it sits -- inside src or beside it."""
    files: list[str] = []
    for folder, subfolders, names in os.walk(repo_dir):
        subfolders[:] = [name for name in subfolders if name not in EXCLUDED_DIRS]
        for name in names:
            if not name.endswith(DATA_EXTENSIONS):
                continue
            path = os.path.join(folder, name)
            if os.path.islink(path):
                continue
            files.append(path)
    files.sort()
    return files


def _repo_dirs(root: str) -> list[str]:
    """The git repos of a checkout root, which is what a repo node stands for.

    A root that is itself a repo is its own single repo node."""
    if os.path.exists(os.path.join(root, ".git")):
        return [root]
    dirs: list[str] = []
    for name in sorted(os.listdir(root)):
        candidate = os.path.join(root, name)
        if not os.path.isdir(candidate) or os.path.islink(candidate):
            continue
        git_dir = os.path.join(candidate, ".git")
        if os.path.exists(git_dir):
            dirs.append(candidate)
    assert dirs, f"Invalid input. No git repos found under {root}"
    return dirs


def _dirty_files(repo_dir: str) -> set[str]:
    """Files differing from HEAD, so the picture can show what this session touched."""
    completed = subprocess.run(
        ["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    )
    dirty: set[str] = set()
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        relative = raw_line[3:]
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        dirty.add(os.path.join(repo_dir, relative.strip()))
    return dirty


class ModelBuilder:
    """Reads a checkout root into nodes and edges. One instance per run."""

    def __init__(self, root: str, include_tests: bool) -> None:
        self.root = os.path.abspath(root)
        self.include_tests = include_tests
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.symbol_owner: dict[str, str] = {}
        self.module_owner: dict[str, str] = {}
        self.problems: list[str] = []

    def build(self) -> dict[str, Any]:
        readers: list[_ModuleReader] = []
        for repo_dir in _repo_dirs(self.root):
            readers.extend(self._read_repo(repo_dir))
        for reader in readers:
            for node in reader.nodes:
                if "test" in node.flags and not self.include_tests:
                    continue
                self._register(node)
        self._resolve_edges(readers)
        self._resolve_types(readers)
        self._mark_dirty()
        nodes = [node.as_dict() for node in self.nodes.values()]
        edges = [edge.as_dict() for edge in self.edges]
        return {"root": self.root, "nodes": nodes, "edges": edges,
                "problems": self.problems}

    # ------------------------------------------------------------------ repos
    def _read_repo(self, repo_dir: str) -> list[_ModuleReader]:
        repo = os.path.basename(repo_dir)
        repo_node = Node(id=repo_id(repo), kind=KIND_REPO, name=repo, parent=None,
                         repo=repo, file=repo_dir)
        self.nodes[repo_node.id] = repo_node
        python_root = _python_root(repo_dir)
        readers: list[_ModuleReader] = []
        for path in _python_files(python_root):
            module_fq = _module_fq_of(python_root, path)
            if not module_fq:
                continue
            reader = self._read_module(module_fq, repo, path)
            if reader is not None:
                readers.append(reader)
        for path in _data_files(repo_dir):
            self._read_data_file(repo, repo_dir, python_root, path)
        return readers

    def _read_data_file(self, repo: str, repo_dir: str, python_root: str,
                        path: str) -> None:
        relative = os.path.relpath(path, repo_dir)
        if not self.include_tests and f"{os.sep}tests{os.sep}" in path:
            return
        folder = os.path.dirname(path)
        parent = self._ensure_folder_chain(repo, repo_dir, python_root, folder)
        shape, note = _data_summary(path)
        extension = os.path.splitext(path)[1].lstrip(".")
        node = Node(id=file_id(repo, relative), kind=KIND_FILE,
                    name=os.path.basename(path), parent=parent, repo=repo, file=path,
                    doc=note, signature=shape, flags=[extension])
        self.nodes[node.id] = node

    def _folder_identifier(self, repo: str, python_root: str, folder: str) -> str:
        """A folder that is a package is the package node -- one row, not two."""
        relative = os.path.relpath(folder, python_root)
        if not relative.startswith(".."):
            package_fq = relative.replace(os.sep, ".")
            return package_id(package_fq)
        return folder_id(repo, os.path.relpath(folder, os.path.dirname(python_root)))

    def _ensure_folder_chain(self, repo: str, repo_dir: str, python_root: str,
                             folder: str) -> str:
        if folder == repo_dir or folder == python_root:
            return repo_id(repo)
        parent_folder = os.path.dirname(folder)
        parent = self._ensure_folder_chain(repo, repo_dir, python_root, parent_folder)
        identifier = self._folder_identifier(repo, python_root, folder)
        if identifier not in self.nodes:
            node = Node(id=identifier, kind=KIND_FOLDER, name=os.path.basename(folder),
                        parent=parent, repo=repo, file=folder)
            self.nodes[identifier] = node
        return identifier

    def _read_module(self, module_fq: str, repo: str,
                     path: str) -> Optional[_ModuleReader]:
        with open(path, "r", encoding="utf-8", errors="replace") as source_file:
            source = source_file.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as error:
            self.problems.append(f"{path}: cannot parse ({error})")
            return None
        reader = _ModuleReader(module_fq, repo, path, source)
        reader.read(tree)
        module_node = reader.nodes[0]
        if "test" in module_node.flags and not self.include_tests:
            return None
        is_package_init = path.endswith(os.sep + "__init__.py")
        if is_package_init:
            # A package's __init__ belongs inside that package, not beside it under the
            # same name: the two are one thing to a reader and two rows on a tree.
            module_node.name = "__init__"
            self._ensure_packages(f"{module_fq}.__init__", repo)
            module_node.parent = package_id(module_fq)
        else:
            module_node.parent = self._ensure_packages(module_fq, repo)
        owner = self.module_owner.get(module_fq)
        if owner is not None:
            self.problems.append(
                f"Module name {module_fq} is claimed by both {owner} and {path}; "
                f"edges naming it are attributed to the first")
        else:
            self.module_owner[module_fq] = path
        return reader

    def _ensure_packages(self, module_fq: str, repo: str) -> str:
        """Package nodes for every dotted level above a module, created once each."""
        parts = module_fq.split(".")
        parent = repo_id(repo)
        for depth in range(1, len(parts)):
            package_fq = ".".join(parts[:depth])
            identifier = package_id(package_fq)
            if identifier not in self.nodes:
                node = Node(id=identifier, kind=KIND_PACKAGE, name=parts[depth - 1],
                            parent=parent, repo=repo)
                self.nodes[identifier] = node
            parent = identifier
        return parent

    def _register(self, node: Node) -> None:
        self.nodes[node.id] = node
        prefix, _, symbol_fq = node.id.partition(":")
        if prefix in ("cls", "fn", "mod"):
            self.symbol_owner.setdefault(symbol_fq, node.id)

    # ------------------------------------------------------------------ edges
    def _resolve_edges(self, readers: list[_ModuleReader]) -> None:
        seen: set[tuple[str, str, str]] = set()
        for reader in readers:
            for source_id, written, kind, line in reader.pending:
                if source_id not in self.nodes:
                    continue
                if reader.module_fq not in self.module_owner:
                    continue
                target_id = self._resolve(written, reader)
                if target_id is None:
                    continue
                if kind == EDGE_IMPORTS:
                    target_id = self._module_of(target_id)
                if target_id is None or target_id == source_id:
                    continue
                resolved_kind = self._edge_kind(kind, target_id)
                if resolved_kind is None:
                    continue
                key = (source_id, target_id, resolved_kind)
                if key in seen:
                    continue
                seen.add(key)
                self.edges.append(Edge(source_id, target_id, resolved_kind, line))

    def _resolve_types(self, readers: list[_ModuleReader]) -> None:
        """Which of a def's annotations name a class this checkout defines.

        Not an edge: `def _track(boxes: list[Box])` says what a caller has to hand over,
        which is a weaker claim than constructing or calling one, and drawing it as an
        arrow would put a line between everything and its data classes. It is kept on the
        def instead, so the page can offer it where the type is written and nowhere else.
        """
        for reader in readers:
            for source_id, written in reader.pending_types:
                node = self.nodes.get(source_id)
                if node is None or written in node.types:
                    continue
                target_id = self._resolve(written, reader)
                if target_id is None or target_id == source_id:
                    continue
                if self.nodes[target_id].kind != KIND_CLASS:
                    continue
                node.types[written] = target_id

    def _module_of(self, node_id: str) -> Optional[str]:
        """The module a symbol lives in -- what `from x import Y` really names."""
        current: Optional[str] = node_id
        while current is not None:
            node = self.nodes[current]
            if node.kind in (KIND_MODULE, KIND_PACKAGE):
                return current
            current = node.parent
        return None

    def _edge_kind(self, written_kind: str, target_id: str) -> Optional[str]:
        """A call is a construction when it names a class, and a call when it names a def."""
        target = self.nodes[target_id]
        if written_kind == EDGE_IMPORTS:
            if target.kind in (KIND_MODULE, KIND_PACKAGE):
                return EDGE_IMPORTS
            return None
        if written_kind == EDGE_INHERITS:
            if target.kind == KIND_CLASS:
                return EDGE_INHERITS
            return None
        if target.kind == KIND_CLASS:
            return EDGE_CONSTRUCTS
        if target.kind in (KIND_FUNCTION, KIND_METHOD):
            return EDGE_CALLS
        return None

    def _resolve(self, written: str, reader: _ModuleReader) -> Optional[str]:
        """The node a written name points at: through the module's imports, then globally.

        Longest prefix first, so `tunnel_node.TunnelNode` resolves through the module the
        name `tunnel_node` was imported as before anything shorter is tried.
        """
        parts = written.split(".")
        for length in range(len(parts), 0, -1):
            prefix = ".".join(parts[:length])
            mapped = reader.symbols.get(prefix)
            if mapped is None:
                continue
            suffix = parts[length:]
            candidate = ".".join([mapped] + suffix) if suffix else mapped
            resolved = self._lookup(candidate)
            if resolved is not None:
                return resolved
            resolved = self._lookup(mapped)
            if resolved is not None:
                return resolved
        return self._lookup(written)

    def _lookup(self, symbol_fq: str) -> Optional[str]:
        owner = self.symbol_owner.get(symbol_fq)
        if owner is not None:
            return owner
        identifier = package_id(symbol_fq)
        if identifier in self.nodes:
            return identifier
        return None

    # ------------------------------------------------------------------ state
    def _mark_dirty(self) -> None:
        for repo_dir in _repo_dirs(self.root):
            dirty = _dirty_files(repo_dir)
            if not dirty:
                continue
            for node in self.nodes.values():
                if node.file in dirty:
                    node.flags.append("dirty")


def build_model(root: str, include_tests: bool) -> dict[str, Any]:
    builder = ModelBuilder(root, include_tests)
    return builder.build()


def _looks_like_test(node: dict[str, Any]) -> bool:
    """Whether a snapshot's record is one this run would leave out as a test.

    Read off the name rather than the flags: a snapshot written before this tool learnt
    to spot a `test_` beside ordinary code carries no flag saying so, and every one of
    those would otherwise read as a deletion.
    """
    if "test" in node["flags"]:
        return True
    name = node["name"]
    return name == "test" or name.startswith("test_") or name.startswith("Test")


def diff_against(model: dict[str, Any],
                 snapshot: Optional[dict[str, Any]]) -> dict[str, str]:
    """Per node id: added / changed / removed, against a snapshot of an earlier run.

    Removed ids are in the result too, and the page draws them as ghosts -- what a change
    deleted is exactly as interesting as what it added.
    """
    if snapshot is None:
        return {}
    previous: dict[str, str] = {}
    previous_kinds: set[str] = set()
    for node in snapshot["nodes"]:
        previous[node["id"]] = node["fingerprint"]
        previous_kinds.add(node["kind"])
    # What the reader wants coloured is what the code did, not what this tool started or
    # stopped showing. A kind the snapshot never held at all, and definitions the snapshot
    # held only because tests were still included, are the tool changing its mind -- and
    # both corrections disappear by themselves with the next saved snapshot.
    tests_dropped = True
    for node in model["nodes"]:
        if "test" in node["flags"]:
            tests_dropped = False
            break
    status: dict[str, str] = {}
    current_ids: set[str] = set()
    for node in model["nodes"]:
        identifier = node["id"]
        current_ids.add(identifier)
        if node["kind"] not in previous_kinds:
            continue
        if identifier not in previous:
            status[identifier] = "added"
        elif previous[identifier] != node["fingerprint"]:
            status[identifier] = "changed"
    for node in snapshot["nodes"]:
        if node["id"] in current_ids:
            continue
        if tests_dropped and _looks_like_test(node):
            continue
        status[node["id"]] = "removed"
    return status


def removed_nodes(model: dict[str, Any], snapshot: Optional[dict[str, Any]],
                  status: dict[str, str]) -> Iterable[dict[str, Any]]:
    """The snapshot's records for ids the current run no longer has."""
    if snapshot is None:
        return []
    current_ids = {node["id"] for node in model["nodes"]}
    ghosts: list[dict[str, Any]] = []
    for node in snapshot["nodes"]:
        if node["id"] in current_ids:
            continue
        if status.get(node["id"]) != "removed":
            continue
        ghosts.append(node)
    return ghosts
