#!/usr/bin/env python3
"""Answer "who uses this / what does this use" off a generated architecture page.

    ./arch_query.py memory                      # match by name, any kind
    ./arch_query.py pkg:kamerdyner.memory       # or by exact node id
    ./arch_query.py --page ../../build/arch/architecture-kamerdyner.html Chat

A subtree is the node plus everything under it, so asking about a package answers for the
package: incoming edges are its real consumers, outgoing edges are what a move elsewhere
would drag along. Reads the `const MODEL = {...}` embedded in the page -- no browser, no
env, stdlib only. Regenerate the page with run_arch.py first if the code has moved since.
"""
import argparse
import json
import os
import re
import sys
from typing import Any, Optional

ARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATTERN = re.compile(r"const MODEL = (\{.*?\});\n", re.DOTALL)


def _default_page() -> str:
    """The page run_arch.py writes by default: that of the repo this tool lives in."""
    repo_dir = os.path.dirname(os.path.dirname(ARCH_DIR))
    root_name = os.path.basename(repo_dir)
    return os.path.join(repo_dir, "build", "arch", f"architecture-{root_name}.html")


def load_model(page: str) -> dict[str, Any]:
    with open(page, "r", encoding="utf-8") as page_file:
        text = page_file.read()
    match = _MODEL_PATTERN.search(text)
    assert match is not None, \
        f"Invalid input. {page} holds no 'const MODEL = ...' line; is it an arch page?"
    return json.loads(match.group(1))


def resolve(model: dict[str, Any], wanted: str) -> list[dict[str, Any]]:
    """Exact id first -- a name can be ambiguous, an id never is."""
    by_id = {node["id"]: node for node in model["nodes"]}
    if wanted in by_id:
        return [by_id[wanted]]
    matches = []
    for node in model["nodes"]:
        if node["name"] == wanted:
            matches.append(node)
    return matches


def subtree(model: dict[str, Any], root_id: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for node in model["nodes"]:
        parent = node["parent"]
        if parent is None:
            continue
        if parent not in children:
            children[parent] = []
        children[parent].append(node["id"])
    inside = set()
    pending = [root_id]
    while pending:
        current = pending.pop()
        inside.add(current)
        for child in children.get(current, []):
            pending.append(child)
    return inside


def _describe(model: dict[str, Any], node_id: str) -> str:
    for node in model["nodes"]:
        if node["id"] == node_id:
            if node["file"]:
                return f"{node_id}  ({node['file']}:{node['line']})"
            return node_id
    return node_id


def report(model: dict[str, Any], node: dict[str, Any], show_files: bool) -> None:
    inside = subtree(model, node["id"])
    incoming = []
    outgoing = []
    for edge in model["edges"]:
        source_in = edge["source"] in inside
        target_in = edge["target"] in inside
        if target_in and not source_in:
            incoming.append(edge)
        elif source_in and not target_in:
            outgoing.append(edge)

    where = f" -- {node['file']}:{node['line']}" if node["file"] else ""
    print(f"{node['id']}  [{node['kind']} {node['name']}]{where}")
    print(f"subtree   {len(inside)} nodes")

    print(f"\nincoming  {len(incoming)}  (outside -> here: the consumers)")
    for edge in sorted(incoming, key=lambda e: (e["kind"], e["source"])):
        target = _describe(model, edge["target"]) if show_files else edge["target"]
        print(f"  {edge['kind']:<10} {edge['source']}  ->  {target}")
    if not incoming:
        print("  (none -- nothing outside this subtree names it)")

    print(f"\noutgoing  {len(outgoing)}  (here -> outside: what it depends on)")
    for edge in sorted(outgoing, key=lambda e: (e["kind"], e["target"])):
        print(f"  {edge['kind']:<10} {edge['source']}  ->  {edge['target']}")
    if not outgoing:
        print("  (none)")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="Node id (e.g. pkg:kamerdyner.memory) or a bare name")
    parser.add_argument("--page", default=None,
                        help="Architecture page to read (default: build/arch/architecture-<repo>.html)")
    parser.add_argument("--files", action="store_true",
                        help="Show file:line beside each consumer")
    args = parser.parse_args(argv)

    page = args.page
    if page is None:
        page = _default_page()
    page = os.path.abspath(page)
    assert os.path.exists(page), \
        f"Invalid input. No page at {page}; run run_arch.py to write one"

    model = load_model(page)
    matches = resolve(model, args.target)
    if not matches:
        print(f"no node named {args.target!r} in {page}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"{args.target!r} is ambiguous -- pass one of these ids:", file=sys.stderr)
        for node in matches:
            print(f"  {node['id']}  [{node['kind']}]", file=sys.stderr)
        sys.exit(1)

    report(model, matches[0], args.files)


if __name__ == "__main__":
    main()
