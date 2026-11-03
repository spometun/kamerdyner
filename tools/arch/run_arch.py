#!/usr/bin/env python3
"""Read a checkout root and write the interactive architecture page for it.

    tools/arch/run_arch.py                   # read this repo, write build/arch/architecture-kamerdyner.html
    tools/arch/run_arch.py --save-snapshot   # ... and make this run the baseline a diff is against
    tools/arch/run_arch.py --root DIR        # another root: a git repo or a folder of them

The page compares itself to the last saved snapshot: what a session added, changed or
deleted is coloured, and so is every file that differs from its repo's git HEAD. So the
loop is: save a snapshot before planning, look at the picture, decide, and run it again
afterwards to see what moved.

Stdlib only, and nothing here imports the code it reads -- it runs under any python 3.10+
on PATH, with no conda env and no GPU.
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arch_html
import arch_model

ARCH_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ARCH_DIR, "vendor")
# The repo this tool lives in (tools/arch/ inside it): the default root.
REPO_DIR = os.path.dirname(os.path.dirname(ARCH_DIR))
# Generated page and snapshots: inside build/, which git ignores and the reader never walks.
OUT_DIR = os.path.join(REPO_DIR, "build", "arch")
SNAPSHOT_DIR = os.path.join(OUT_DIR, "snapshots")


def _is_checkout_root(folder: str) -> bool:
    """A root is a git repo, or a folder holding git repos."""
    if not os.path.isdir(folder):
        return False
    if os.path.exists(os.path.join(folder, ".git")):
        return True
    for name in os.listdir(folder):
        if os.path.exists(os.path.join(folder, name, ".git")):
            return True
    return False


def _load_snapshot(path: str) -> Optional[dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as snapshot_file:
        return json.load(snapshot_file)


def _write_snapshot(path: str, model: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as snapshot_file:
        json.dump(model, snapshot_file)


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record[key]
        counts[value] = counts.get(value, 0) + 1
    return counts


def _summarize(counts: dict[str, int]) -> str:
    parts = []
    for name in sorted(counts):
        parts.append(f"{name} {counts[name]}")
    return ", ".join(parts)


def _generate(root: str, output: str, include_tests: bool, save_snapshot: bool,
              quiet: bool = False) -> str:
    """Build the model, write the page, optionally save the snapshot; return the page HTML."""
    root_name = os.path.basename(root)
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{root_name}.json")

    model = arch_model.build_model(root, include_tests)
    snapshot = _load_snapshot(snapshot_path)
    status = arch_model.diff_against(model, snapshot)
    ghosts = list(arch_model.removed_nodes(model, snapshot, status))

    node_counts = _counts(model["nodes"], "kind")
    edge_counts = _counts(model["edges"], "kind")
    status_counts = _counts([{"s": value} for value in status.values()], "s")
    dirty = 0
    for node in model["nodes"]:
        if "dirty" in node["flags"]:
            dirty += 1

    meta_parts = [root, _summarize(node_counts), _summarize(edge_counts)]
    if status:
        meta_parts.append("vs snapshot: " + _summarize(status_counts))
    if dirty:
        meta_parts.append(f"{dirty} nodes in files differing from git HEAD")
    meta = " · ".join(meta_parts)

    page = arch_html.render_page(model, status, f"{root_name} architecture", meta,
                                 VENDOR_DIR, ghosts)
    with open(output, "w", encoding="utf-8") as page_file:
        page_file.write(page)

    if save_snapshot:
        _write_snapshot(snapshot_path, model)

    if not quiet:
        print(f"root      {root}")
        print(f"nodes     {_summarize(node_counts)}")
        print(f"edges     {_summarize(edge_counts)}")
        if status:
            print(f"diff      {_summarize(status_counts)} (snapshot {snapshot_path})")
        elif snapshot is None:
            print(f"diff      no snapshot yet -- run with --save-snapshot to make one")
        else:
            print(f"diff      nothing changed since the snapshot")
        if dirty:
            print(f"git       {dirty} nodes live in files differing from HEAD")
        for problem in model["problems"]:
            print(f"problem   {problem}")
        print(f"page      {output}")
        if save_snapshot:
            print(f"snapshot  {snapshot_path}")
    return page


def _serve(root: str, output: str, include_tests: bool, port: int, open_browser: bool) -> None:
    """A file:// page can't regenerate on F5 -- there's nothing to run a reload against. This
    serves the same page over http:// instead: every GET rebuilds the model fresh (~0.4s for
    this checkout) and writes it to `output` same as a plain run, so F5 in the browser is the
    reload. --save-snapshot happens once here, at startup; afterwards the baseline moves only
    when the page's "reset colours" button POSTs to /reset-diff."""
    import http.server
    import importlib

    def rebuild(save_snapshot: bool) -> str:
        """Re-import the two modules first, so editing the renderer needs no restart either.

        Without this a reload answers with the freshly read repos rendered by whatever
        arch_html was on disk when the server started -- the confusing half of a live page.
        Only this file itself still needs a restart to take effect.
        """
        importlib.reload(arch_model)
        importlib.reload(arch_html)
        return _generate(root, output, include_tests, save_snapshot, quiet=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            body = rebuild(save_snapshot=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            """Take the current code as the new baseline, so the next GET colours nothing."""
            if self.path != "/reset-diff":
                self.send_error(404)
                return
            rebuild(save_snapshot=True)
            print("reset     baseline moved to the current code")
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *log_args: Any) -> None:
            print("served    " + (format % log_args))

    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"serving   {url}  (every GET regenerates -- F5 in the browser refreshes the picture)")
    print(f"page      {output}  (kept up to date on disk too, same path a plain run writes)")
    if open_browser:
        subprocess.run(["xdg-open", url], check=False)
    httpd.serve_forever()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None,
                        help="Root to read: a git repo or a folder of them (default: the "
                             "repo this tool lives in)")
    parser.add_argument("--output", default=None,
                        help="Page to write (default: build/arch/architecture-<root>.html)")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Make this run the baseline later runs are diffed against")
    parser.add_argument("--include-tests", action="store_true",
                        help="Keep test modules, which are left out by default")
    parser.add_argument("--open", action="store_true",
                        help="Open the page in the desktop browser when it is written")
    parser.add_argument("--serve", action="store_true",
                        help="Serve the page over http:// instead of writing it once; every "
                             "browser reload (F5) regenerates it. A file:// page can't do this "
                             "on its own -- there is no process behind it to rerun.")
    parser.add_argument("--port", type=int, default=8770,
                        help="Port for --serve (default: 8770)")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root) if args.root else REPO_DIR
    assert _is_checkout_root(root), f"Invalid input. {root} holds no git repos"
    root_name = os.path.basename(root)
    output = args.output
    if output is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        output = os.path.join(OUT_DIR, f"architecture-{root_name}.html")

    if args.serve:
        if args.save_snapshot:
            _generate(root, output, args.include_tests, save_snapshot=True)
        _serve(root, output, args.include_tests, args.port, args.open)
        return

    _generate(root, output, args.include_tests, args.save_snapshot)
    if args.open:
        subprocess.run(["xdg-open", output], check=False)


if __name__ == "__main__":
    main()
