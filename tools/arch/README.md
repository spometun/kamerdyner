# arch — the code as a picture you can fold

`run_arch.py` reads the repo with `ast` and writes one self-contained HTML page: the packages
as a tree you can fold, a graph of what constructs/calls/imports what, and a diff against the
last snapshot so a session's changes are visible rather than remembered.

Nothing here imports the code it reads, so it needs no conda env and no network — just a
`python3` on PATH. The page and the snapshots go to `build/arch/`, which git ignores and the
reader never walks. `data/` (real users' memory and logs) is never read, so no user data ends
up in a page. Brought over from the cleantrak checkout; there a root is a folder of several
repos, here the repo itself is the root — both still work (`--root`).

## Use

```bash
tools/arch/run_arch.py --save-snapshot   # baseline before planning
xdg-open build/arch/architecture-kamerdyner.html
# ... plan, decide, work ...
tools/arch/run_arch.py                   # same command; the page now colours what moved
```

The page remembers what you had open (localStorage), so a regeneration comes back to the
same view rather than to the top level.

Flags: `--root DIR` (default: the repo this tool lives in), `--output FILE`, `--include-tests`
(reads only under `src/`, so it changes nothing here), `--open`, `--save-snapshot`.

## Live mode (F5 regenerates)

A `file://` page has nothing behind it to rerun, so reloading it just reloads the same bytes
off disk. `--serve` instead puts a tiny local HTTP server in front of the same page: every GET
rebuilds the model fresh (~0.1s for this repo) before answering, and still writes it to
`build/arch/architecture-kamerdyner.html`.

```bash
tools/arch/run_arch.py --serve --open   # first time: opens the browser tab
```

Then bookmark `http://127.0.0.1:8770/` (or whatever `--port` you passed) instead of the
`file://...html` path -- F5 there re-reads the code and re-renders. Leave the command running
in a terminal for as long as you want the live page; `Ctrl-C` stops it, after which the
last-written file on disk is still openable as a plain static page. `--save-snapshot` with
`--serve` takes the baseline once at startup, not on every reload.

Each request re-imports `arch_model` and `arch_html` too, so editing the reader or the page
itself also lands on F5. Only `run_arch.py` -- the server around them -- needs a restart to
take effect.

## Asking it questions from the terminal

`arch_query.py` answers the one question the page is most often opened for -- who uses this,
and what does it use -- without a browser:

```bash
tools/arch/arch_query.py memory --files   # a package: consumers of the whole subtree
tools/arch/arch_query.py Chat             # a class; zero incoming means nothing builds it
```

It reads the `const MODEL = {...}` embedded in the page, so it is only as fresh as the last
`run_arch.py`. `incoming` is what would break if the thing went away; `outgoing` is what a move
elsewhere would have to bring along. A bare name that matches several nodes is refused with
the candidate ids rather than guessed at.

## Reading the page

**Fold.** A node you have not opened stands for everything under it, and an edge is drawn
between the nearest opened ancestors of its two ends — so `apps → chat` at the top
level is the same data as every class-to-class edge between them further down. The
number beside a name is how many things are inside it. `open to repos/packages/modules/
classes` folds everything to one level at once; double-click opens one node, right-click
closes its parent, the tree on the left is the same state.

**Edges.** `constructs` (blue) — a scope calls a class: the pipeline's real shape.
`calls` (green) — a scope calls a module-level function. `imports` (dashed) — module to
module, the loosest of the four. `inherits` (purple). Constructs is on by default because
it is the one that reads like architecture; imports is the one that turns into fog.

**Focus** answers a different question from the overview. Instead of the folded boxes, it
shows the selected symbol's own neighbourhood N hops out, with a method's edges credited
to its class (`Chat.receive → Message` reads as `Chat → Message`).
A node that everything touches — a shared value type, a logger — is shown but never walked
through, otherwise one hop drags in the whole checkout.

**Signatures.** The selected class or module hangs each of its definitions below itself as
its own box, signature and all, while what it *uses* goes off to the right -- so one picture
says both what a thing is made of and what it depends on. Private definitions (one leading
underscore, not two) collapse into a `+ N private` box; click it to open them.
`signatures in graph` turns the lower branch off.

**Types, one def at a time.** A hung box shows parameter names only, and ends in a `·` when
it is holding types back. Click it and that one box rewrites itself with them
(`_track(bboxes: list, timestamp: float)`); click again to fold them away. It is per box on
purpose -- types are what you want for the function you are asking about and what turns the
other twenty into a wall -- and which boxes are open is remembered with the rest of the view.
Defaults are never drawn; the details panel on the right carries the full signature.

A type that names a class of this checkout is somewhere to go, not just more text. Opening a
box's types hangs each such class beside it as a small box of its own -- `__init__(layout:
TunnelLayout, model_config: ModelConfig)` grows a `TunnelLayout` and a `ModelConfig` at its
right -- and clicking one makes it the selection, so the picture re-renders around it.

**Ctrl+click a type where it is written.** The name inside the signature works too: hold
Ctrl and the word under the pointer says where it goes (`→ ModelConfig`), and clicking
follows it. The graph is a canvas -- a word in a label cannot be underlined the way the
panel's can -- so Ctrl is what tells the click it is aimed at a word rather than at the
box, and a Ctrl+click that lands on anything else does nothing rather than folding the
types away. The same names are underlined and clickable in the details panel's signatures.

This is deliberately not an edge: `def _track(boxes: list[Box])` says what a caller has to
hand over, which is a weaker claim than constructing one, and drawing it as an arrow would
put a line between everything and its data classes.

To make a def the subject instead -- the picture then answers what *it* calls, with its own
edges drawn whatever the edge buttons say -- click its row in the details panel. (A click in
the graph is spent on types now, so it is the panel that navigates.)

**Getting back.** Every selection is remembered: the arrows at the top left, Alt+Left /
Alt+Right, and the mouse's own back/forward buttons walk the trail.

**Diff.** Green = added since the snapshot, amber = changed body, dashed red = deleted
(drawn as a ghost), purple = the file differs from its repo's git HEAD. `changed only`
hides everything else.

`reset colours` starts the diff over: it makes the code as it stands the new baseline, so
everything goes back to one colour and the next change is measured from here. Under `--serve`
the button does it and reloads; otherwise it is `run_arch.py --save-snapshot`. Note that purple
is not part of this — it comes from git, not from the snapshot, and clears when you commit.

The details panel gives the docstring's first line, `file:line` (click to copy),
`copy for prompt` (a `[architecture-<root>.html node:<id>] ` tag, then the signature plus
`file:line`, no prose — the thing to paste when telling Claude what to work on: the tag says
which page and which node, so Claude opens that file and reads its own edges instead of
guessing from the name), and the full signatures of everything the selection sits among:
public first, then private, every row copyable and one click away from becoming the subject
itself.

## What it cannot see

It reads names, not behaviour. A class reached through a factory that picks it by string,
a callback handed over as a value, an object built in one place and used everywhere — none
of that is an edge here (a type annotation naming one is at least a link, see above). A `Foo()` whose `Foo` came from outside the repo is dropped
rather than guessed at. Two modules with the same importable name are reported as a
problem line and the first wins.

Only `src/` is read, so `tests/` never shows up.

Nor can it state the runtime story — which task runs what, what the model is sent, which
tool call it makes, what lands in which file under `data/`. Those the
code has to be read for.
