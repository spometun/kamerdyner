# Working guidelines for this codebase

## Project

Kamerdyner — one AI assistant per couple, with a separate chat for each partner. The bot knows
both of them (one shared memory, structured by whom a fact is about and whose words it is) and
is a general-purpose assistant. When one partner asks it to tell the other something, the model
calls `send_to_partner`: the message is translated so it lands for the other person, delivered
to their chat immediately, and the sender sees the translation.

**Nothing is written yet; the server is Python.** Read `docs/idea.md` at the start of the
session: it holds the decisions so far (bot on a server, one bot and one memory per couple,
`send_to_partner` without confirmation, cross-partner leakage accepted as a conscious risk,
couples-only MVP, Telegram bot as the recommended first client, hosting on a laptop then a VPS
with Cloud Run as a possible later migration, Gemini 3.8+ Flash as the LLM, stage 1 single user
/ stage 2 couple), privacy deliberately deferred to stage 2, and the open questions. Settle
those with the user before writing code, then update this section.

`docs/gemini_chat_2026-10-02.md` is the original brainstorming chat with Gemini (~450 KB — grep
it, don't read it whole). It is raw input, not a spec: several of its choices are deliberately
rejected in `docs/idea.md`. It holds personal data, so it lives only locally and is kept out of
git (`.gitignore`); never commit it or quote its personal details anywhere in the repo.

## Workflow

- **Say what was asked and how you will do it BEFORE doing it.** Open every task with a couple
  of sentences — your reading of the request, and the idea/plan you are about to carry out —
  and only then start the tool calls. Keep it short: the ask restated, the approach, anything
  you are assuming. Where an interface needs an OK, still stop and get it.
- **"Investigate" means investigate, not fix.** When asked to investigate (a failing test, a bug,
  a crash), do the diagnosis, report what you found and the proposed solution, then STOP. Do not
  edit code until the user confirms. Same for "look into", "why does X happen", "what's wrong
  with". An explicit "fix it" / "investigate and fix" lifts this.
- **Run the tests freely, not only at the end.** If a suite ever takes minutes, that is a bug in
  a test, not a fact of life — find the test that is eating the time.
- **Never write Ukrainian (or any non-English reply) in Latin transliteration.** Native script
  (Cyrillic for Ukrainian) always. If that is somehow not possible, write in English instead.

## Git & commits

- **Commit on the current branch.** Just `git add` + `git commit` on whatever is checked out. Do
  NOT auto-create a new branch first.
- **Use `git -C <absolute-dir> <command>`** instead of `cd <dir> && git ...`.
- **Push with `git -C <absolute-repo-dir> push origin <branch>`** — that exact form. Never
  `git push` bare, and never with force flags or `+refspec`.
- **Commit messages: concise, subsystem first.** `<subsystem>: <what changed>`, minimal or no
  body. E.g. `memory: cap recent notes at 500 tokens`.
- **Attribution trailer — actual model, no email.** End commit messages with
  `Co-Authored-By: Claude <model>`, where `<model>` is the model actually powering the session
  (e.g. `Co-Authored-By: Claude Opus 5.5`); fall back to `Co-Authored-By: Claude AI` if unknown.
  Never append the `<noreply@anthropic.com>` email. This OVERRIDES the built-in default.
- **Never commit secrets or user data.** The Telegram bot token and the Gemini API key live in
  `.env`; real users' memory, logs and state live under `data/`. Both stay out of git
  (`.gitignore`).

## Errors: bugs fail fast, the outside world is expected to fail

The bot lives in a hostile environment: the network drops, the Gemini API answers 429 or 5xx or
cuts a reply off, Telegram delivers an update twice, a user sends a sticker where text was
expected, the laptop sleeps mid-request and the process is killed. None of that is a bug, and
none of it may take the bot down. A bug in our own code is different: it must surface loudly,
not be papered over. Keep the two strictly apart.

- **Our own bugs: fail fast, no defensive fallbacks.** Don't add default fallbacks unless
  genuinely necessary. No `x or ""`, `d.get(key, [])`, `getattr(obj, name, None)` on a value
  that is required — if it is missing, that is a bug and the exception (with its traceback in
  the log) is how we find out. Avoid `if x is None: return`-style guards unless `None` is a
  legitimate expected value. Prefer non-optional types so the question never arises.
- **Expected external failures: handle them at the boundary, as values.** Network, Telegram API,
  Gemini API, file I/O on user data — the layer that talks to the outside world catches the
  specific exceptions it knows (`OSError`, `httpx.HTTPError`, the SDK's own error types, HTTP
  status codes) and turns them into an explicit outcome: a union of frozen dataclasses that the
  caller handles with `match` and `typing.assert_never`, so a forgotten case is a type error.
  The caller then shows the user a short message, retries, or defers the work. Everything above
  that layer works with values, not with try/except.
- **Catch narrowly, never swallow.** No bare `except:`, no `except: pass`. No
  `except Exception` / `except BaseException` / `contextlib.suppress(Exception)` around
  ordinary code — that turns our bugs into silent wrong behaviour. The only broad catch is a
  top-level boundary (one per Telegram update handler, one per background task) that logs the
  traceback (`logger.exception`) and tells the user something went wrong.
- **Never swallow `asyncio.CancelledError`.** Cancellation travels as an exception; a handler
  that eats it breaks structured concurrency. It derives from `BaseException`, which is one more
  reason never to catch `BaseException`; if you catch it for cleanup, re-raise it.
- **Retries are bounded and deliberate.** Retry only what is idempotent and transient (429, 503,
  timeouts), with backoff and a cap, and say so at the call site. Work that can be delivered
  twice is made idempotent by an id: a Telegram `update_id` already processed is dropped, and
  the memory-update pass advances a watermark so a message is folded into memory exactly once.
- **Treat everything that arrives from outside as untrusted input.** A Telegram update, a
  model's tool call and its arguments, a model's memory-update output (later: a message from the
  partner's side): validate its shape where it enters, reject what is out of contract with a
  logged reason, never crash on it and never let it reach code that assumes it is well-formed.
  Model output is not our code.
- **Survive process death.** Anything the user would be upset to lose (an incoming message, an
  updated profile, the memory-pass watermark) is persisted before we depend on it, not held only
  in memory: an incoming message goes to the log before the model is called. Files are written
  atomically (write a temp file, then `os.replace`), so a crash never leaves a half-written
  profile.
- **Don't validate a thing until it is actually used.** Don't pre-check that a file exists when
  you only build its path — the eventual read will fail on its own and be handled where reads
  are handled.
- **But raise silent errors explicitly.** When something can end early or come back short (a
  reply cut off by `max_output_tokens` — thinking tokens count against it — a response blocked
  by safety filters, an empty candidate, a tool call with missing arguments, a memory update over
  its size budget), don't quietly carry on with incomplete data — check the finish reason and
  the end state and report it.
- **Error messages must be meaningful and include the offending values.** Every `assert` and
  raised exception carries an f-string message with the concrete values, e.g.
  `assert watermark <= last_id, f"Internal error. Watermark {watermark} is past the last message {last_id} of user {user_id}"`.
- **Internal-invariant violations start with `"Internal error. "`** — a condition that can only
  fail through a bug in our own code. Use `assert cond, f"Internal error. ..."`; we never run
  Python with `-O`, so asserts are always on.
- **Argument-precondition violations start with `"Invalid input. "`** — a documented constraint
  on a function's own parameters. Raise `ValueError` (or `TypeError`), name the parameter and
  show what arrived: `raise ValueError(f"Invalid input. limit must be positive, got {limit}")`.
  `Internal error. ` means we corrupted our own state; `Invalid input. ` means the caller should
  look at its own call site first.
- **No silencing of optionality.** Where a `X | None` value is guaranteed present, narrow it
  with `assert x is not None, f"Internal error. ..."` so the failure says what was missing — not
  `typing.cast` or `# type: ignore`. Better still, restructure so the type is not optional in the
  first place.
- **Don't re-assert what's already guaranteed** — by the type checker, by a dataclass's
  `__post_init__` validation, or by an assignment two lines above. Reserve checks for genuinely
  non-obvious invariants that have no other safety net.

## Coding style

- **Handle edge cases by design, not by branching.** Prefer a structure where the edge case is
  just the normal path taken with a boundary value: an empty list instead of `None` + a guard, a
  variant of the result union that already means "nothing here". An unreadable trick is worse
  than a plain `if`.
- **Type annotations everywhere, checked strictly.** Every function declares its parameter and
  return types; the type checker runs in strict mode. Local variables may rely on inference.
  Prefer frozen dataclasses for domain values over dicts and tuples.
- **Name intermediate call results; never nest a call inside another call.** Re-read the lines
  you just wrote and split every `f(g(...))`: `prompt = build_system_prompt(profile)` then
  `llm.generate(prompt)`, not `llm.generate(build_system_prompt(profile))`. That includes
  constructing a thing inside the call that consumes it (`log.append(Message(...))` → name the
  message, then append it). Not covered: an attribute as an argument, and calls inside an
  f-string for a log or an error message.
- **Comprehensions stay short.** A one-step comprehension that reads at a glance is fine.
  Nested comprehensions, side effects, multi-clause conditions or accumulation — write a plain
  `for` loop.
- **The event loop does no blocking work.** The bot runs on `asyncio`: network and model calls
  use async clients; anything blocking (a sync SDK call, heavy file work) goes through
  `asyncio.to_thread`. Reading and writing our own small files (a profile of a few KB) inline is
  fine. Tasks live in a scope tied to a lifecycle — `asyncio.TaskGroup`, or a kept task whose
  result is awaited — never a fire-and-forget `create_task` whose exception nobody observes.
- **Randomness and time are deterministic in tests.** Never rely on an unseeded RNG or the wall
  clock in a test: pass a seeded `random.Random(0)` in, and pass the clock in wherever behaviour
  depends on time (the silence before a memory pass, the dates on recent notes).
- **Public declarations at the top of the module, private ones below.** Module-private names
  start with `_`. A reader sees what a module offers before how it works.
- **The better design wins over a clean linter.** Don't weaken a design to silence a false
  positive; suppress it narrowly (`# type: ignore[code]`, `# noqa: CODE`, always with the
  specific code) and note why in a comment.

## Docs

- **Document what a value IS and how it's USED, not how it was obtained.** Construction history
  belongs at the call site. Describe an input format by its structure, never by which component
  currently produces it.

## Architecture

- **Write a domain library first, assemble the program from it.** While coding, keep
  extracting the abstractions, components and primitives of the domain (memory, profile,
  conversation, memory-update pass, partner message) into a reusable library. A program is a
  thin assembly on top: the laptop / VPS build (long polling, local files, in-process timers) and a Cloud Run
  build (webhook, Cloud SQL / Firestore, Cloud Scheduler) differ only in that assembly, never in
  the library.
- **Boundaries with the outside world stay migratable, with the minimum abstraction.** The
  known boundaries: transport (Telegram polling / webhook, later an own app), storage, deferred
  tasks, LLM provider. A boundary gets the simplest abstraction that works — often a small
  interface with one implementation. Sometimes none at all is right, but then all access to that
  thing lives in one module and its specifics (SQL, SDK types, Telegram objects) don't leak out,
  so an abstraction can be introduced later as a local change. No speculative layers for
  scenarios that don't exist.

## API design

- **Aim for the middle on generality.** Don't tailor functionality so tightly it only serves the
  immediate caller, nor make it gratuitously broad. Parameterize the genuinely-varying axis, but
  don't invent parameters for scenarios that don't exist.

## Reporting

- **Don't mention pre-existing diagnostic issues.** After an edit, report IDE/lint warnings only
  if they are clearly caused by the edit and clearly fixable.
