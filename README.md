# Kamerdyner

*An AI assistant for a couple: one bot that knows both partners, a separate chat for each, and a
way to tell your partner something so that it actually lands.*

*Kamerdyner* is Ukrainian for a valet — a personal servant who knows the household inside out.
Here it is a general-purpose AI assistant shared by two people. Each partner talks to it in their
own private chat about anything, from the weather to paying a parking fine. But it knows both of
them well: their character, what is worrying them right now, what each one values and how to
talk to each of them.

## The idea

Saying something important to a partner is often hard: what is a cry from the heart for one of
them can sound like a reproach to the other. Kamerdyner translates between two people who speak
the same language but hear it differently.

- Marta tells her assistant: "Tell Andriy that…".
- The assistant translates the meaning rather than the words, phrased so that it lands for
  Andriy given what it knows about both of them. It delivers the message straight to Andriy's
  chat: "From Marta: …".
- Marta sees exactly how her words were passed on, and can ask why they were rephrased that way.
- Andriy talks it over with his own assistant, understands Marta better, and replies the same way.

## Design

- **The assistant lives on a server; clients are thin adapters.** The first client is a Telegram
  bot, which provides chat, voice messages, push notifications and identity out of the box. A
  native app can come later as another adapter over the same core.
- **One memory per couple**, structured not only by *whom* a fact is about but by *whose words*
  it is: "he avoids talking about money", said by her, is her perception, not a fact about him.
- **`send_to_partner`** is a tool the model calls on its own when asked to pass something on.
  It delivers immediately, without a confirmation step, and the sender sees the translation.
- **Privacy risks are explicit design decisions.** The assistant sees both partners' memory in
  each chat and might leak something one told it in confidence. This is mitigated by the prompt,
  not guaranteed, and accepted consciously. Consent, data access and deletion are planned before
  any real couple uses it (Canadian PIPEDA).
- **Stack:** Python, `asyncio`, Gemini Flash. Memory is stored in plain files first, so it can be
  read and fixed by hand.

## Engineering approach

- **A domain library first, the program as a thin assembly on top.** Memory, profile,
  conversation, the memory-update pass and partner messages are reusable components. A laptop
  build (long polling, local files, in-process timers) and a Cloud Run build (webhook, managed
  database, Cloud Scheduler) are meant to differ only in that assembly.
- **The outside world is expected to fail; our own bugs must not hide.** Network drops, LLM rate
  limits, truncated replies and duplicate Telegram updates are handled at the boundary as typed
  outcomes (unions of frozen dataclasses, matched exhaustively). Internal invariants fail fast
  and loudly, with no defensive fallbacks.
- **Model output is untrusted input.** Tool calls and memory updates are validated where they
  enter the system.
- **Survives process death.** Incoming messages are persisted before the model is called, files
  are written atomically, and the memory pass advances a watermark so each message is folded in
  exactly once.
- **Strict typing and deterministic tests**: an injected clock and seeded randomness.

## Roadmap

1. **Single user:** an assistant that works well on its own and maintains a good, compact memory
   of the person it talks to.
2. **Couple:** shared memory, `send_to_partner`, and the privacy minimum (consent, `/memory`,
   `/delete`).

## Status

Stage 1 in progress: a console chat with one user on top of the library. The conversation is
stored and trimmed at 16K tokens; the memory that will absorb the trimmed part is a placeholder
that remembers nothing yet. The design notes, decisions, accepted risks and open questions are
in [`docs/idea.md`](docs/idea.md) (in Ukrainian).

## Running

```sh
conda env create -f environment.yml && conda activate kamerdyner
cp .env.example .env                    # fill in GEMINI_API_KEY and GEMINI_MODEL
mkdir -p data/marta && cp user.example.toml data/marta/user.toml
python -m kamerdyner.apps.console data/marta
```

Checks: `pytest`, `pyright`, `ruff check`, `ruff format --check`.
