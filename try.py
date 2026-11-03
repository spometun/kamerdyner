"""Playground: one call to Gemini with every knob that matters spelled out.

None means "the model's default". Run: python try.py
"""

import time

t0 = time.time()
from dotenv import load_dotenv  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

load_dotenv()  # GEMINI_API_KEY from .env
client = genai.Client()
t1 = time.time()
print(f"import + client: {1000 * (t1 - t0):.0f} ms")

MODEL = "gemini-3.8-flash"
PROMPT = "Explain how AI works in a few words"

# Who the model is and how it should answer. Sent before the conversation.
SYSTEM_INSTRUCTION = "You are a concise assistant."

# How much the model reasons before answering: LOW, MEDIUM or HIGH (3.8 Flash has no MINIMAL).
# Thinking is the main cost in latency. LOW means "think little", not "don't think": the model
# decides, and on the same short question has spent 0 (~1.5 s) and 260 thought tokens (~3.6 s);
# the default level spent ~350 (~4 s). Thinking tokens are billed as output.
THINKING_LEVEL = types.ThinkingLevel.LOW
# True: the response also carries a summary of the model's thoughts (parts with thought=True).
INCLUDE_THOUGHTS = False

# Randomness of the choice of each token. 0 = always the most likely one, 1 = by the model's own
# probabilities. Google advises keeping Gemini 3 at its default 1.0: lower values can make it
# loop or reason worse.
TEMPERATURE: float | None = None
# Sample only among the most likely tokens: top_p — those making up this share of probability,
# top_k — this many. Rarely worth touching.
TOP_P: float | None = None
TOP_K: int | None = None
# Fixes the random generator: makes answers repeat more often, never guarantees it.
SEED: int | None = None

# Cap on the output, thinking tokens INCLUDED: set it too low and the answer is cut off
# (finish reason MAX_TOKENS) after the model has spent it all on thinking.
MAX_OUTPUT_TOKENS: int | None = None

# STANDARD (default), FLEX (half price, may queue at busy times), PRIORITY (~1.8x price, not
# shed or queued behind standard traffic under load).
SERVICE_TIER: types.ServiceTier | None = None

thinking = types.ThinkingConfig(thinking_level=THINKING_LEVEL, include_thoughts=INCLUDE_THOUGHTS)
config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    thinking_config=thinking,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    seed=SEED,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    service_tier=SERVICE_TIER,
    # No tools here; without this the SDK warns about its automatic function calling.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

response = client.models.generate_content(model=MODEL, contents=PROMPT, config=config)
t2 = time.time()
print(f"model call: {1000 * (t2 - t1):.0f} ms")

candidate = response.candidates[0]
print(f"finish reason: {candidate.finish_reason}")
for part in candidate.content.parts:
    label = "THOUGHTS" if part.thought else "ANSWER"
    print(f"--- {label}\n{part.text}")

usage = response.usage_metadata
print(
    f"--- tokens: prompt {usage.prompt_token_count} (cached {usage.cached_content_token_count}), "
    f"thoughts {usage.thoughts_token_count}, answer {usage.candidates_token_count}, "
    f"traffic {usage.traffic_type}"
)
