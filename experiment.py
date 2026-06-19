#!/usr/bin/env python3
"""
GreenPT Research Experiment
Compares three AI brainstorming interaction conditions on quality × sustainability.

Conditions:
  A — Freeform Chat           (no system prompt, open conversation)
  B — Specialised Agent Free  (role + goal in system prompt, open conversation)
  C — Specialised Agent Struct (role + 5-phase template: Understand→Diverge→Deepen→Stress-test→Converge)

Usage:
  python experiment.py              # run all three conditions
  python experiment.py --condition A  # run a single condition
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — swap BRAINSTORMING_GOAL for each experiment run
# ─────────────────────────────────────────────────────────────────────────────

BRAINSTORMING_GOAL = (
    "How might we get kids to eat more vegetables?"
)

MODEL = "claude-sonnet-4-6"
INPUT_PRICE_PER_MTOK  = 3.00   # USD per million input tokens
OUTPUT_PRICE_PER_MTOK = 15.00  # USD per million output tokens

def _numbered_paths() -> tuple[Path, Path]:
    n = 1
    while Path(f"results{n}.json").exists() or Path(f"results_summary{n}.md").exists():
        n += 1
    return Path(f"results{n}.json"), Path(f"results_summary{n}.md")

RESULTS_FILE, SUMMARY_FILE = _numbered_paths()


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnMetrics:
    turn: int
    role: str                   # "user" | "assistant"
    content_preview: str        # first 200 chars
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    response_chars: int = 0


@dataclass
class ConditionResult:
    condition: str
    label: str
    goal: str
    turns: list = field(default_factory=list)          # list[TurnMetrics]
    conversation: list = field(default_factory=list)   # raw message history

    # ── aggregate metrics (populated by finalise()) ──
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_response_chars: int = 0
    num_turns: int = 0
    estimated_cost_usd: float = 0.0
    prompt_efficiency: float = 0.0   # output_tokens / input_tokens
    duration_seconds: float = 0.0

    def finalise(self) -> None:
        """Compute aggregate metrics from per-turn data."""
        self.total_input_tokens          = sum(t.input_tokens           for t in self.turns)
        self.total_output_tokens         = sum(t.output_tokens          for t in self.turns)
        self.total_cache_creation_tokens = sum(t.cache_creation_tokens  for t in self.turns)
        self.total_cache_read_tokens     = sum(t.cache_read_tokens      for t in self.turns)
        self.total_response_chars = sum(
            t.response_chars for t in self.turns if t.role == "assistant"
        )
        self.num_turns = sum(1 for t in self.turns if t.role == "assistant")

        # Effective input cost:
        #   - regular input tokens  → 1.00× base price
        #   - cache creation tokens → 1.25× base price (writing to cache)
        #   - cache read tokens     → 0.10× base price (serving from cache)
        effective_input = (
            self.total_input_tokens
            + self.total_cache_creation_tokens * 1.25
            + self.total_cache_read_tokens * 0.10
        )
        self.estimated_cost_usd = (
            effective_input             / 1_000_000 * INPUT_PRICE_PER_MTOK
            + self.total_output_tokens  / 1_000_000 * OUTPUT_PRICE_PER_MTOK
        )

        if self.total_input_tokens > 0:
            self.prompt_efficiency = self.total_output_tokens / self.total_input_tokens
        else:
            self.prompt_efficiency = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content: list) -> str:
    """Pull text from a list of content blocks (SDK objects or dicts)."""
    parts = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _send(
    client: anthropic.Anthropic,
    messages: list[dict],
    system: Optional[list[dict]] = None,
) -> tuple[str, anthropic.types.Usage]:
    """Send one API call; return (assistant_text, usage)."""
    kwargs: dict = dict(model=MODEL, max_tokens=4096, messages=messages)
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return _extract_text(response.content), response.usage


def _safe_int(usage, attr: str) -> int:
    return int(getattr(usage, attr, None) or 0)


def _record_user(result: ConditionResult, turn_num: int, content: str) -> None:
    result.turns.append(TurnMetrics(
        turn=turn_num, role="user",
        content_preview=content[:200],
    ))
    result.conversation.append({"role": "user", "content": content})


def _record_assistant(
    result: ConditionResult,
    turn_num: int,
    content: str,
    usage: anthropic.types.Usage,
) -> None:
    result.turns.append(TurnMetrics(
        turn=turn_num,
        role="assistant",
        content_preview=content[:200],
        input_tokens=_safe_int(usage, "input_tokens"),
        output_tokens=_safe_int(usage, "output_tokens"),
        cache_creation_tokens=_safe_int(usage, "cache_creation_input_tokens"),
        cache_read_tokens=_safe_int(usage, "cache_read_input_tokens"),
        response_chars=len(content),
    ))
    result.conversation.append({"role": "assistant", "content": content})


def _log(turn: int, role: str, text: str, usage: Optional[anthropic.types.Usage] = None) -> None:
    tag = f"[Turn {turn}] {role.upper()}"
    print(f"\n{tag}: {text[:100]}{'…' if len(text) > 100 else ''}")
    if usage:
        print(
            f"  → tokens  in={_safe_int(usage,'input_tokens')}  "
            f"out={_safe_int(usage,'output_tokens')}  "
            f"cache_create={_safe_int(usage,'cache_creation_input_tokens')}  "
            f"cache_read={_safe_int(usage,'cache_read_input_tokens')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION A — FREEFORM CHAT
# ─────────────────────────────────────────────────────────────────────────────

def run_condition_a(goal: str = BRAINSTORMING_GOAL) -> ConditionResult:
    """
    Condition A — Freeform Chat
    No system prompt. General-purpose model. Open conversation.
    5 simulated user turns.
    """
    print("\n" + "=" * 64)
    print("CONDITION A: Freeform Chat")
    print("=" * 64)

    client = anthropic.Anthropic()
    result = ConditionResult(condition="A", label="Freeform Chat", goal=goal)
    messages: list[dict] = []
    t0 = time.monotonic()

    # ── Turn 1: Introduce the goal ──────────────────────────────────────────
    u1 = f"I'd like to brainstorm ideas about: {goal}"
    messages.append({"role": "user", "content": u1})
    _record_user(result, 1, u1)
    _log(1, "user", u1)

    resp, usage = _send(client, messages)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 1, resp, usage)
    _log(1, "assistant", resp, usage)

    # ── Turn 2: Push for unconventional ideas ───────────────────────────────
    u2 = (
        "Interesting. What about less obvious or more radical directions — "
        "ones that challenge the underlying assumptions rather than optimising "
        "within the current paradigm?"
    )
    messages.append({"role": "user", "content": u2})
    _record_user(result, 2, u2)
    _log(2, "user", u2)

    resp, usage = _send(client, messages)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 2, resp, usage)
    _log(2, "assistant", resp, usage)

    # ── Turn 3: Drill into one idea ─────────────────────────────────────────
    u3 = (
        "Pick the single most promising idea from everything you've suggested "
        "and develop it in depth — what does it look like in practice, "
        "who needs to be involved, what are the key risks?"
    )
    messages.append({"role": "user", "content": u3})
    _record_user(result, 3, u3)
    _log(3, "user", u3)

    resp, usage = _send(client, messages)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 3, resp, usage)
    _log(3, "assistant", resp, usage)

    # ── Turn 4: Stress-test ─────────────────────────────────────────────────
    u4 = (
        "Good. Now critically challenge that idea — "
        "what are its real weaknesses, and what would have to be true for it to fail?"
    )
    messages.append({"role": "user", "content": u4})
    _record_user(result, 4, u4)
    _log(4, "user", u4)

    resp, usage = _send(client, messages)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 4, resp, usage)
    _log(4, "assistant", resp, usage)

    # ── Turn 5: Action plan ─────────────────────────────────────────────────
    u5 = (
        "Fair points. Despite those risks, give me a concise action plan: "
        "three concrete next steps someone could take this week to get started."
    )
    messages.append({"role": "user", "content": u5})
    _record_user(result, 5, u5)
    _log(5, "user", u5)

    resp, usage = _send(client, messages)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 5, resp, usage)
    _log(5, "assistant", resp, usage)

    result.duration_seconds = time.monotonic() - t0
    result.finalise()
    _print_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION B — SPECIALISED AGENT, FREEFORM
# ─────────────────────────────────────────────────────────────────────────────

_SPECIALIST_SYSTEM = (
    "You are an expert brainstorming facilitator with deep experience in "
    "creative problem-solving and innovation strategy. Your goal is to help "
    "users generate rich, diverse, and actionable ideas on any topic.\n\n"
    "Your approach:\n"
    "- Ask clarifying questions to understand the problem deeply\n"
    "- Generate a wide range of ideas from incremental to radical\n"
    "- Identify both opportunities and potential pitfalls\n"
    "- Help synthesise ideas into clear, prioritised insights\n"
    "- Keep responses focused and useful, not exhaustive"
)


def run_condition_b(goal: str = BRAINSTORMING_GOAL) -> ConditionResult:
    """
    Condition B — Specialised Agent, Freeform
    System prompt defines the brainstorming specialist role.
    Conversation stays open and unstructured.
    System prompt is prompt-cached across turns.
    """
    print("\n" + "=" * 64)
    print("CONDITION B: Specialised Agent — Freeform")
    print("=" * 64)

    client = anthropic.Anthropic()
    result = ConditionResult(
        condition="B", label="Specialised Agent — Freeform", goal=goal
    )
    messages: list[dict] = []

    # System prompt with cache_control — reused on every turn
    system = [{
        "type": "text",
        "text": _SPECIALIST_SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }]

    t0 = time.monotonic()

    # ── Turn 1: State goal ──────────────────────────────────────────────────
    u1 = f"I want to brainstorm about: {goal}"
    messages.append({"role": "user", "content": u1})
    _record_user(result, 1, u1)
    _log(1, "user", u1)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 1, resp, usage)
    _log(1, "assistant", resp, usage)

    # ── Turn 2: Challenge assumptions ───────────────────────────────────────
    u2 = (
        "Some of those are interesting. What about approaches that would require "
        "rethinking the whole system rather than just optimising within it? "
        "I'm looking for ideas that might feel uncomfortable or disruptive."
    )
    messages.append({"role": "user", "content": u2})
    _record_user(result, 2, u2)
    _log(2, "user", u2)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 2, resp, usage)
    _log(2, "assistant", resp, usage)

    # ── Turn 3: Go deep on one idea ─────────────────────────────────────────
    u3 = (
        "The systemic direction is compelling. Take the idea you think has "
        "the highest potential and develop it fully — who needs to be involved, "
        "what would success look like in two years, and what's the hardest part?"
    )
    messages.append({"role": "user", "content": u3})
    _record_user(result, 3, u3)
    _log(3, "user", u3)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 3, resp, usage)
    _log(3, "assistant", resp, usage)

    # ── Turn 4: Stress-test ─────────────────────────────────────────────────
    u4 = (
        "Good. Now critically challenge that idea — "
        "what are its real weaknesses, and what would have to be true for it to fail?"
    )
    messages.append({"role": "user", "content": u4})
    _record_user(result, 4, u4)
    _log(4, "user", u4)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 4, resp, usage)
    _log(4, "assistant", resp, usage)

    # ── Turn 5: Concrete next steps ─────────────────────────────────────────
    u5 = (
        "Fair points. Despite those risks, give me a concise action plan: "
        "three concrete next steps someone could take this week to get started."
    )
    messages.append({"role": "user", "content": u5})
    _record_user(result, 5, u5)
    _log(5, "user", u5)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 5, resp, usage)
    _log(5, "assistant", resp, usage)

    result.duration_seconds = time.monotonic() - t0
    result.finalise()
    _print_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONDITION C — SPECIALISED AGENT, STRUCTURED
# ─────────────────────────────────────────────────────────────────────────────

_STRUCTURED_SYSTEM = (
    "You are an expert brainstorming facilitator with deep experience in "
    "creative problem-solving and innovation strategy. Your goal is to help "
    "users generate rich, diverse, and actionable ideas on any topic.\n\n"
    "You follow a five-phase structured process. "
    "Adapt the pace to the human's signals — if they skip ahead, follow them.\n\n"
    "PHASE 1 — UNDERSTAND (moves 1–2)\n"
    "  Move 1: Receive the goal exactly as stated. Confirm understanding briefly.\n"
    "  Move 2: Ask exactly ONE focused clarifying question to sharpen context.\n\n"
    "PHASE 2 — DIVERGE (moves 3–4)\n"
    "  Move 3: Generate 5–10 distinct, labelled ideas (A, B, C … J).\n"
    "           Cover a range from incremental to radical.\n"
    "  Move 4: Invite the human to choose one label to explore further.\n\n"
    "PHASE 3 — DEEPEN (moves 5–6)\n"
    "  Move 5: Expand the chosen idea into a full concept "
    "(problem it solves / mechanism / expected outcome).\n"
    "  Move 6: Offer 2–3 variations of the concept that explore different angles.\n\n"
    "PHASE 4 — STRESS-TEST (moves 7–8)\n"
    "  Move 7: Critique the chosen concept honestly — name its real weaknesses.\n"
    "  Move 8: Compare it to the other shortlisted ideas; name the key trade-offs.\n\n"
    "PHASE 5 — CONVERGE (moves 9–11)\n"
    "  Move 9: Detect signals of commitment in the human's language; name them.\n"
    "  Move 10: Structure the final recommendation as: Problem / Target User / Scope.\n"
    "  Move 11: Ask what output format is wanted, then deliver that final output.\n\n"
    "Be decisive and concise. Do not pad or repeat yourself between phases."
)


def run_condition_c(goal: str = BRAINSTORMING_GOAL) -> ConditionResult:
    """
    Condition C — Specialised Agent, Structured
    Same specialist role as B, guided through a 5-phase template
    (Understand → Diverge → Deepen → Stress-test → Converge, 11 moves).
    System prompt is prompt-cached across turns.
    """
    print("\n" + "=" * 64)
    print("CONDITION C: Specialised Agent — Structured")
    print("=" * 64)

    client = anthropic.Anthropic()
    result = ConditionResult(
        condition="C", label="Specialised Agent — Structured", goal=goal
    )
    messages: list[dict] = []

    system = [{
        "type": "text",
        "text": _STRUCTURED_SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }]

    t0 = time.monotonic()

    # ── Turn 1 — Phase 1 Understand: agent confirms + asks one question ─────
    u1 = f"Here is my brainstorming goal: {goal}"
    messages.append({"role": "user", "content": u1})
    _record_user(result, 1, u1)
    _log(1, "user", u1)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 1, resp, usage)
    _log(1, "assistant", resp, usage)

    # ── Turn 2 — Phase 2 Diverge: user answers context Q → agent generates ideas ─
    u2 = (
        "We're looking for practical approaches that could work within 1–2 years. "
        "The main constraint is limited resources, so solutions should be low-cost "
        "or easy to scale. The target audience is everyday people, not specialists."
    )
    messages.append({"role": "user", "content": u2})
    _record_user(result, 2, u2)
    _log(2, "user", u2)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 2, resp, usage)
    _log(2, "assistant", resp, usage)

    # ── Turn 3 — Phase 3 Deepen: user picks idea → agent deepens + offers variations
    u3 = "Let's go with option C — that one feels most relevant to our situation."
    messages.append({"role": "user", "content": u3})
    _record_user(result, 3, u3)
    _log(3, "user", u3)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 3, resp, usage)
    _log(3, "assistant", resp, usage)

    # ── Turn 4 — Phase 4 Stress-test: user reacts → agent critiques + compares ─
    u4 = (
        "The first variation looks most promising. "
        "Now give me the honest critique — where could this actually go wrong, "
        "and how does it compare to the other ideas you listed?"
    )
    messages.append({"role": "user", "content": u4})
    _record_user(result, 4, u4)
    _log(4, "user", u4)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 4, resp, usage)
    _log(4, "assistant", resp, usage)

    # ── Turn 5 — Phase 5 Converge: user signals commitment → agent structures + delivers
    u5 = (
        "Those trade-offs are acceptable. I think this is the right direction — "
        "I'm ready to move forward. Please give me a final structured recommendation "
        "formatted as a one-page brief I can share with my team."
    )
    messages.append({"role": "user", "content": u5})
    _record_user(result, 5, u5)
    _log(5, "user", u5)

    resp, usage = _send(client, messages, system=system)
    messages.append({"role": "assistant", "content": resp})
    _record_assistant(result, 5, resp, usage)
    _log(5, "assistant", resp, usage)

    result.duration_seconds = time.monotonic() - t0
    result.finalise()
    _print_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(result: ConditionResult) -> None:
    print(
        f"\n  ── Condition {result.condition} complete ──"
        f"  turns={result.num_turns}"
        f"  input_tok={result.total_input_tokens:,}"
        f"  output_tok={result.total_output_tokens:,}"
        f"  cache_read={result.total_cache_read_tokens:,}"
        f"  cost=${result.estimated_cost_usd:.4f}"
        f"  efficiency={result.prompt_efficiency:.3f}"
        f"  duration={result.duration_seconds:.1f}s"
    )


def save_results(results: list[ConditionResult]) -> None:
    """Serialise all ConditionResult objects to results.json."""
    payload = [asdict(r) for r in results]
    RESULTS_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n✓ Raw results → {RESULTS_FILE}")


def _build_summary_markdown(results: list[ConditionResult]) -> str:
    """Build and return the full markdown summary string."""

    def _row(label: str, fn) -> str:
        cells = " | ".join(fn(r) for r in results)
        return f"| {label} | {cells} |"

    lines = [
        "# Brainstorming Experiment — Results Summary",
        "",
        f"**Goal:** {results[0].goal}",
        f"**Model:** `{MODEL}`",
        f"**Pricing:** input ${INPUT_PRICE_PER_MTOK}/MTok · output ${OUTPUT_PRICE_PER_MTOK}/MTok",
        "",
        "---",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | Condition A | Condition B | Condition C |",
        "| --- | --- | --- | --- |",
        _row("Label",                     lambda r: r.label),
        _row("Interaction turns",         lambda r: str(r.num_turns)),
        _row("Total input tokens",        lambda r: f"{r.total_input_tokens:,}"),
        _row("Total output tokens",       lambda r: f"{r.total_output_tokens:,}"),
        _row("Cache created (tokens)",    lambda r: f"{r.total_cache_creation_tokens:,}"),
        _row("Cache read (tokens)",       lambda r: f"{r.total_cache_read_tokens:,}"),
        _row("Total response chars",      lambda r: f"{r.total_response_chars:,}"),
        _row("Avg chars / turn",          lambda r: f"{r.total_response_chars // max(r.num_turns, 1):,}"),
        _row("Estimated cost (USD)",      lambda r: f"${r.estimated_cost_usd:.4f}"),
        _row("Prompt efficiency (out/in)", lambda r: f"{r.prompt_efficiency:.3f}"),
        _row("Duration (s)",              lambda r: f"{r.duration_seconds:.1f}"),
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Definition |",
        "| --- | --- |",
        "| Interaction turns | Number of assistant responses |",
        "| Prompt efficiency | `total_output_tokens ÷ total_input_tokens` — signal generated per input token |",
        "| Estimated cost | Accounts for caching: cache-creation billed at 1.25×, cache-reads at 0.10× of input price |",
        "| Cache created | Tokens written to the prompt cache (first request per session) |",
        "| Cache read | Tokens served from the prompt cache (subsequent requests) |",
        "",
        "---",
        "",
    ]

    # ── Conclusions ──────────────────────────────────────────────────────────
    if len(results) == 3:
        ra, rb, rc = results[0], results[1], results[2]

        most_output   = max(results, key=lambda r: r.total_output_tokens)
        least_cost    = min(results, key=lambda r: r.estimated_cost_usd)
        most_efficient = max(results, key=lambda r: r.prompt_efficiency)
        fastest       = min(results, key=lambda r: r.duration_seconds)
        most_chars    = max(results, key=lambda r: r.total_response_chars)

        b_cache_pct = (
            rb.total_cache_read_tokens / max(rb.total_input_tokens + rb.total_cache_creation_tokens, 1) * 100
        )
        c_cache_pct = (
            rc.total_cache_read_tokens / max(rc.total_input_tokens + rc.total_cache_creation_tokens, 1) * 100
        )

        cost_a_vs_b = ((rb.estimated_cost_usd - ra.estimated_cost_usd) / ra.estimated_cost_usd * 100)
        cost_a_vs_c = ((rc.estimated_cost_usd - ra.estimated_cost_usd) / ra.estimated_cost_usd * 100)

        # Derived comparisons used across multiple sections
        agent_conditions = [rb, rc]
        agent_avg_cost = (rb.estimated_cost_usd + rc.estimated_cost_usd) / 2
        agent_avg_eff  = (rb.prompt_efficiency + rc.prompt_efficiency) / 2
        agents_cheaper_than_freeform = agent_avg_cost < ra.estimated_cost_usd
        agents_more_efficient = agent_avg_eff > ra.prompt_efficiency
        structured_vs_freeform_eff_delta = (
            (rc.prompt_efficiency - ra.prompt_efficiency) / max(ra.prompt_efficiency, 0.001) * 100
        )

        lines += [
            "## Conclusions",
            "",
            "### Output volume",
            f"Condition **{most_output.condition} ({most_output.label})** generated the most output tokens "
            f"({most_output.total_output_tokens:,}), and Condition **{most_chars.condition}** produced the "
            f"longest responses by character count ({most_chars.total_response_chars:,} chars, "
            f"{most_chars.total_response_chars // max(most_chars.num_turns, 1):,} avg per turn). "
            f"Higher token counts do not automatically mean better output — volume needs to be weighed "
            f"against the usefulness and focus of what was produced.",
            "",
            "### Cost",
            f"Condition **{least_cost.condition} ({least_cost.label})** was the cheapest at "
            f"${least_cost.estimated_cost_usd:.4f}. "
            f"Condition B cost {'more' if cost_a_vs_b > 0 else 'less'} than A "
            f"({abs(cost_a_vs_b):.1f}%), and Condition C cost "
            f"{'more' if cost_a_vs_c > 0 else 'less'} than A ({abs(cost_a_vs_c):.1f}%). "
            + (
                "Despite carrying a system prompt overhead, the specialised agents kept costs "
                "competitive with freeform chat — prompt caching on B and C is designed to "
                "offset that overhead in repeated-use scenarios."
                if agents_cheaper_than_freeform
                else
                "The system prompt overhead in B and C pushed their costs above freeform chat (A), "
                "though caching would reduce this gap if the same agent were reused across multiple sessions."
            ),
            "",
            "### Prompt efficiency",
            f"Condition **{most_efficient.condition} ({most_efficient.label})** had the highest prompt "
            f"efficiency ({most_efficient.prompt_efficiency:.3f} output tokens per input token). "
            + (
                f"The structured agent (C) achieved {abs(structured_vs_freeform_eff_delta):.1f}% "
                f"{'higher' if structured_vs_freeform_eff_delta > 0 else 'lower'} efficiency than "
                f"freeform chat (A), suggesting that a well-defined task structure "
                + ("reduces the input overhead relative to the useful output it generates."
                   if structured_vs_freeform_eff_delta > 0
                   else "does not automatically translate to leaner token use — the richer system "
                        "prompt adds input cost that the structured output does not always offset.")
            ),
            "",
            "### Caching and reuse",
            f"Conditions B and C used prompt caching on their system prompts. "
            f"Cache read rates this run: B = {b_cache_pct:.1f}%, C = {c_cache_pct:.1f}%. "
            + (
                "No cache hits occurred because each condition ran as a fresh session — this is expected. "
                "The real sustainability benefit of caching appears at scale: when the same specialised "
                "agent handles many sessions, the system prompt is served from cache at 10× lower cost "
                "per token, meaning the per-interaction footprint of B and C shrinks significantly "
                "compared to freeform chat as usage grows."
                if rb.total_cache_read_tokens == 0 and rc.total_cache_read_tokens == 0
                else
                f"B read {rb.total_cache_read_tokens:,} cached tokens; C read {rc.total_cache_read_tokens:,}. "
                "These cache hits reduce effective input cost and represent the sustainability advantage "
                "of reusing a specialised agent across sessions."
            ),
            "",
            "### Speed",
            f"Condition **{fastest.condition} ({fastest.label})** completed fastest "
            f"({fastest.duration_seconds:.1f}s). "
            f"A={ra.duration_seconds:.1f}s, B={rb.duration_seconds:.1f}s, C={rc.duration_seconds:.1f}s. "
            f"Faster completion means fewer compute-seconds per task, which is a direct component "
            f"of resource consumption at scale.",
            "",
            "### Sustainability of specialised agents vs. freeform chat",
            _sustainability_takeaway(ra, rb, rc, agents_cheaper_than_freeform, agents_more_efficient,
                                     structured_vs_freeform_eff_delta),
            "",
            "---",
            "",
        ]

    # ── Conversation Previews ─────────────────────────────────────────────────
    lines.append("## Conversation Previews")
    lines.append("")

    for r in results:
        lines.append(f"### Condition {r.condition}: {r.label}")
        lines.append("")
        for t in r.turns:
            role_tag = "**User**" if t.role == "user" else "**Assistant**"
            preview = t.content_preview
            suffix = "…" if len(t.content_preview) == 200 else ""
            tok_info = ""
            if t.role == "assistant":
                tok_info = (
                    f" *(in={t.input_tokens:,} out={t.output_tokens:,}"
                    + (f" cache_read={t.cache_read_tokens:,}" if t.cache_read_tokens else "")
                    + ")*"
                )
            lines.append(f"- {role_tag} [turn {t.turn}]{tok_info}: {preview}{suffix}")
        lines.append("")

    return "\n".join(lines)


def _sustainability_takeaway(
    ra: "ConditionResult",
    rb: "ConditionResult",
    rc: "ConditionResult",
    agents_cheaper: bool,
    agents_more_efficient: bool,
    structured_eff_delta: float,
) -> str:
    """
    Answer the core research question: are specialised agents more sustainable
    than freeform chat for the same goal?
    """
    parts = []

    # Single-run cost picture
    if agents_cheaper:
        parts.append(
            "Across this single run, specialised agents (B and C) were on average cheaper than "
            "freeform chat (A), even when accounting for the system prompt overhead."
        )
    else:
        parts.append(
            "In this single run, freeform chat (A) had a lower raw cost than the specialised agents, "
            "primarily because it carries no system prompt overhead."
        )

    # Efficiency angle — the sustainability core
    if agents_more_efficient:
        parts.append(
            "More importantly, the agents generated more output per input token consumed — meaning "
            "they extracted more useful work from the same compute budget."
        )
    else:
        parts.append(
            "On prompt efficiency, freeform chat matched or exceeded the agents in this run, "
            "which suggests the system prompt cost is not yet being offset by tighter output."
        )

    # Structured vs freeform: the clearest signal
    if structured_eff_delta > 5:
        parts.append(
            f"The structured agent (C) showed the sharpest result: its constrained five-phase "
            f"protocol produced {structured_eff_delta:.1f}% more output per input token than "
            f"freeform chat. When a task is well-defined, giving the model a clear structure "
            f"appears to reduce token waste — the model spends less of its context budget "
            f"on orientation, hedging, and reformulation."
        )
    elif structured_eff_delta > 0:
        parts.append(
            f"The structured agent (C) edged out freeform chat on efficiency "
            f"(+{structured_eff_delta:.1f}%), a modest signal that structure helps but is not "
            f"yet decisive at this scale."
        )
    else:
        parts.append(
            f"The structured agent (C) did not outperform freeform chat on efficiency in this run "
            f"({structured_eff_delta:.1f}%), suggesting the protocol overhead outweighed its "
            f"focusing effect here."
        )

    # Scale and caching — where the sustainability story really plays out
    parts.append(
        "The stronger sustainability case for specialised agents appears at scale rather than "
        "per-session. A reusable agent with a cached system prompt sees its per-interaction "
        "input cost fall sharply as sessions accumulate — a cost structure that freeform chat, "
        "which rebuilds context from scratch every time, cannot match. "
        "For a repeated task like brainstorming across many teams or sessions, "
        "a well-designed specialised agent is likely to be meaningfully more resource-efficient "
        "than asking the same question in an open conversation each time."
    )

    return " ".join(parts)


def generate_summary(results: list[ConditionResult]) -> None:
    """Write a markdown summary (with conclusions) to results_summary.md."""
    markdown = _build_summary_markdown(results)
    SUMMARY_FILE.write_text(markdown, encoding="utf-8")
    print(f"✓ Summary     → {SUMMARY_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

def run_all(goal: str = BRAINSTORMING_GOAL) -> list[ConditionResult]:
    """Run all three conditions and persist outputs."""
    print(f"\nBrainstorming goal:\n  {goal}\n")
    results = [
        run_condition_a(goal),
        run_condition_b(goal),
        run_condition_c(goal),
    ]
    save_results(results)
    generate_summary(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GreenPT brainstorming experiment — compare three AI interaction conditions"
    )
    parser.add_argument(
        "--condition",
        choices=["A", "B", "C"],
        default=None,
        help="Run a single condition (default: run all three)",
    )
    parser.add_argument(
        "--goal",
        default=BRAINSTORMING_GOAL,
        help="Override the brainstorming goal string",
    )
    args = parser.parse_args()

    if args.condition is None:
        run_all(goal=args.goal)
    elif args.condition == "A":
        r = run_condition_a(goal=args.goal)
        save_results([r])
        generate_summary([r])
    elif args.condition == "B":
        r = run_condition_b(goal=args.goal)
        save_results([r])
        generate_summary([r])
    elif args.condition == "C":
        r = run_condition_c(goal=args.goal)
        save_results([r])
        generate_summary([r])
