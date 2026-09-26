"""Gold symbol discovery and verification (spec §7, §55).

Brokers spell gold differently — ``XAUUSD``, ``XAUUSDm``, ``XAUUSD.a``,
``GOLD``, ``GOLDm`` … — so the symbol is discovered, never hardcoded.
Discovery is two-stage:

1. **Match** the broker's symbol list against gold naming patterns
   (score-ordered, silver/platinum/palladium explicitly denied).
2. **Verify** the broker metadata of each candidate (visible, trading allowed,
   digits, point, tick size/value, contract size, volume constraints, stops &
   freeze levels) — the first candidate that fully verifies wins.

Gold-only protection: an explicitly configured symbol that does not match a
gold pattern raises ``SymbolNotGoldError``.  AurumX never silently trades
another instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.brokers.interface import BrokerInterface
from app.core.exceptions import (
    SymbolDiscoveryError,
    SymbolNotGoldError,
    SymbolVerificationError,
)
from app.core.logging import get_logger
from app.core.models import SymbolSpec, ValidationReport

log = get_logger("market.symbol_discovery")

#: Base names accepted as "the gold instrument" (quote vs USD only).
_GOLD_BASES: dict[str, int] = {
    "XAUUSD": 100,
    "XAU/USD": 90,
    "GOLD": 80,
    "GOLDUSD": 80,
}

#: Suffix score penalty for broker decorations: XAUUSDm scores below XAUUSD.
_SUFFIX_SCORE = {100: 85, 90: 80, 80: 70}

#: Broker decoration tokens seen in the wild — after a separator
#: ('.', '-', '_', '#', '+') e.g. ``XAUUSD.a``, ``GOLD-ECN``, or appended
#: directly e.g. ``XAUUSDm``, ``GOLDmicro``.
_KNOWN_SUFFIX_WORDS = frozenset(
    {
        "RAW", "PRO", "ECN", "STP", "MICRO", "MINI", "STD", "STANDARD",
        "SPOT", "CASH", "ZERO", "PLUS", "LIVE", "DEF", "DEFAULT", "MAIN",
    }
)

#: Single letters used as direct-append suffixes (``XAUUSDm``, ``GOLDp``).
_KNOWN_SUFFIX_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_SEPARATORS = {".", "-", "_", "#", "+"}

#: Other metals that must NEVER be treated as gold (defense in depth — these
#: would not match the gold bases anyway, but the denial is explicit).
_DENY_PREFIXES = ("XAG", "XPT", "XPD", "XCU", "XBR")


def _last_separator_index(name: str) -> int:
    return max(name.rfind(sep) for sep in _SEPARATORS)


def _strip_one_suffix(name: str, *, single_letter_allowed: bool) -> tuple[str, str] | None:
    """Strip one decoration token from ``name``; ``None`` when nothing
    recognizable can be stripped.  Returns ``(stripped_name, token)``.

    Rules (conservative by design):

    * a trailing separator alone is stripped (``XAUUSD+`` -> ``XAUUSD``);
    * a separator-delimited trailing token of 1-3 alnum chars is stripped
      (``XAUUSD.a``, ``GOLD-ECN``, ``XAUUSD.1``); longer/unknown tokens stop
      stripping (``XAUUSDGIBBERISH`` never becomes gold);
    * a known appended *word* is stripped (``XAUUSDmicro`` -> ``XAUUSD``);
    * a single appended letter is stripped at most once per name
      (``GOLDm`` -> ``GOLD`` is fine; ``GOLDEN`` -> ``GOLD`` is not).
    """
    if not name:
        return None
    if name[-1] in _SEPARATORS:
        return name[:-1], ""
    sep_idx = _last_separator_index(name)
    if sep_idx > 0:
        head, tail = name[:sep_idx], name[sep_idx + 1 :]
        if head and tail and len(tail) <= 3 and tail.isalnum():
            return head, tail
        return None
    for word in sorted(_KNOWN_SUFFIX_WORDS, key=len, reverse=True):
        if name.endswith(word) and len(name) > len(word):
            return name[: -len(word)], word
    if single_letter_allowed and name[-1] in _KNOWN_SUFFIX_LETTERS and len(name) > 1:
        return name[:-1], name[-1]
    return None


def _stripped_variants(name: str) -> list[str]:
    """Progressive decoration-stripped variants of ``name`` (max 3 tokens)."""
    variants: list[str] = []
    current = name
    single_letter_used = False
    for _ in range(3):
        result = _strip_one_suffix(current, single_letter_allowed=not single_letter_used)
        if result is None or not result[0]:
            break
        current, token = result
        if len(token) == 1 and token.isalpha():
            single_letter_used = True
        variants.append(current)
    return variants


def gold_match_score(name: str) -> int:
    """0 means *not gold*.  Higher scores are preferred candidates.

    >>> gold_match_score("XAUUSD")
    100
    >>> gold_match_score("XAUUSDm")
    85
    >>> gold_match_score("XAGUSD")
    0
    """
    if not name or not name.strip():
        return 0
    cleaned = name.upper().strip().replace(" ", "")
    if any(cleaned.startswith(prefix) for prefix in _DENY_PREFIXES):
        return 0
    if cleaned in _GOLD_BASES:
        return _GOLD_BASES[cleaned]
    slashless = cleaned.replace("/", "")
    if slashless in _GOLD_BASES:
        return _GOLD_BASES[slashless] - 5  # XAU/USD, GOLD/USD spellings
    for variant in _stripped_variants(cleaned):
        if variant in _GOLD_BASES:
            return _SUFFIX_SCORE.get(_GOLD_BASES[variant], 60)
        variant_slashless = variant.replace("/", "")
        if variant_slashless in _GOLD_BASES:
            return _SUFFIX_SCORE.get(_GOLD_BASES[variant_slashless], 60) - 5
    return 0


def is_gold_symbol(name: str) -> bool:
    """Gold-only protection check (spec §55)."""
    return gold_match_score(name) > 0


@dataclass
class RejectedCandidate:
    """A gold-named candidate that failed verification, with the reasons."""

    name: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SymbolDiscoveryResult:
    """Outcome of a discovery run — fully auditable."""

    chosen: SymbolSpec
    method: str  # "auto" | "explicit"
    score: int = 0
    alternatives: list[SymbolSpec] = field(default_factory=list)
    rejected: list[RejectedCandidate] = field(default_factory=list)
    report: ValidationReport = field(default_factory=ValidationReport)

    def summary(self) -> dict[str, object]:
        return {
            "symbol": self.chosen.name,
            "method": self.method,
            "score": self.score,
            "alternatives": [s.name for s in self.alternatives],
            "rejected": {r.name: r.reasons for r in self.rejected},
        }


class SymbolDiscovery:
    """Discovers and verifies the broker's gold symbol."""

    def __init__(self, broker: BrokerInterface) -> None:
        self._broker = broker

    # ------------------------------------------------------------------
    def discover(self, configured: str = "AUTO") -> SymbolDiscoveryResult:
        """``configured`` is the ``SYMBOL`` config value: ``AUTO`` or an exact
        broker symbol name."""
        if configured and configured.strip().upper() != "AUTO":
            return self._discover_explicit(configured.strip())
        return self._discover_auto()

    # ------------------------------------------------------------------
    def _discover_explicit(self, name: str) -> SymbolDiscoveryResult:
        # Gold-only protection: reject non-gold symbols unconditionally.
        score = gold_match_score(name)
        if score == 0:
            raise SymbolNotGoldError(
                f"Configured symbol {name!r} is not a gold (XAUUSD-equivalent) symbol. "
                "AurumX v1 trades gold only (spec §55) — refusing."
            )
        spec = self._fetch_visible(name)
        if spec is None:
            raise SymbolDiscoveryError(
                f"Configured symbol {name!r} was not found at the broker."
            )
        report = spec.validate()
        if not report.ok:
            raise SymbolVerificationError(
                f"Configured symbol {name!r} failed broker verification: {report.summary()}"
            )
        result = SymbolDiscoveryResult(chosen=spec, method="explicit", score=score, report=report)
        self._log_success(result)
        return result

    # ------------------------------------------------------------------
    def _discover_auto(self) -> SymbolDiscoveryResult:
        names = self._broker.list_symbols()
        scored: list[tuple[int, str]] = []
        for name in names:
            score = gold_match_score(name)
            if score > 0:
                scored.append((score, name))
        if not scored:
            raise SymbolDiscoveryError(
                "No gold symbol (XAUUSD/GOLD equivalent) found at this broker. "
                f"Scanned {len(names)} symbols."
            )

        # Prefer the highest score; among equals prefer already-visible symbols.
        scored.sort(key=lambda item: (-item[0], item[1]))

        rejected: list[RejectedCandidate] = []
        verified: list[SymbolSpec] = []
        chosen: SymbolSpec | None = None
        chosen_score = 0

        for score, name in scored:
            spec = self._fetch_visible(name)
            if spec is None:
                rejected.append(RejectedCandidate(name=name, reasons=["not found at broker"]))
                continue
            report = spec.validate()
            if report.ok:
                if chosen is None:
                    chosen, chosen_score = spec, score
                    # keep verifying remaining candidates for the audit trail
                else:
                    verified.append(spec)
            else:
                rejected.append(RejectedCandidate(name=name, reasons=[i.message for i in report.errors]))

        if chosen is None:
            reasons = "; ".join(f"{r.name}: {', '.join(r.reasons)}" for r in rejected)
            raise SymbolVerificationError(
                f"Gold symbol candidates found but none passed broker verification. {reasons}"
            )

        result = SymbolDiscoveryResult(
            chosen=chosen,
            method="auto",
            score=chosen_score,
            alternatives=verified,
            rejected=rejected,
        )
        self._log_success(result)
        return result

    # ------------------------------------------------------------------
    def _fetch_visible(self, name: str) -> SymbolSpec | None:
        """Fetch the symbol spec, trying to make it visible if it isn't."""
        spec = self._broker.get_symbol(name)
        if spec is None:
            return None
        if not spec.visible and self._broker.select_symbol(name):
            spec = self._broker.get_symbol(name)
        return spec

    def _log_success(self, result: SymbolDiscoveryResult) -> None:
        log.info(
            "gold symbol verified",
            event="SYMBOL_DISCOVERED",
            **result.summary(),  # type: ignore[arg-type]
        )
