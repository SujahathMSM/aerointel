"""Check an answer's numbers against the passages it cites.

A language model that has been given the right passage can still misquote it:
0.032 becomes 0.041, "inch-pounds" becomes "foot-pounds". In maintenance work
that is the dangerous failure, and it is checkable without another model call:
every number in the answer must occur in a passage the answer cites, with a
compatible unit.

What this does *not* catch: a number copied correctly from the wrong passage,
the wrong condition or aircraft, or a number the passage gives in a different
form (3/8 versus 0.375 is reported, on purpose: the prompt asks for exact
figures). Numbers that are part of an identifier ("AN960", "A320") and numbers
the question itself contains are not checked. Passage numbers and answer numbers
go through the same extraction, so identifier digits never cause a mismatch.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

# Unit spellings -> one canonical name (torque units are handled in _unit, and
# listed first in the pattern so "inch-pounds" is one unit, not "inch" + "pounds").
_UNIT_CANON = {
    "inch": "in",
    "in": "in",
    "foot": "ft",
    "feet": "ft",
    "ft": "ft",
    "pound": "lb",
    "lb": "lb",
    "psi": "psi",
    "percent": "%",
    "%": "%",
    "mm": "mm",
    "cm": "cm",
    "kg": "kg",
    "kilogram": "kg",
    "ounce": "oz",
    "oz": "oz",
    "hour": "h",
    "hr": "h",
    "minute": "min",
    "min": "min",
    "second": "s",
    "sec": "s",
    "volt": "V",
    "amp": "A",
    "ampere": "A",
    "ohm": "ohm",
    "mil": "mil",
    "gallon": "gal",
    "gal": "gal",
    "knot": "kt",
    "rpm": "rpm",
    "bar": "bar",
    "kpa": "kPa",
    "mpa": "MPa",
}

_UNIT_RE = (
    r"(?:inch[- ]pounds?|in\.?-lbs?\.?|foot[- ]pounds?|ft\.?-lbs?\.?"
    r"|inches|inch|in\.(?=\W|$)|in(?![\w.])(?!\s+[A-Za-z])"
    r"|feet|foot|ft\.?(?![\w])|pounds?|lbs?\.?(?![\w])"
    r"|psi|percent|%|degrees?\s*[FC]\b|°\s?[FC]\b"
    r"|mm|cm|kg|kilograms?|ounces?|oz|hours?|hrs?|minutes?|min(?![\w])"
    r"|seconds?|sec(?![\w])|volts?|amperes?|amps?|ohms?|mils?|gallons?|gal"
    r"|knots?|rpm|bar|kpa|mpa)"
)

_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\w.])"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?|\.\d+)"
    rf"(?:\s*-?\s*(?P<unit>{_UNIT_RE}))?",
    re.IGNORECASE,
)
# "min" after a number is minutes in running text ("dwell time 10 min") and
# minimum in tables ("0.057 min."). A passage's "min" is therefore compatible
# with any unit that is not a time; against another time unit it still counts.
_AMBIGUOUS_MIN = "min"
_TIME_UNITS = {"h", "min", "s"}

_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,\-–]\s*\d+)*)\s*\]")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")

_VULGAR = {
    "½": "1/2",
    "¼": "1/4",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}


@dataclass
class Verification:
    """What the check found. `ok` means nothing was unsupported or mismatched."""

    checked: int = 0
    cited: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    unit_mismatch: list[str] = field(default_factory=list)
    bad_citations: list[int] = field(default_factory=list)
    # Unsupported figures that *do* occur in a block the answer did not cite:
    # figure as written -> those block numbers. Usually a wrong citation (the
    # answer read block 3, cited block 2), not an invented number. The figure
    # stays in `unsupported`: the cited block still does not back it.
    found_in_uncited: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.unsupported or self.unit_mismatch or self.bad_citations)


def _clean(text: str) -> str:
    for glyph, plain in _VULGAR.items():
        text = text.replace(glyph, plain)
    return text.replace(" ", " ").replace("−", "-").replace("–", "-").replace("—", "-")


def _value(raw: str) -> str:
    """Canonical key for a number: 0.50 == .5 == 0.5; fractions stay as written."""
    raw = raw.replace(",", "")
    if "/" in raw:
        return raw
    try:
        return format(Decimal(raw).normalize(), "f")
    except InvalidOperation:  # pragma: no cover - the regex only yields numbers
        return raw


def _unit(raw: str | None) -> str | None:
    """Canonical unit name: "inches" / "in." / "inch" -> "in"; degrees -> "°F"."""
    if not raw:
        return None
    low = re.sub(r"\s+", " ", raw.lower().strip())
    if low.startswith(("°", "degree")):
        return "°" + low[-1].upper()
    if re.fullmatch(r"(inch[- ]pounds?|in\.?-lbs?\.?)", low):
        return "in-lb"
    if re.fullmatch(r"(foot[- ]pounds?|ft\.?-lbs?\.?)", low):
        return "ft-lb"
    low = low.rstrip(".")
    if low == "inches":
        return "in"
    if low not in _UNIT_CANON and low.endswith("s"):
        low = low[:-1]
    return _UNIT_CANON.get(low, low)


def numbers_in(text: str) -> list[tuple[str, str | None, str]]:
    """Every number in `text` as (canonical value, canonical unit or None, as written)."""
    text = _clean(_CITATION.sub(" ", _LIST_MARKER.sub("", text)))
    matches = list(_NUMBER_WITH_UNIT.finditer(text))
    found = []
    for i, m in enumerate(matches):
        unit = _unit(m.group("unit"))
        # "20-25 inch-pounds": the unit belongs to both ends of the range
        if unit is None and i + 1 < len(matches):
            nxt = matches[i + 1]
            if text[m.end() : nxt.start()].strip().lower() in ("-", "to"):
                unit = _unit(nxt.group("unit"))
        found.append((_value(m.group("num")), unit, m.group(0).strip()))
    return found


def parse_citations(answer: str) -> list[int]:
    """Block numbers cited as [1], [2, 3] or [1-3], in order, without repeats."""
    seen: dict[int, None] = {}
    for m in _CITATION.finditer(_clean(answer)):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if "-" in part:
                lo, _, hi = part.partition("-")
                if lo.strip().isdigit() and hi.strip().isdigit() and int(hi) >= int(lo):
                    for n in range(int(lo), min(int(hi), int(lo) + 50) + 1):
                        seen.setdefault(n, None)
            elif part.strip().isdigit():
                seen.setdefault(int(part), None)
    return list(seen)


def verify_answer(answer: str, passages: list[str], question: str = "") -> Verification:
    """Compare the answer's numbers with the passages it cites.

    `passages[i]` is context block i+1. If the answer cites nothing valid, every
    passage counts as cited (the model was given only those). Numbers that the
    question itself contains are accepted: an answer may restate them.
    """
    cited = parse_citations(answer)
    bad = [n for n in cited if not 1 <= n <= len(passages)]
    good = [n for n in cited if 1 <= n <= len(passages)]
    used = [passages[n - 1] for n in good] if good else list(passages)

    source_units: dict[str, set[str]] = {}
    for passage in used:
        for value, unit, _written in numbers_in(passage):
            source_units.setdefault(value, set())
            if unit:
                source_units[value].add(unit)
    echoed = {value for value, _u, _w in numbers_in(question)}

    uncited = {
        n: {value for value, _u, _w in numbers_in(passages[n - 1])}
        for n in range(1, len(passages) + 1)
        if good and n not in good
    }

    result = Verification(cited=good, bad_citations=bad)
    for value, unit, written in numbers_in(answer):
        result.checked += 1
        if value in echoed:
            continue
        if value not in source_units:
            if written not in result.unsupported:
                result.unsupported.append(written)
                where = [n for n, found in uncited.items() if value in found]
                if where:
                    result.found_in_uncited[written] = where
        elif (
            unit
            and source_units[value]
            and not _units_compatible(unit, source_units[value])
            and written not in result.unit_mismatch
        ):
            result.unit_mismatch.append(written)
    return result


def _units_compatible(unit: str, source_units: set[str]) -> bool:
    """Is the answer's unit one the passage gives this number?"""
    if unit in source_units:
        return True
    return _AMBIGUOUS_MIN in source_units and unit not in _TIME_UNITS
