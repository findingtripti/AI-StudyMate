import re


# ---------------------------------------------------------
# Unit heading detection
# ---------------------------------------------------------

UNIT_PATTERNS = [
    # UNIT 1 / UNIT-1 / UNIT: 1 / UNIT 01
    re.compile(
        r"^\s*UNIT\s*[-:–—]?\s*(\d{1,2})\s*$",
        re.IGNORECASE
    ),

    # UNIT I / UNIT-II / UNIT: III
    re.compile(
        r"^\s*UNIT\s*[-:–—:]?\s*"
        r"(I|II|III|IV|V|VI|VII|VIII|IX|X)\s*$",
        re.IGNORECASE
    ),

    # Unit 1: Introduction
    # Unit-2 - Neural Networks
    re.compile(
        r"^\s*UNIT\s*[-:–—:]?\s*(\d{1,2})"
        r"\s*[-:–—:]?\s+.+$",
        re.IGNORECASE
    ),

    # Unit I: Introduction
    re.compile(
        r"^\s*UNIT\s*[-:–—:]?\s*"
        r"(I|II|III|IV|V|VI|VII|VIII|IX|X)"
        r"\s*[-:–—:]?\s+.+$",
        re.IGNORECASE
    ),
]


ROMAN_TO_NUMBER = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
}


def _clean_line(line):
    """Clean extracted PDF text before checking headings."""

    if not line:
        return ""

    line = str(line)
    line = line.replace("\r", " ")
    line = re.sub(r"\s+", " ", line)

    return line.strip()


def _extract_unit_number(line):
    """
    Detect a unit heading and return its number.

    Examples:
        UNIT 1
        UNIT-2
        Unit 3: Neural Networks
        UNIT IV
        UNIT-V: Optimization

    Returns:
        int or None
    """

    line = _clean_line(line)

    if not line:
        return None

    # First check Arabic numbers.
    arabic_match = re.match(
        r"^\s*UNIT\s*[-:–—]?\s*(\d{1,2})"
        r"(?:\s*[-:–—:]?\s+.*)?$",
        line,
        re.IGNORECASE
    )

    if arabic_match:
        return int(arabic_match.group(1))

    # Then check Roman numerals.
    roman_match = re.match(
        r"^\s*UNIT\s*[-:–—:]?\s*"
        r"(I|II|III|IV|V|VI|VII|VIII|IX|X)"
        r"(?:\s*[-:–—:]?\s+.*)?$",
        line,
        re.IGNORECASE
    )

    if roman_match:
        roman = roman_match.group(1).upper()
        return ROMAN_TO_NUMBER.get(roman)

    return None


def detect_unit_headings(pages_data):
    """
    Detect unit headings from page-based PDF data.

    Expected pages_data format:

    [
        {
            "page": 1,
            "text": "...",
            "source": "file.pdf"
        },
        ...
    ]

    Returns a list like:

    [
        {
            "unit": 1,
            "page": 1,
            "source": "file.pdf",
            "heading": "UNIT 1"
        },
        ...
    ]
    """

    detected = []

    for page in pages_data:
        text = page.get("text", "")

        if not text:
            continue

        lines = str(text).splitlines()

        for line in lines:
            cleaned = _clean_line(line)

            if not cleaned:
                continue

            unit_number = _extract_unit_number(cleaned)

            if unit_number is None:
                continue

            detected.append({
                "unit": unit_number,
                "page": page.get("page", 0),
                "source": page.get("source", "Unknown"),
                "heading": cleaned,
            })

    return detected


def group_pages_by_unit(pages_data):
    """
    Divide PDF pages into units.

    If unit headings are detected:

        Unit 1 -> content until Unit 2
        Unit 2 -> content until Unit 3
        etc.

    If no unit headings are detected:

        returns one group named 'Full Document'

    Returns:

    {
        "Unit 1": [...pages...],
        "Unit 2": [...pages...],
        ...
    }
    """

    if not pages_data:
        return {}

    headings = detect_unit_headings(pages_data)

    # Remove duplicate detections of the same unit on
    # the same page.
    unique_headings = []
    seen = set()

    for item in headings:
        key = (
            item["unit"],
            item["page"],
            item["source"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique_headings.append(item)

    # Sort by source and page.
    unique_headings.sort(
        key=lambda item: (
            item["source"],
            item["page"],
            item["unit"]
        )
    )

    # No units detected.
    if not unique_headings:
        return {
            "Full Document": pages_data
        }

    groups = {}

    # Process each source independently.
    sources = []

    for page in pages_data:
        source = page.get("source", "Unknown")

        if source not in sources:
            sources.append(source)

    for source in sources:

        source_pages = [
            page
            for page in pages_data
            if page.get("source", "Unknown") == source
        ]

        source_pages.sort(
            key=lambda page: page.get("page", 0)
        )

        source_headings = [
            item
            for item in unique_headings
            if item["source"] == source
        ]

        if not source_headings:
            groups.setdefault(
                "Full Document",
                []
            ).extend(source_pages)

            continue

        for index, heading in enumerate(source_headings):

            unit_number = heading["unit"]

            start_page = heading["page"]

            if index + 1 < len(source_headings):
                end_page = (
                    source_headings[index + 1]["page"] - 1
                )
            else:
                end_page = source_pages[-1].get(
                    "page",
                    start_page
                )

            unit_pages = [
                page
                for page in source_pages
                if start_page <= page.get("page", 0) <= end_page
            ]

            unit_name = f"Unit {unit_number}"

            if unit_name not in groups:
                groups[unit_name] = []

            groups[unit_name].extend(unit_pages)

    # Remove empty groups.
    groups = {
        name: pages
        for name, pages in groups.items()
        if pages
    }

    return groups


def get_available_units(pages_data):
    """
    Return the detected unit names.

    Example:

        ['Unit 1', 'Unit 2', 'Unit 3', 'Unit 4', 'Unit 5']

    For a one-unit PDF:

        ['Unit 1']

    If no units are detected:

        ['Full Document']
    """

    groups = group_pages_by_unit(pages_data)

    return list(groups.keys())


def get_unit_text(pages_data, unit_name):
    """
    Return all text belonging to a selected unit.

    Example:

        get_unit_text(pages_data, "Unit 4")
    """

    groups = group_pages_by_unit(pages_data)

    selected_pages = groups.get(unit_name, [])

    if not selected_pages:
        return ""

    text_parts = []

    for page in selected_pages:
        text = page.get("text", "").strip()

        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts).strip()