#!/usr/bin/env python3
"""Export the local knowledge base PDFs under original_papers/ to AI-friendly Markdown."""

from __future__ import annotations

import argparse
import csv
import html
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("original_papers")
DEFAULT_INVENTORY = "knowledge_base_inventory.csv"
DEFAULT_OUTPUT_DIR = "markdown"

CAPTION_RE = re.compile(r"^(Figure|Fig\.?|Table)\s*[A-Za-z0-9\.\-:]*", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^(Table)\s*[A-Za-z0-9\.\-:]*", re.IGNORECASE)
INLINE_CAPTION_RE = re.compile(r"(Figure|Fig\.?|Table)\s*\d+[A-Za-z0-9\.\-:]*", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^(\d+(\.\d+)*\s+[A-Z][A-Za-z].{0,80}|Abstract|ABSTRACT|References|REFERENCES|Acknowledg(e)?ments?|Appendix.*)$"
)
ROMAN_HEADING_RE = re.compile(r"^[IVXLC]+\.\s+[A-Z][A-Za-z].{0,80}$")
TITLE_SKIP_RE = re.compile(
    r"^(Published as |Latest updates:|RESEARCH-ARTICLE|PDF Download|Accepted:|Received:|"
    r"Total Citations:|Total Downloads:|Citation in BibTeX format|Open Access Support|provided by:)"
)
LINE_SKIP_RE = re.compile(
    r"^(PDF Download|Total Citations:|Total Downloads:|Accepted:|Received:|Citation in BibTeX format|"
    r"Open Access Support|provided by:|ACM Transactions on Embedded Computing Systems|EISSN:|"
    r"ACM\b.*|ACM Trans\..*|http://dx\.doi\.org/.*|doi\.org/.*|arXiv:.*|RESEARCH-ARTICLE|"
    r"\d{6,}\.pdf|\d{1,2} [A-Z][a-z]+ \d{4}|This work has been submitted to the IEEE\b.*|"
    r"Manuscript received\b.*|Date of publication\b.*|Color versions of one or more of the figures\b.*|"
    r"Digital Object Identifier\b.*|\(Corresponding author:.*|This paper was approved by Associate Editor\b.*|accessible\.)$",
    re.IGNORECASE,
)
SECTION_START_RE = re.compile(r"^\d+(\.\d+)*\s+")
ROMAN_SECTION_START_RE = re.compile(r"^[IVXLC]+\.\s+")
FRONT_MATTER_LABEL_RE = re.compile(
    r"^(Abstract|CCS CONCEPTS|Additional Keywords and Phrases|Keywords|Index Terms)\b(?:\s*[:•—-]\s*(.*))?$",
    re.IGNORECASE,
)
AFFILIATION_HINT_RE = re.compile(
    r"\b("
    r"University|College|Institute|School|Department|Laborator(?:y|ies)|Faculty|Center|Centre|"
    r"Engineering|Computer Science|Electrical|Research|Campus|Hospital|Academy|Laboratory|"
    r"State University|National|Apple|Google|Microsoft|Meta|Amazon|NVIDIA|Intel|IBM|OpenAI"
    r")\b",
    re.IGNORECASE,
)
AUTHOR_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "AS",
    "AT",
    "BY",
    "FOR",
    "FROM",
    "IN",
    "INTO",
    "OF",
    "ON",
    "THE",
    "TO",
    "USING",
    "VIA",
    "WITH",
}
SECTION_KEYWORDS = {
    "ABSTRACT",
    "ACKNOWLEDGMENTS",
    "APPENDIX",
    "CONCEPTS",
    "CONCLUSION",
    "DISCUSSION",
    "EXPERIMENTS",
    "INTRODUCTION",
    "KEYWORDS",
    "METHOD",
    "METHODS",
    "REFERENCES",
    "RESULTS",
}
KNOWN_TEXT_REPLACEMENTS = (
    (r"\bCNNand\b", "CNN- and"),
    (r"\binferencerelated\b", "inference-related"),
    (r"\bSpatiallevel\b", "Spatial-level"),
    (r"\blongdistance\b", "long-distance"),
    (r"\bhighdimensional\b", "high-dimensional"),
    (r"\bdigitaltoanalog\b", "digital-to-analog"),
    (r"\bMOBILE-FRIENDLYVISION\b", "MOBILE-FRIENDLY VISION"),
    (r"\bMobile-FriendlyVISION\b", "Mobile-Friendly VISION"),
    (r"\bgener-[A-Za-z0-9\-\s]{0,40}alization\b", "generalization"),
)
LONG_FORM_CHUNK_SPECS = {
    "08_books_and_design_references/Silicon_Photonics_Design_From_Devices_to_Systems_CN_Translation.pdf": {
        "index_dirname": "Silicon_Photonics_Design_From_Devices_to_Systems_CN_Translation__chapters",
        "preferred_usage": (
            "Prefer the chapter index and per-chapter Markdown slices for AI reading. "
            "Use the full-book Markdown only for cross-chapter search, audit, or recovery."
        ),
        "sections": [
            {
                "slug": "00_front_matter_and_contents",
                "label": "Front Matter and Contents",
                "part": "Front Matter",
                "source_page_start": 2,
            },
            {
                "slug": "01_chapter_01_fabless_silicon_photonics",
                "label": "Chapter 1 Fabless Silicon Photonics",
                "part": "Part I Introduction",
                "source_page_start": 21,
                "book_page_start": 3,
                "book_page_end": 26,
            },
            {
                "slug": "02_chapter_02_silicon_photonics_modeling_and_design_methods",
                "label": "Chapter 2 Silicon Photonics Modeling and Design Methods",
                "part": "Part I Introduction",
                "source_page_start": 46,
                "book_page_start": 27,
                "book_page_end": 46,
            },
            {
                "slug": "03_chapter_03_optical_materials_and_waveguides",
                "label": "Chapter 3 Optical Materials and Waveguides",
                "part": "Part II Passive Photonic Devices",
                "source_page_start": 63,
                "book_page_start": 47,
                "book_page_end": 92,
            },
            {
                "slug": "04_chapter_04_photonic_device_modeling_fundamentals",
                "label": "Chapter 4 Fundamentals of Photonic Device Modeling",
                "part": "Part II Passive Photonic Devices",
                "source_page_start": 110,
                "book_page_start": 93,
                "book_page_end": 164,
            },
            {
                "slug": "05_chapter_05_optical_input_output",
                "label": "Chapter 5 Optical Input / Output",
                "part": "Part II Passive Photonic Devices",
                "source_page_start": 182,
                "book_page_start": 165,
                "book_page_end": 222,
            },
            {
                "slug": "06_chapter_06_optical_modulators",
                "label": "Chapter 6 Optical Modulators",
                "part": "Part III Active Photonic Devices",
                "source_page_start": 237,
                "book_page_start": 223,
                "book_page_end": 267,
            },
            {
                "slug": "07_chapter_07_photodetectors",
                "label": "Chapter 7 Photodetectors",
                "part": "Part III Active Photonic Devices",
                "source_page_start": 283,
                "book_page_start": 268,
                "book_page_end": 304,
            },
            {
                "slug": "08_chapter_08_lasers",
                "label": "Chapter 8 Lasers",
                "part": "Part III Active Photonic Devices",
                "source_page_start": 320,
                "book_page_start": 305,
                "book_page_end": 322,
            },
            {
                "slug": "09_chapter_09_silicon_photonic_circuit_modeling",
                "label": "Chapter 9 Silicon Photonic Circuit Modeling",
                "part": "Part IV System Design",
                "source_page_start": 335,
                "book_page_start": 323,
                "book_page_end": 362,
            },
            {
                "slug": "10_chapter_10_design_tools_and_techniques",
                "label": "Chapter 10 Silicon Photonics Design Tools and Techniques",
                "part": "Part IV System Design",
                "source_page_start": 376,
                "book_page_start": 363,
                "book_page_end": 382,
            },
            {
                "slug": "11_chapter_11_silicon_photonic_wafer_fabrication",
                "label": "Chapter 11 Silicon Photonic Wafer Fabrication",
                "part": "Part IV System Design",
                "source_page_start": 396,
                "book_page_start": 383,
                "book_page_end": 394,
            },
            {
                "slug": "12_chapter_12_testing_and_packaging",
                "label": "Chapter 12 Silicon Photonics Testing and Packaging",
                "part": "Part IV System Design",
                "source_page_start": 408,
                "book_page_start": 395,
                "book_page_end": 417,
            },
            {
                "slug": "13_chapter_13_system_examples",
                "label": "Chapter 13 Silicon Photonics System Examples",
                "part": "Part IV System Design",
                "source_page_start": 431,
                "book_page_start": 418,
                "book_page_end": 423,
            },
        ],
    }
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _long_form_chunk_spec(row: dict[str, str]) -> dict[str, object] | None:
    return LONG_FORM_CHUNK_SPECS.get(row["relative_path"])


def _full_markdown_relative_path(row: dict[str, str]) -> Path:
    return Path(row["relative_path"]).with_suffix(".md")


def _primary_markdown_relative_path(row: dict[str, str]) -> Path:
    spec = _long_form_chunk_spec(row)
    if spec is None:
        return _full_markdown_relative_path(row)
    return Path(row["theme_dir"]) / str(spec["index_dirname"]) / "INDEX.md"


def _run_command(args: list[str]) -> str:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"command failed: {' '.join(args)}\n{stderr}")
    return result.stdout.decode("utf-8", errors="replace")


def _clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("hps://", "https://")
    text = re.sub(r"(Figure\s+\d+)[μλ]\s*", r"\1: ", text)
    text = re.sub(r"(Table\s+\d+)[μλ]\s*", r"\1: ", text)
    text = re.sub(r"\b\w*arXiv:\S+\s+\[[^\]]+\]\s+\d+\s+[A-Za-z]{3}\s+\d{4}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\barXiv:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ACM\s+Trans\.?\s*Embed\.?\s*Comput\.?\s*Syst\.?", "", text)
    text = re.sub(r"http://dx\.doi\.org/\S+", "", text)
    text = re.sub(
        r"Month Date, Year\.\s+Date of publication Month Date, Year; date of current version Month Date, Year\.\s+This paper was approved by Associate Editor Name",
        "",
        text,
    )
    text = re.sub(
        r"This work has been submitted to the IEEE for possible publication\.\s*Copyright may be transferred without notice, after which this version may no longer be\s*\d*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"Digital Object Identifier 10\.1109/\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([A-Za-z])\s*-\s*([A-Za-z])", r"\1-\2", text)
    text = re.sub(r"([,;])(?=\S)", r"\1 ", text)
    text = re.sub(r":(?=[A-Za-z])", ": ", text)
    text = re.sub(r"\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"\b([A-Z])\s+([A-Z]{2,}[a-z])", r"\1\2", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = _repair_token_boundaries(_repair_spaced_caps(text))
    return _normalize_known_text(text)


def _repair_spaced_caps(text: str) -> str:
    tokens = text.split()
    if len(tokens) < 2:
        return text
    repaired: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        normalized = re.sub(r"[^A-Z]", "", token)
        if len(normalized) == 1 and normalized.isalpha() and not re.search(r"[a-z]", token):
            merged = token
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                nxt_alpha = re.sub(r"[^A-Z]", "", nxt)
                if len(nxt_alpha) == 1 and nxt_alpha.isalpha():
                    merged += nxt
                    j += 1
                    continue
                if (
                    nxt_alpha
                    and nxt_alpha == re.sub(r"[^A-Za-z]", "", nxt).upper()
                    and len(nxt_alpha) <= 14
                    and not re.search(r"[a-z]", nxt)
                ):
                    merged += nxt
                    j += 1
                    continue
                break
            repaired.append(merged)
            i = j
            continue
        repaired.append(token)
        i += 1
    return " ".join(repaired)


def _repair_token_boundaries(text: str) -> str:
    camel_boundary = re.compile(r"\b([A-Za-z]+?)([A-Z][a-z][A-Za-z0-9]*)\b")

    def split_token(match: re.Match[str]) -> str:
        prefix = match.group(1)
        suffix = match.group(2)
        suffix_letters = len(re.sub(r"[^A-Za-z]", "", suffix))
        if len(prefix) == 1 and suffix_letters >= 5:
            return f"{prefix} {suffix}"
        if prefix.isupper() and len(prefix) <= 3:
            return prefix + suffix
        if (len(prefix) >= 5 and suffix_letters >= 4) or (len(prefix) >= 4 and suffix_letters >= 6):
            return f"{prefix} {suffix}"
        return prefix + suffix

    repaired = text
    while True:
        next_text = camel_boundary.sub(split_token, repaired)
        if next_text == repaired:
            break
        repaired = next_text
    repaired = re.sub(
        r"\b([A-Z]{5,}?)(TRANSFORMER|TRANSFORMERS|NETWORK|NETWORKS|COMPUTING|PHOTONICS|METHODOLOGIES)\b",
        r"\1 \2",
        repaired,
    )
    repaired = re.sub(r"\s+", " ", repaired)
    return repaired.strip()


def _normalize_known_text(text: str) -> str:
    normalized = text
    for pattern, replacement in KNOWN_TEXT_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _split_layout_line_chunks(raw_line: str) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=(?:\s{8,}\S)|$)", raw_line.rstrip()):
        chunk = match.group(0).strip()
        if chunk:
            chunks.append((match.start(), chunk))
    return chunks


def _reflow_layout_page(raw_lines: list[str]) -> list[str]:
    chunk_rows = [_split_layout_line_chunks(line) for line in raw_lines]
    right_starts = [chunks[1][0] for chunks in chunk_rows if len(chunks) >= 2]
    if len(right_starts) < 6:
        return [_clean_text(line.rstrip()) for line in raw_lines]
    right_start = sorted(right_starts)[len(right_starts) // 2]
    if right_start < 40:
        return [_clean_text(line.rstrip()) for line in raw_lines]

    first_multi = next((idx for idx, chunks in enumerate(chunk_rows) if len(chunks) >= 2), None)
    if first_multi is None:
        return [_clean_text(line.rstrip()) for line in raw_lines]

    prefix: list[str] = []
    left_column: list[str] = []
    right_column: list[str] = []

    for idx, chunks in enumerate(chunk_rows):
        if not chunks:
            continue
        if idx < first_multi and len(chunks) == 1:
            prefix.append(_clean_text(chunks[0][1]))
            continue
        if len(chunks) >= 2:
            left_column.append(_clean_text(chunks[0][1]))
            right_text = _clean_text(" ".join(chunk for _, chunk in chunks[1:]))
            if right_text:
                right_column.append(right_text)
            continue
        start, text = chunks[0]
        cleaned = _clean_text(text)
        if start >= right_start - 4:
            right_column.append(cleaned)
        else:
            left_column.append(cleaned)

    return prefix + left_column + right_column


def _extract_page_texts(pdf_path: Path) -> list[list[str]]:
    text = _run_command(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"])
    raw_pages = text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    if raw_pages and not raw_pages[-1].strip():
        raw_pages.pop()
    pages: list[list[str]] = []
    for raw_page in raw_pages:
        lines = _reflow_layout_page(raw_page.split("\n"))
        pages.append(lines)
    return pages


def _extract_image_counts(pdf_path: Path) -> list[int]:
    with tempfile.TemporaryDirectory(prefix="fyp_pdfxml_") as tmp_dir:
        xml_path = Path(tmp_dir) / "out.xml"
        _run_command(
            [
                "pdftohtml",
                "-xml",
                "-enc",
                "UTF-8",
                "-nodrm",
                str(pdf_path),
                str(xml_path),
            ]
        )
        root = ET.parse(xml_path).getroot()
    counts: list[int] = []
    for page_elem in root.findall("page"):
        page_height = int(page_elem.attrib["height"])
        usable = 0
        for image in page_elem.findall("image"):
            width = int(image.attrib["width"])
            height = int(image.attrib["height"])
            top = int(image.attrib["top"])
            area = width * height
            if area < 12000:
                continue
            if top < page_height * 0.2 and area < 45000:
                continue
            usable += 1
        counts.append(usable)
    return counts


def _is_heading_line(text: str) -> bool:
    if not text or CAPTION_RE.match(text):
        return False
    if _looks_like_author_line(text):
        return False
    if _split_front_matter_label(text) is not None:
        return True
    if _looks_like_structural_heading(text):
        return True
    if any(char.isdigit() for char in text):
        return False
    if len(text) > 90 or len(re.findall(r"\d", text)) >= 4:
        return False
    letters = [char for char in text if char.isalpha()]
    if 3 <= len(text) <= 120 and letters:
        upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        if upper_ratio > 0.7 and len(text.split()) <= 12:
            return True
    return False


def _is_skippable_line(text: str, *, page_number: int, total_pages: int) -> bool:
    if not text:
        return True
    if LINE_SKIP_RE.search(text):
        return True
    if TITLE_SKIP_RE.search(text):
        return True
    if re.fullmatch(r"\d+", text):
        return True
    if page_number > 1 and text in {"1", str(page_number), str(total_pages)}:
        return True
    return False


def _normalize_heading(text: str) -> str:
    cleaned = _clean_text(text)
    cleaned = re.sub(r"^([IVXLC]+)\.([A-Z])", r"\1. \2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _join_wrapped_lines(parts: list[str]) -> str:
    paragraph = ""
    for part in parts:
        text = _clean_text(part)
        if not text:
            continue
        if not paragraph:
            paragraph = text
            continue
        if paragraph.endswith("-") and text[0].islower():
            paragraph = paragraph[:-1] + text
        elif text[0] in ".,;:!?)]}%":
            paragraph += text
        elif paragraph.endswith(("(", "[", "{", "/")):
            paragraph += text
        else:
            paragraph += " " + text
    return _clean_text(paragraph)


def _looks_tabular(line: str) -> bool:
    if re.search(r"\s{2,}", line):
        return True
    digit_groups = len(re.findall(r"\d+(?:\.\d+)?", line))
    short_chunks = len([chunk for chunk in re.split(r"\s{2,}", line) if chunk.strip()])
    return digit_groups >= 3 and short_chunks >= 2


def _digit_group_count(text: str) -> int:
    return len(re.findall(r"\d+(?:\.\d+)?", text))


def _looks_like_structural_heading(text: str) -> bool:
    if _split_front_matter_label(text) is not None:
        return True
    normalized = re.sub(r"^([IVXLC]+)\.([A-Z])", r"\1. \2", text)
    if not (HEADING_RE.match(normalized) or SECTION_START_RE.match(normalized) or ROMAN_HEADING_RE.match(normalized)):
        return False
    roman_match = re.match(r"^([IVXLC]+)\.\s+(.*)$", normalized)
    if roman_match:
        remainder = roman_match.group(2).strip()
        if not remainder or len(remainder) > 70:
            return False
        if _digit_group_count(remainder) > 1:
            return False
        if any(symbol in remainder for symbol in ("<", ">", "=", "#")):
            return False
        return True
    match = re.match(r"^(\d+)(?:\.\d+)*\s+(.*)$", normalized)
    if not match:
        return HEADING_RE.match(normalized) is not None
    lead_num = int(match.group(1))
    remainder = match.group(2).strip()
    heading_words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", remainder)
    if not heading_words:
        return False
    first_alpha = next((char for char in remainder if char.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    if lead_num > 12:
        return False
    if len(remainder) > 60:
        return False
    if _digit_group_count(text) > 2:
        return False
    if any(symbol in remainder for symbol in ("<", ">", "=", "#")):
        return False
    return True


def _looks_like_author_line(text: str) -> bool:
    if not text or "@" in text or ":" in text or CAPTION_RE.match(text):
        return False
    if "IEEE" in text and text.count(",") >= 2:
        return True
    if any(char.isdigit() for char in text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\.\-]*", text)
    if not 2 <= len(words) <= 6:
        return False
    upper_words = {word.upper() for word in words}
    if upper_words & SECTION_KEYWORDS:
        return False
    if upper_words & AUTHOR_STOPWORDS:
        return False
    if any(len(word) > 18 for word in words):
        return False
    if not all(word[:1].isupper() for word in words):
        return False
    lower_words = sum(1 for word in words if any(char.islower() for char in word))
    return lower_words >= 2 or all(word.isupper() for word in words)


def _looks_like_affiliation_line(text: str, *, previous_authorish: bool) -> bool:
    if not text:
        return False
    if "@" in text:
        return True
    if AFFILIATION_HINT_RE.search(text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'\.\-]*", text)
    if previous_authorish and 1 <= len(words) <= 8 and not any(punct in text for punct in ".?!:;"):
        return True
    return False


def _looks_like_sentence_line(text: str) -> bool:
    if not text or CAPTION_RE.match(text) or _looks_tabular(text):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    if len(words) < 6:
        return False
    letters = sum(1 for char in text if char.isalpha())
    digits = sum(1 for char in text if char.isdigit())
    if letters < 20 or digits > max(3, len(words) // 3):
        return False
    lower_ratio = sum(1 for char in text if char.islower()) / max(letters, 1)
    return lower_ratio >= 0.18


def _looks_like_caption_continuation(text: str) -> bool:
    return (
        _looks_like_sentence_line(text)
        and _digit_group_count(text) <= 1
        and not re.match(r"^\d", text)
        and "Top 1 accuracy" not in text
        and "mAP" not in text
        and "MobileNet" not in text
        and "Mobile Net" not in text
    )


def _clean_caption_text(text: str) -> str:
    cleaned = _normalize_known_text(text)
    cleaned = re.sub(r"\bgener-[A-Za-z0-9\-\s]{0,40}alization\b", "generalization", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned.startswith(("Figure", "Fig.", "Fig ", "Table")):
        for marker in (" Better :", " Top 1 accuracy", " mAP", " GPU-", " MobileNetv", " Mobile Netv", " 70 71 ", " 60 60 "):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].rstrip(" ,;:-")
                break
        if "." in cleaned and (_digit_group_count(cleaned) > 2 or len(cleaned) > 220):
            first_sentence = cleaned.split(".", 1)[0].strip()
            if len(first_sentence.split()) >= 5:
                cleaned = first_sentence + "."
    return cleaned


def _cleanup_markdown_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        current = _normalize_known_text(line)
        if current.startswith("## "):
            body = current[3:].strip()
            first_alpha = next((char for char in body if char.isalpha()), "")
            suspicious_heading = (
                bool(re.match(r"^\d", body))
                and (
                    not first_alpha
                    or first_alpha.islower()
                    or any(symbol in body for symbol in ("<", ">", "=", "#"))
                    or len(body) > 70
                )
            )
            if suspicious_heading:
                current = body
        elif current.startswith("> "):
            body = current[2:].strip()
            if CAPTION_RE.match(body):
                current = f"> {_clean_caption_text(body)}"
        cleaned.append(current)

    squashed: list[str] = []
    for line in cleaned:
        if line == "" and squashed and squashed[-1] == "":
            continue
        squashed.append(line)
    return squashed


def _split_front_matter_label(text: str) -> tuple[str, str] | None:
    match = FRONT_MATTER_LABEL_RE.match(text)
    if not match:
        return None
    label = _repair_token_boundaries(match.group(1).title())
    body = (match.group(2) or "").replace("•", "; ").strip()
    body = re.sub(r"\s*;\s*", "; ", body)
    return label, body


def _split_inline_caption(text: str) -> list[str]:
    if not text:
        return [""]
    if CAPTION_RE.match(text):
        return [text]
    match = INLINE_CAPTION_RE.search(text)
    if not match:
        return [text]
    prefix = text[: match.start()].strip()
    caption = text[match.start() :].strip()
    if prefix and _looks_like_sentence_line(prefix):
        return [text]
    return [caption]


def _looks_like_title_line(text: str) -> bool:
    if not text or CAPTION_RE.match(text) or "@" in text:
        return False
    if _split_front_matter_label(text) is not None:
        return False
    if SECTION_START_RE.match(text) or HEADING_RE.match(text):
        return False
    if _looks_like_author_line(text):
        return False
    if re.search(r"(https?://|doi\.org)", text, re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 8 or not words or len(words) > 16:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    title_case_words = sum(1 for word in words if word[:1].isupper())
    return ":" in text or upper_ratio > 0.45 or title_case_words >= max(2, len(words) - 1)


def _title_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"^\d{4}\.\d{5}_", "", stem)
    stem = re.sub(r"_arXiv\d{4}\.\d{5}(?:v\d+)?$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip(" -_.")
    return _normalize_known_text(stem)


def _title_looks_suspicious(title: str) -> bool:
    lowered = title.lower()
    suspicious_phrases = (
        "section ",
        "respectively",
        "performance evaluation",
        "conclusion are summarized",
        "submitted to the ieee",
        "copyright may be transferred",
        "available:",
    )
    if len(title.split()) < 2:
        return True
    if any(phrase in lowered for phrase in suspicious_phrases):
        return True
    if lowered.startswith(("section ", "this work ", "available ", "figure ")):
        return True
    return False


def _title_token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)
        if token.lower() not in {"a", "an", "and", "arxiv", "at", "conference", "for", "in", "of", "on", "paper", "the", "to", "with"}
    }


def _select_title(extracted_title: str, row: dict[str, str], fallback: str) -> str:
    filename_title = _title_from_filename(row["file_name"])
    title_hint = _normalize_known_text(row.get("title_hint", ""))
    extracted = _normalize_known_text(extracted_title)

    if extracted and not _title_looks_suspicious(extracted):
        extracted_sig = re.sub(r"[^a-z0-9]", "", extracted.lower())
        filename_sig = re.sub(r"[^a-z0-9]", "", filename_title.lower())
        if filename_sig.startswith(extracted_sig) and len(filename_sig) - len(extracted_sig) >= 12:
            return filename_title
        extracted_tokens = _title_token_set(extracted)
        filename_tokens = _title_token_set(filename_title)
        overlap = (
            len(extracted_tokens & filename_tokens) / max(len(filename_tokens), 1)
            if filename_tokens
            else 1.0
        )
        if overlap >= 0.2 or len(extracted.split()) <= 4:
            return extracted

    if filename_title:
        return filename_title
    if title_hint and not _title_looks_suspicious(title_hint):
        return title_hint
    return fallback


def _extract_title(pages: list[list[str]], fallback: str) -> str:
    if not pages:
        return fallback
    candidates: list[tuple[int, list[str]]] = []
    total_pages = len(pages)
    for page_number, page in enumerate(pages[:2], start=1):
        page_lines = [
            _clean_text(raw_line)
            for raw_line in page[:60]
            if not _is_skippable_line(_clean_text(raw_line), page_number=page_number, total_pages=total_pages)
        ]
        current: list[str] = []
        for line in page_lines:
            if _looks_like_title_line(line):
                if current and len(current) >= 3:
                    candidates.append((page_number, current))
                    current = [line]
                else:
                    current.append(line)
                continue
            if current:
                candidates.append((page_number, current))
                current = []
            if _looks_like_author_line(line) or SECTION_START_RE.match(line) or "ABSTRACT" in line.upper():
                break
        if current:
            candidates.append((page_number, current))

    if candidates:
        scored = sorted(
            candidates,
            key=lambda item: (
                sum(len(re.findall(r"[A-Za-z]", part)) for part in item[1])
                + sum(part.count(" ") for part in item[1]) * 3
                + sum(15 for part in item[1] if ":" in part)
                + (30 if len(item[1]) > 1 else 0)
                - (item[0] - 1) * 2
            ),
            reverse=True,
        )
        title = _join_wrapped_lines(scored[0][1])
        title_sig = re.sub(r"[^a-z0-9]", "", title.lower())
        if title and "totalcitations" not in title_sig and len(title_sig) >= 8:
            return title

    for page_number, page in enumerate(pages[:2], start=1):
        for raw_line in page[:30]:
            line = _clean_text(raw_line)
            if _is_skippable_line(line, page_number=page_number, total_pages=total_pages):
                continue
            if _looks_like_title_line(line):
                return line
    return fallback


def _compact_figure_page(lines: list[str], image_count: int) -> list[str]:
    expanded_lines: list[str] = []
    for line in lines:
        expanded_lines.extend(_split_inline_caption(line))

    caption_count = sum(1 for line in expanded_lines if CAPTION_RE.match(line))
    short_count = sum(1 for line in lines if 0 < len(line) < 50)
    figure_dense = caption_count > 0 and short_count >= max(10, len(expanded_lines) // 3)
    if not (figure_dense or (image_count > 0 and short_count >= max(10, len(expanded_lines) // 3))):
        return expanded_lines

    caption_indexes = [idx for idx, line in enumerate(expanded_lines) if CAPTION_RE.match(line)]
    trailing_prose_start: int | None = None
    if caption_indexes:
        last_caption = caption_indexes[-1]
        for idx in range(last_caption + 1, len(expanded_lines) - 2):
            trio = [line for line in expanded_lines[idx : idx + 3] if line]
            if len(trio) == 3 and all(_looks_like_sentence_line(line) for line in trio):
                trailing_prose_start = idx
                break

    compacted: list[str] = []
    for idx, line in enumerate(expanded_lines):
        if not line:
            compacted.append("")
            continue
        if CAPTION_RE.match(line) or _looks_like_structural_heading(line):
            compacted.append(line)
            continue
        if trailing_prose_start is not None and idx >= trailing_prose_start and _looks_like_sentence_line(line):
            if idx == trailing_prose_start and compacted and compacted[-1] != "":
                compacted.append("")
            compacted.append(line)
            continue
        if any(caption_idx < idx <= caption_idx + 12 for caption_idx in caption_indexes) and _looks_like_caption_continuation(line):
            compacted.append(line)
            continue
        if idx < (caption_indexes[0] if caption_indexes else len(expanded_lines)) and _looks_like_caption_continuation(line):
            compacted.append(line)
            continue
        if line.lower().startswith(("results.", "discussion.", "note.")):
            compacted.append(line)
    return compacted


def _collapse_layout_breaks(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for index, line in enumerate(lines):
        if line != "":
            collapsed.append(line)
            continue

        prev_line = next((item for item in reversed(collapsed) if item), "")
        next_line = next((item for item in lines[index + 1 :] if item), "")
        if prev_line and next_line:
            wrapped_sentence = (
                prev_line.endswith("-")
                or prev_line.endswith((",", ";", ":", "("))
                or next_line[:1].islower()
                or (_looks_like_sentence_line(prev_line) and _looks_like_sentence_line(next_line))
            )
            if wrapped_sentence:
                continue
        if collapsed and collapsed[-1] == "":
            continue
        collapsed.append("")
    return collapsed


def _page_blocks(lines: list[str], *, page_number: int, total_pages: int, image_count: int) -> list[dict[str, str]]:
    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = _clean_text(raw_line)
        if not line:
            cleaned_lines.append("")
            continue
        if _is_skippable_line(line, page_number=page_number, total_pages=total_pages):
            continue
        cleaned_lines.extend(_split_inline_caption(line))
    cleaned_lines = _collapse_layout_breaks(cleaned_lines)
    cleaned_lines = _compact_figure_page(cleaned_lines, image_count)

    blocks: list[dict[str, str]] = []
    if image_count > 0:
        blocks.append(
            {
                "type": "note",
                "text": "Source page contains non-text figure/table regions in the original PDF.",
            }
        )

    i = 0
    paragraph_parts: list[str] = []
    saw_abstract = False
    saw_body_heading = False

    def flush_paragraph() -> None:
        if not paragraph_parts:
            return
        paragraph = _join_wrapped_lines(paragraph_parts)
        if paragraph:
            blocks.append({"type": "paragraph", "text": paragraph})
        paragraph_parts.clear()

    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        if not line:
            flush_paragraph()
            i += 1
            continue

        label_block = _split_front_matter_label(line)
        if label_block is not None:
            flush_paragraph()
            heading, body = label_block
            blocks.append({"type": "heading", "text": heading})
            if body:
                blocks.append({"type": "paragraph", "text": body})
            i += 1
            continue

        if _is_heading_line(line):
            flush_paragraph()
            heading = _normalize_heading(line)
            if "ABSTRACT" in heading.upper():
                saw_abstract = True
            if SECTION_START_RE.match(heading) or ROMAN_SECTION_START_RE.match(heading):
                saw_body_heading = True
            blocks.append({"type": "heading", "text": heading})
            i += 1
            continue

        previous_block_type = blocks[-1]["type"] if blocks else ""
        previous_authorish = previous_block_type in {"author", "meta"}
        if page_number <= 2 and not saw_body_heading and _looks_like_author_line(line):
            flush_paragraph()
            blocks.append({"type": "author", "text": line})
            i += 1
            continue

        if page_number <= 2 and not saw_body_heading and _looks_like_affiliation_line(
            line, previous_authorish=previous_authorish
        ):
            flush_paragraph()
            blocks.append({"type": "meta", "text": line})
            i += 1
            continue

        if CAPTION_RE.match(line):
            flush_paragraph()
            caption_parts = [line]
            j = i + 1
            while j < len(cleaned_lines):
                nxt = cleaned_lines[j]
                if not nxt or _is_heading_line(nxt) or CAPTION_RE.match(nxt):
                    break
                if len(nxt) < 12 and not _looks_tabular(nxt):
                    break
                caption_parts.append(nxt)
                j += 1
                if len(caption_parts) >= 4:
                    break
            blocks.append({"type": "caption", "text": _join_wrapped_lines(caption_parts)})

            if TABLE_CAPTION_RE.match(line):
                table_lines: list[str] = []
                while j < len(cleaned_lines):
                    nxt = cleaned_lines[j]
                    if not nxt:
                        break
                    if _is_heading_line(nxt) or CAPTION_RE.match(nxt):
                        break
                    if not _looks_tabular(nxt):
                        break
                    table_lines.append(nxt)
                    j += 1
                if table_lines:
                    blocks.append({"type": "table", "text": "\n".join(table_lines)})

            i = j
            continue

        if (
            page_number <= 2
            and not saw_body_heading
            and not saw_abstract
            and blocks
            and blocks[-1]["type"] in {"author", "meta"}
            and _looks_like_sentence_line(line)
        ):
            flush_paragraph()
            blocks.append({"type": "heading", "text": "ABSTRACT"})
            saw_abstract = True

        paragraph_parts.append(line)
        i += 1

    flush_paragraph()
    return blocks


def _markdown_for_document(row: dict[str, str], page_texts: list[list[str]], image_counts: list[int]) -> str:
    fallback_title = Path(row["file_name"]).stem
    title = _select_title(_extract_title(page_texts, fallback_title), row, fallback_title)

    lines = [
        f"# {title}",
        "",
        f"- `status`: `{row['status']}`",
        f"- `theme_dir`: `{row['theme_dir']}`",
        f"- `source_pdf_relative_path`: `{row['relative_path']}`",
        f"- `validation_status`: `{row.get('validation_status', '')}`",
        f"- `issues`: `{row.get('issues', '').strip() or '(none)'}`",
        f"- `page_count`: `{len(page_texts)}`",
        "",
    ]

    total_pages = len(page_texts)
    authors_heading_written = False
    for page_index, page_lines in enumerate(page_texts, start=1):
        blocks = _page_blocks(
            page_lines,
            page_number=page_index,
            total_pages=total_pages,
            image_count=image_counts[page_index - 1] if page_index - 1 < len(image_counts) else 0,
        )
        if page_index == 1:
            abstract_idx = next(
                (
                    idx
                    for idx, block in enumerate(blocks)
                    if block["type"] == "heading" and "ABSTRACT" in block["text"].upper()
                ),
                None,
            )
            if abstract_idx is not None:
                blocks = blocks[abstract_idx:]

            has_major_heading = any(
                block["type"] == "heading"
                and ("ABSTRACT" in block["text"].upper() or re.match(r"^\d", block["text"]))
                for block in blocks
            )
            word_count = sum(len(block["text"].split()) for block in blocks if block["type"] == "paragraph")
            if not has_major_heading and word_count < 120:
                blocks = []
        if page_index <= 2:
            title_sig = re.sub(r"[^a-z0-9]", "", title.lower())
            filtered_blocks: list[dict[str, str]] = []
            for block in blocks:
                block_sig = re.sub(r"[^a-z0-9]", "", block["text"].lower())
                if block_sig and (block_sig in title_sig or title_sig in block_sig):
                    continue
                filtered_blocks.append(block)
            blocks = filtered_blocks
        if not blocks:
            continue
        lines.extend([f"<!-- source-page: {page_index} -->", ""])
        for block in blocks:
            if block["type"] == "heading":
                lines.extend([f"## {block['text']}", ""])
            elif block["type"] == "author":
                if not authors_heading_written:
                    lines.extend(["## Authors", ""])
                    authors_heading_written = True
                lines.extend([f"- {block['text']}", ""])
            elif block["type"] == "caption":
                lines.extend([f"> {block['text']}", ""])
            elif block["type"] == "note":
                lines.extend([f"> {block['text']}", ""])
            elif block["type"] == "table":
                lines.extend(["```text", block["text"], "```", ""])
            else:
                lines.extend([block["text"], ""])

    lines = _cleanup_markdown_lines(lines)
    return "\n".join(lines).rstrip() + "\n"


def _augment_long_form_markdown(row: dict[str, str], markdown_text: str) -> str:
    spec = _long_form_chunk_spec(row)
    if spec is None:
        return markdown_text

    lines = markdown_text.splitlines()
    first_page_idx = next((idx for idx, line in enumerate(lines) if line.startswith("<!-- source-page:")), None)
    if first_page_idx is None:
        return markdown_text
    if any("ai_primary_entry_relative_path" in line for line in lines[:first_page_idx]):
        return markdown_text

    insertion = [
        f"- `ai_primary_entry_relative_path`: `{_primary_markdown_relative_path(row).as_posix()}`",
        f"- `ai_preferred_usage`: `{spec['preferred_usage']}`",
        "",
    ]
    lines[first_page_idx:first_page_idx] = insertion
    return "\n".join(lines).rstrip() + "\n"


def _split_markdown_pages(markdown_text: str) -> dict[int, list[str]]:
    pages: dict[int, list[str]] = {}
    current_page: int | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        match = re.match(r"<!-- source-page: (\d+) -->", line)
        if match:
            if current_page is not None:
                pages[current_page] = current_lines
            current_page = int(match.group(1))
            current_lines = [line]
            continue
        if current_page is not None:
            current_lines.append(line)

    if current_page is not None:
        pages[current_page] = current_lines
    return pages


def _write_long_form_chunks(row: dict[str, str], markdown_text: str, output_root: Path) -> None:
    spec = _long_form_chunk_spec(row)
    if spec is None:
        return

    page_map = _split_markdown_pages(markdown_text)
    if not page_map:
        return

    sections = [dict(section) for section in spec["sections"]]  # type: ignore[index]
    max_source_page = max(page_map)
    for idx, section in enumerate(sections):
        next_start = sections[idx + 1]["source_page_start"] if idx + 1 < len(sections) else max_source_page + 1
        section["source_page_end"] = int(next_start) - 1

    book_dir = output_root / row["theme_dir"] / str(spec["index_dirname"])
    book_dir.mkdir(parents=True, exist_ok=True)

    full_markdown_rel = Path("..") / _full_markdown_relative_path(row).name
    part_order: list[str] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for section in sections:
        part = str(section["part"])
        if part not in grouped:
            part_order.append(part)
        grouped[part].append(section)

    index_lines = [
        "# Silicon Photonics Design From Devices to Systems CN Translation",
        "",
        "This chapter index is the preferred AI entry for this book-length reference.",
        "",
        f"- `source_pdf_relative_path`: `{row['relative_path']}`",
        f"- `full_book_markdown_relative_path`: `{_full_markdown_relative_path(row).as_posix()}`",
        f"- `chunk_count`: `{len(sections)}`",
        f"- `ai_usage`: `{spec['preferred_usage']}`",
        f"- Full-book fallback: [full markdown]({quote(full_markdown_rel.as_posix(), safe='/')})",
        "",
    ]

    for part in part_order:
        index_lines.extend([f"## {part}", ""])
        for section in grouped[part]:
            chunk_name = f"{section['slug']}.md"
            link = quote(chunk_name, safe="/")
            source_range = f"{section['source_page_start']}-{section['source_page_end']}"
            line = f"- [{section['label']}]({link}) | source pages `{source_range}`"
            if section.get("book_page_start") is not None:
                line += f" | book pages `{section['book_page_start']}-{section['book_page_end']}`"
            index_lines.append(line)
        index_lines.append("")

    (book_dir / "INDEX.md").write_text("\n".join(index_lines).rstrip() + "\n", encoding="utf-8")

    for section in sections:
        chunk_lines = [
            f"# {section['label']}",
            "",
            f"- `book_part`: `{section['part']}`",
            f"- `source_pdf_relative_path`: `{row['relative_path']}`",
            f"- `full_book_markdown_relative_path`: `{_full_markdown_relative_path(row).as_posix()}`",
            f"- `chapter_index_relative_path`: `{_primary_markdown_relative_path(row).as_posix()}`",
            f"- `source_page_range`: `{section['source_page_start']}-{section['source_page_end']}`",
        ]
        if section.get("book_page_start") is not None:
            chunk_lines.append(f"- `book_page_range`: `{section['book_page_start']}-{section['book_page_end']}`")
        chunk_lines.extend(
            [
                "- `ai_usage`: `Focused chapter slice. Prefer this file over the full-book Markdown when the task is local to this section.`",
                "",
            ]
        )

        for source_page in range(int(section["source_page_start"]), int(section["source_page_end"]) + 1):
            page_lines = page_map.get(source_page)
            if not page_lines:
                continue
            chunk_lines.extend(page_lines)
            chunk_lines.append("")

        chunk_path = book_dir / f"{section['slug']}.md"
        chunk_path.write_text("\n".join(chunk_lines).rstrip() + "\n", encoding="utf-8")


def _build_index(rows: list[dict[str, str]], output_dir_name: str) -> str:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["status"]][row["theme_dir"]].append(row)

    lines = [
        "# Local Knowledge Base Markdown Index",
        "",
        "This index lists the AI-optimized Markdown mirrors for the local knowledge base.",
        "",
    ]
    for status in sorted(grouped):
        lines.extend([f"## {status}", ""])
        for theme in sorted(grouped[status]):
            lines.extend([f"### {theme}", ""])
            for row in sorted(grouped[status][theme], key=lambda item: item["file_name"]):
                md_rel = _primary_markdown_relative_path(row)
                md_link = quote(md_rel.as_posix(), safe="/")
                issue_text = row.get("issues", "").strip()
                suffix_parts: list[str] = []
                if _long_form_chunk_spec(row) is not None:
                    full_md_link = quote(_full_markdown_relative_path(row).as_posix(), safe="/")
                    suffix_parts.append("preferred: `chapter index`")
                    suffix_parts.append(f"full book: [md]({full_md_link})")
                if issue_text:
                    suffix_parts.append(f"issues: `{issue_text}`")
                suffix = f" | {' | '.join(suffix_parts)}" if suffix_parts else ""
                lines.append(f"- [{Path(row['file_name']).stem}]({md_link}){suffix}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_markdown(
    *,
    root: Path,
    inventory_name: str,
    output_dir_name: str,
    overwrite: bool,
) -> dict[str, int | str]:
    root_abs = root if root.is_absolute() else (REPO_ROOT / root)
    root_abs = root_abs.resolve()
    inventory_path = root_abs / inventory_name
    output_root = root_abs / output_dir_name
    rows = _read_csv(inventory_path)

    generated = 0
    skipped = 0
    output_root.mkdir(parents=True, exist_ok=True)

    legacy_manifest = output_root / "markdown_manifest.csv"
    if legacy_manifest.exists():
        legacy_manifest.unlink()

    for row in rows:
        markdown_path = output_root / Path(row["relative_path"]).with_suffix(".md")
        existing_text = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None

        if existing_text is not None and not overwrite:
            markdown_text = existing_text
            skipped += 1
        else:
            pdf_path = root_abs / row["relative_path"]
            page_texts = _extract_page_texts(pdf_path)
            image_counts = _extract_image_counts(pdf_path)
            markdown_text = _markdown_for_document(row, page_texts, image_counts)
            generated += 1

        markdown_text = _augment_long_form_markdown(row, markdown_text)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        if markdown_text != existing_text:
            markdown_path.write_text(markdown_text, encoding="utf-8")

        _write_long_form_chunks(row, markdown_text, output_root)

    readme_lines = [
        "# Local Knowledge Base Markdown Mirror",
        "",
        "This directory stores AI-optimized Markdown mirrors of the governed literature PDFs under `original_papers/`.",
        "",
        f"- Source inventory: `{inventory_name}`",
        f"- Mirrored documents: `{len(rows)}`",
        "- Output style: structure-preserving Markdown with page anchors, figure/table notes, and cleaned reading text",
        "- Long-form references may expose nested `__chapters/INDEX.md` entries; prefer those over full-book Markdown when available",
        "- Generation tool: `python3 experiments/tools/export_local_knowledge_base_markdown.py`",
        "",
        "Use `INDEX.md` to navigate the mirror.",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    (output_root / "INDEX.md").write_text(_build_index(rows, output_dir_name), encoding="utf-8")

    md_count = sum(1 for _ in output_root.rglob("*.md"))
    return {
        "source_rows": len(rows),
        "generated": generated,
        "skipped": skipped,
        "missing_markdown": max(len(rows) + 2 - md_count, 0),
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "index_path": str((output_root / "INDEX.md").relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export original_papers knowledge-base PDFs into AI-optimized Markdown."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inventory-name", default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = export_markdown(
        root=args.root,
        inventory_name=args.inventory_name,
        output_dir_name=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        "[knowledge-base-markdown] "
        f"source_rows={summary['source_rows']} "
        f"generated={summary['generated']} "
        f"skipped={summary['skipped']} "
        f"missing_markdown={summary['missing_markdown']} "
        f"output_root={summary['output_root']} "
        f"index_path={summary['index_path']}"
    )


if __name__ == "__main__":
    main()
