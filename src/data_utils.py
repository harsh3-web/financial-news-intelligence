import pandas as pd
import re

def load_phrasebank(file_path):
    """Read a Financial PhraseBank file into a DataFrame with 'sentence' and 'label' columns."""
    rows = []
    bad_lines = []  # lines we couldn't parse, kept so we can inspect them instead of silently dropping them

    with open(file_path, encoding="latin-1") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()  # remove the newline and surrounding spaces
            if not line:
                continue
            if "@" not in line:  # no separator means we can't find the label
                bad_lines.append((line_number, line))
                continue
            sentence, label = line.rsplit("@", 1)  # split on the LAST "@" only
            rows.append({"sentence": sentence.strip(), "label": label.strip()})

    return pd.DataFrame(rows), bad_lines


def build_mojibake_map():
    """Rebuild the corruption rules: UTF-8 bytes read as cp850, then box-drawing lead characters flattened."""
    mapping = {}
    # Rule 1: letters À...ÿ start with byte C3, shown in cp850 as '├', stored as '+'
    for code in range(0xC0, 0x100):
        letter = chr(code)
        garbled = letter.encode("utf-8").decode("cp850")   # "ä" -> "├ñ"
        mapping["+" + garbled[1]] = letter                 # "+ñ" -> "ä"
    # Rule 2: symbols U+00A0...U+00BF start with byte C2, shown in cp850 as '┬', stored as '-'
    for code in range(0xA0, 0xC0):
        symbol = chr(code)
        garbled = symbol.encode("utf-8").decode("cp850")   # "©" -> "┬®"
        mapping["-" + garbled[1]] = symbol                 # "-®" -> "©"
    # Hand overrides: cleaner output than the raw symbol
    mapping["-á"] = " "     # non-breaking space -> normal space
    mapping["-¦ "] = "'"    # "Tekla -¦ s" -> "Tekla 's", matching the dataset's own "'s" style
    return mapping

MOJIBAKE_MAP = build_mojibake_map()


def repair_mojibake(text):
    """Replace every known garbled pair with the letter it came from."""
    for garbled, letter in MOJIBAKE_MAP.items():
        text = text.replace(garbled, letter)
    return text

def normalize_numbers(text):
    """Repair split decimals and unify currency/unit spelling so one concept = one token."""
    text = re.sub(r"(\d) \.(\d)", r"\1.\2", text)               # "10 .1"   -> "10.1"
    text = re.sub(r"\b(EUR|USD|SEK)(?=\d)", r"\1 ", text)       # "EUR10"   -> "EUR 10"
    text = re.sub(r"(\d)(mn|mln|m)\b", r"\1 million", text)     # "50m"     -> "50 million"
    text = re.sub(r"\b(mn|mln)\b", "million", text)             # "mn"      -> "million"
    text = re.sub(r"(\d) m\b", r"\1 million", text)             # "3.2 m"   -> "3.2 million"
    return text


def make_group_key(text):
    """turn a sentence into its 'tample' so look alike sentence get the same key"""
    text = text.lower()
    text = re.sub(r"\d+(?:[.,]\d+)*", "<num>", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text