import re

from typing import Iterable
from typing import List
from typing import Tuple
from typing import cast

import nltk

import brand_detection.util


LABEL_OTHER = "0"
LABEL_BRAND = "BRAND"


def tokenize(string: str) -> List[str]:
    return cast(List[str], nltk.tokenize.word_tokenize(string))


def label_tokens(label: str, tokens: Iterable[str]) -> List[Tuple[str, str]]:
    return [(token, label) for token in tokens]


def label_brand(brand: str, string: str) -> List[Tuple[str, str]]:
    start, end = brand_detection.util.find_pattern_start_end(
        brand, string, re.UNICODE | re.IGNORECASE
    )
    if start is None or end is None:
        return label_tokens(LABEL_OTHER, tokenize(string))
    return (
        label_tokens(LABEL_OTHER, tokenize(string[:start]))
        + label_tokens(LABEL_BRAND, tokenize(string[start:end]))
        + label_tokens(LABEL_OTHER, tokenize(string[end:]))
    )
