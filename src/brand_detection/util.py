import re

from typing import Optional
from typing import Tuple


def find_pattern_start_end(
    pattern: str, string: str, flags: int = 0
) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(pattern, string, flags)
    return match.span() if match else (None, None)
