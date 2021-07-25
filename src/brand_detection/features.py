import string

from typing import Any
from typing import Mapping


def extract_features(sent: Mapping[str, Any], i: int) -> Mapping[str, Any]:
    labeled_tokens = sent["labeled_tokens"]
    word = str(labeled_tokens[i][0])
    features = {
        "root_cat": sent["root_category"],
        "bias": 1,
        "word_position": i,
        "word.lower()": word.lower(),
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }

    if i > 0:
        word1 = str(labeled_tokens[i - 1][0])
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(labeled_tokens) - 1:
        word1 = str(labeled_tokens[i + 1][0])
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:word.anydigit": any(ch.isdigit() for ch in word1),
                "+1:word.ispunctuation": word1 in string.punctuation,
            }
        )
    else:
        features["EOS"] = True

    if i > 1:
        word1 = str(labeled_tokens[i - 1][0])
        word2 = str(labeled_tokens[i - 2][0])
        features.update({"-2:ngram": "{} {}".format(word2, word1)})

    if i < len(labeled_tokens) - 2:
        word1 = str(labeled_tokens[i + 1][0])
        word2 = str(labeled_tokens[i + 2][0])
        features.update({"+2:ngram": "{} {}".format(word1, word2)})

    return features
