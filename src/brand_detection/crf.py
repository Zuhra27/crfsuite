import itertools
import tempfile

from typing import IO
from typing import Any
from typing import Iterable
from typing import List
from typing import Mapping
from typing import cast

import numpy as np
import pandas as pd
import pycrfsuite
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

from brand_detection.text import LABEL_BRAND
from brand_detection.text import LABEL_OTHER


def create_trainer(
    params: Mapping[str, Any], algorithm: str = "lbfgs", verbose: bool = False
) -> pycrfsuite.Trainer:
    return pycrfsuite.Trainer(algorithm=algorithm, params=params, verbose=verbose)


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> List[pd.Series]:
    return cast(
        List[pd.Series],
        sklearn.model_selection.train_test_split(
            df["features"], df["labels"], test_size=test_size, random_state=random_state
        ),
    )


def fit(
    trainer: pycrfsuite.Trainer, x: Iterable[Mapping[str, Any]], y: Iterable[str]
) -> IO[bytes]:
    train_data = zip(x, y)
    for xseq, yseq in train_data:
        trainer.append(xseq, yseq)
    file = tempfile.NamedTemporaryFile(prefix="model_", suffix=".crfsuite", delete=True)
    trainer.train(file.name)
    return file


def create_tagger(filepath: str) -> pycrfsuite.Tagger:
    tagger = pycrfsuite.Tagger()
    tagger.open(filepath)
    return tagger


def predict(
    tagger: pycrfsuite.Tagger, x: Iterable[Iterable[Mapping[str, Any]]]
) -> List[str]:
    titles = [[features["word.lower()"] for features in item] for item in x]
    predictions = list(map(tagger.tag, x))
    masks = [[tag == LABEL_BRAND for tag in item] for item in predictions]
    return [" ".join(np.array(titles[i])[masks[i]]) for i in range(len(titles))]


def get_classification_report(
    tagger: pycrfsuite.Tagger,
    x: Iterable[Iterable[Mapping[str, Any]]],
    y: Iterable[Iterable[str]],
) -> str:
    lb = sklearn.preprocessing.LabelBinarizer()
    y_pred = list(map(tagger.tag, x))
    y_true_combined = lb.fit_transform(list(itertools.chain.from_iterable(y)))
    y_pred_combined = lb.transform(list(itertools.chain.from_iterable(y_pred)))

    tags, labels = zip(
        *[(cls, idx) for idx, cls in enumerate(tagger.labels()) if cls != LABEL_OTHER]
    )

    report: str = sklearn.metrics.classification_report(
        y_true_combined, y_pred_combined, labels=labels, target_names=tags, digits=3
    )
    return report
