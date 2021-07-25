import pandas as pd

import brand_detection.features
import brand_detection.text


def assign_feature_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["labeled_tokens"] = result.apply(
        lambda row: brand_detection.text.label_brand(row["brand"], row["title"]), axis=1
    )
    result["features"] = result.apply(
        lambda row: [
            brand_detection.features.extract_features(row.to_dict(), i)
            for i in range(len(row["labeled_tokens"]))
        ],
        axis=1,
    )
    result["labels"] = result.apply(
        lambda row: [label for _, label in row["labeled_tokens"]], axis=1
    )
    return result
