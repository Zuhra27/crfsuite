import io
import pkgutil

from typing import cast

import pandas as pd


def load_brand_data() -> pd.DataFrame:
    brand_data = pkgutil.get_data("brand_detection", "data/brand_data.csv")
    assert brand_data is not None
    return cast(pd.DataFrame, pd.read_csv(io.BytesIO(brand_data), encoding="utf8"))
