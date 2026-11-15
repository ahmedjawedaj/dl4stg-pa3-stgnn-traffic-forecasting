import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "sd_2020_2021.npz")
RAW = os.environ.get("PA3_RAW_DIR", os.path.join(ROOT, "data", "raw"))


@pytest.fixture(scope="session")
def prep():
    if not os.path.exists(DATA):
        pytest.skip("processed data missing, run scripts/prepare_data.py")
    from pa3.data import Prepared
    return Prepared(DATA)


def need_raw():
    if not os.path.exists(os.path.join(RAW, "2021.h5")):
        pytest.skip("raw LargeST files not available (set PA3_RAW_DIR)")
