import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = tempfile.mkdtemp(prefix="unpriced-test-")
os.environ["UNPRICED_ROOT"] = _TMP
os.environ["UNPRICED_STORE"] = os.path.join(_TMP, "store")
os.environ.pop("SEC_CONTACT_EMAIL", None)
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture(scope="session")
def market():
    from unpriced.data.synthetic import SyntheticMarket
    m = SyntheticMarket(n=120, start="2012-01-01", end="2020-12-31", seed=11)
    m.write_store()
    return m


@pytest.fixture(scope="session")
def ctx(market):
    from unpriced.features.context import load_context
    return load_context()
