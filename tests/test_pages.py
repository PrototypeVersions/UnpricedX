import json
from urllib.parse import urlsplit

import pytest

from ops import pages


@pytest.fixture
def site(tmp_path):
    for name in pages.REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if name.endswith('.json') else 'test site')
    pages.prepare(tmp_path, "abc123")
    return tmp_path


def public_copy(site, overrides=None):
    overrides = overrides or {}

    def fetcher(url):
        name = urlsplit(url).path.removeprefix('/UnpricedX/')
        return overrides.get(name, (site / name).read_bytes())

    return fetcher


def test_verify_checks_public_content_not_just_manifest(site):
    url = 'https://example.org/UnpricedX/'
    assert pages.verify(site, url, public_copy(site))["sourceCommit"] == "abc123"
    with pytest.raises(ValueError, match='data/forecasts.json'):
        pages.verify(site, url, public_copy(site, {'data/forecasts.json': b'{"stale":true}'}))


def test_verify_rejects_stale_manifest(site):
    old = json.loads((site / pages.MANIFEST).read_text())
    old['sourceCommit'] = 'older'
    with pytest.raises(ValueError, match='manifest is stale'):
        pages.verify(site, 'https://example.org/UnpricedX/',
                     public_copy(site, {pages.MANIFEST: json.dumps(old).encode()}))


def test_prepare_rejects_missing_export_and_nonfinite_json(site):
    path = site / 'data/export/latest.json'
    path.unlink()
    with pytest.raises(ValueError, match='required site file missing'):
        pages.prepare(site, 'abc123')
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError):
        pages.prepare(site, 'abc123')
