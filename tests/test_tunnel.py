import pytest

import tunnel

LOG = """2026-10-02T07:01:12Z INF Requesting new quick Tunnel on trycloudflare.com...
2026-10-02T07:01:14Z INF +--------------------------------------------------------------------------------------------+
2026-10-02T07:01:14Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-10-02T07:01:14Z INF |  https://bright-otter-sample-words.trycloudflare.com                                       |
2026-10-02T07:01:14Z INF +--------------------------------------------------------------------------------------------+
"""


def test_finds_the_address_in_cloudflared_log():
    assert tunnel.tunnel_url(LOG) == "https://bright-otter-sample-words.trycloudflare.com"


def test_no_address_yet():
    assert tunnel.tunnel_url("2026-10-02T07:01:12Z INF Requesting new quick Tunnel on trycloudflare.com...") is None


def test_missing_cloudflared_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setattr(tunnel, "find_cloudflared", lambda: None)
    with pytest.raises(RuntimeError, match="winget install Cloudflare.cloudflared"):
        tunnel.start_quick_tunnel(5000, str(tmp_path / "cloudflared.log"))
