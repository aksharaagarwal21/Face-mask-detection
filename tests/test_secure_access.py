import os

import pytest
from flask import Flask, jsonify

from secure_access import require_access_key, ensure_certificate, COOKIE_NAME, new_access_key

REMOTE = {"REMOTE_ADDR": "192.168.1.50"}       # a phone on the same Wi-Fi


@pytest.fixture
def locked():
    app = Flask(__name__)

    @app.route("/")
    def index():
        return "dashboard"

    @app.route("/api/stats")
    def stats():
        return jsonify(ok=True)

    require_access_key(app, "s3cret-key")
    return app.test_client()


def test_other_devices_need_the_key(locked):
    r = locked.get("/", environ_base=REMOTE)
    assert r.status_code == 401 and b"Access key required" in r.data
    r = locked.get("/api/stats", environ_base=REMOTE)
    assert r.status_code == 401 and r.get_json()["error"] == "access key required"
    assert locked.get("/?key=wrong", environ_base=REMOTE).status_code == 401


def test_key_in_url_sets_a_cookie_for_later_requests(locked):
    r = locked.get("/?key=s3cret-key", environ_base=REMOTE)
    assert r.status_code == 200
    cookie = r.headers["Set-Cookie"]
    assert COOKIE_NAME in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    # the test client keeps the cookie: no key needed any more
    assert locked.get("/api/stats", environ_base=REMOTE).status_code == 200


def test_key_in_header(locked):
    r = locked.get("/api/stats", environ_base=REMOTE, headers={"X-Access-Key": "s3cret-key"})
    assert r.status_code == 200
    assert "Set-Cookie" not in r.headers


def test_this_computer_needs_no_key(locked):
    assert locked.get("/", environ_base={"REMOTE_ADDR": "127.0.0.1"}).status_code == 200


def test_no_key_means_open():
    app = Flask(__name__)
    app.add_url_rule("/", "i", lambda: "ok")
    require_access_key(app, None)
    assert app.test_client().get("/", environ_base=REMOTE).status_code == 200


def test_generated_keys_are_unique_and_url_safe():
    keys = {new_access_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(len(k) == 12 and k.replace("-", "").replace("_", "").isalnum() for k in keys)


def test_certificate_is_created_reused_and_extended(tmp_path):
    x509 = pytest.importorskip("cryptography.x509")
    cert, key = ensure_certificate(str(tmp_path), ["192.168.1.3"])
    first = open(cert, "rb").read()
    assert os.path.exists(key)
    assert ensure_certificate(str(tmp_path), ["192.168.1.3"]) == (cert, key)
    assert open(cert, "rb").read() == first                  # reused
    ensure_certificate(str(tmp_path), ["10.0.0.7"])          # new network -> new certificate
    c = x509.load_pem_x509_certificate(open(cert, "rb").read())
    ips = {str(i) for i in c.extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value.get_values_for_type(x509.IPAddress)}
    assert {"127.0.0.1", "10.0.0.7"} <= ips


def test_connect_endpoint_only_on_this_computer(monkeypatch):
    import app as appmod
    monkeypatch.setitem(appmod.app.config, "PHONE_URLS", ["https://192.168.1.3:5000/field?key=k"])
    c = appmod.app.test_client()
    r = c.get("/api/connect", environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert r.status_code == 200 and r.get_json()["phone_urls"][0].endswith("key=k")
    assert c.get("/api/connect", environ_base=REMOTE).status_code == 403
