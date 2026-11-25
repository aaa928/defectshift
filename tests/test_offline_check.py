"""Checks of the network block, the self-test, the exit status and the synthetic table of scripts/checks/check_offline_load.py. No model is loaded and no connection
leaves the machine. Run: python -m pytest tests/test_offline_check.py -q
"""
import importlib.util
import os
import socket

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def chk(monkeypatch):
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):          # the script sets both at import; they are restored after the test
        monkeypatch.setenv(k, "0")
    spec = importlib.util.spec_from_file_location("check_offline_load", os.path.join(ROOT, "scripts", "checks", "check_offline_load.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_external_paths_are_blocked_and_recorded(chk):
    orig = [getattr(socket, n) for n in ("getaddrinfo", "gethostbyname", "gethostbyname_ex")] + [getattr(socket.socket, n) for n in ("connect", "connect_ex", "sendto")]
    with chk.NetworkGuard() as g:
        with pytest.raises(OSError, match="blocked"):
            socket.create_connection(("example.invalid", 443), timeout=1)                # name resolution by getaddrinfo
        with pytest.raises(OSError, match="blocked"):
            socket.gethostbyname("example.invalid")
        with pytest.raises(OSError, match="blocked"):
            socket.gethostbyname_ex("example.invalid")
        with socket.socket() as s:
            with pytest.raises(OSError, match="blocked"):
                s.connect(("192.0.2.1", 80))                                              # an address that is not loopback
            assert s.connect_ex(("192.0.2.1", 80)) == 111
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
            with pytest.raises(OSError, match="blocked"):
                u.sendto(b"x", ("192.0.2.1", 9))
    assert [a["host"] for a in g.attempts] == ["example.invalid"] * 3 + ["192.0.2.1"] * 3
    assert g.attempts[3]["port"] == 80
    assert orig == [getattr(socket, n) for n in ("getaddrinfo", "gethostbyname", "gethostbyname_ex")] + [getattr(socket.socket, n) for n in ("connect", "connect_ex", "sendto")]


def test_the_caller_is_recorded_without_a_site_packages_path(chk):
    """Code outside site-packages (an editable install, this test) is named by its last two path components."""
    with chk.NetworkGuard() as g:
        with pytest.raises(OSError):
            socket.gethostbyname("example.invalid")
    assert g.attempts[0]["caller"] == "tests/test_offline_check.py"


def test_loopback_stays_open(chk):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        with chk.NetworkGuard() as g:
            cli = socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2)
            cli.close()
        assert g.attempts == []
    finally:
        srv.close()


def test_selftest_covers_the_three_kinds_of_paths(chk):
    assert chk.selftest() == dict(name_resolution=True, tcp_connect=True, udp_sendto=True, attempts_recorded=True)


def test_exit_status_reflects_failures_and_network_attempts(chk):
    ok = dict(status="ok", network_attempts=[])
    attempt = dict(status="ok", network_attempts=[dict(host="h", port=443, caller="tabpfn")])
    assert chk.overall([ok, ok]) == ("ok", 0)
    assert chk.overall([ok, attempt]) == ("network_attempts", 2)                          # every model worked, but a library tried to reach the network
    assert chk.overall([ok, dict(status="failed", network_attempts=[])]) == ("models_failed", 1)
    assert chk.overall([attempt, dict(status="failed", network_attempts=[])]) == ("models_failed", 1)


def test_synthetic_table_has_the_benchmark_layout(chk):
    Xtr, ytr, Xte, cat_idx = chk.synthetic()
    assert cat_idx == [0, 1] and Xtr.shape == (200, 8) and Xte.shape == (40, 8) and ytr.shape == (200,)
    assert set(np.unique(Xtr[:, :2])) <= {0.0, 1.0, 2.0, 3.0}                      # integer ids of the categorical columns
    assert np.isnan(Xtr[:, 2:]).any() and not np.isnan(Xtr[:, :2]).any()
    assert np.array_equal(chk.synthetic()[0], Xtr, equal_nan=True)                   # deterministic
