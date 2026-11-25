"""Offline load check of the pinned foundation-model weights. No benchmark data is needed: for each model a small synthetic regression table is fitted and
predicted with the network blocked, so that a library that still fetches a file after it has been handed the pinned weights (for example a configuration file from
the main branch of a Hub repository) is caught, not only a missing weight file.

Run it after scripts/00_fetch_weights.py (the pinned files must be in the local cache) and, for Kumo-Tabular, after its first online load. The check sets
HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 and, in addition, blocks the network paths of Python sockets in this process: name resolution (getaddrinfo, gethostbyname,
gethostbyname_ex), TCP connections (connect, connect_ex) and UDP sendto; loopback and local (unix) sockets stay open. A process that a library starts (for example git
through subprocess) is not covered. An attempt raises an OSError and is recorded with its host, port and the package or file that made it (a package of site-packages by
its name, any other code, such as an editable install, by its last two path components). A self-test at the start shows that the three kinds of paths are blocked.

Record: results/offline_check/<run-id>.json, per model status "ok" (loaded, fitted and predicted finite values) or "failed" (error string), the seconds, the version
string that a benchmark run records, and the network attempts that were made while the model was handled (none are expected; an attempt that a library swallows,
such as usage statistics, is listed although the model works). Overall "status": "ok", "models_failed" or "network_attempts" (all models worked, but attempts were made).
Exit code: 0 for "ok", 1 for "models_failed", 2 for "network_attempts"; the attempts are also written to the log as warnings.
The record can contain local absolute paths (error strings, tracebacks): do not put it into public material. Existing records are not overwritten.
Usage: python scripts/checks/check_offline_load.py --run-id offline_check_1 [--models TabPFN,TabICL,TabFM,TabDPT,EXAONE,Kumo]
"""
import argparse
import json
import os
import socket
import sys
import sysconfig
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402

LOOPBACK = ("127.0.0.1", "::1", "localhost", "0.0.0.0", "")
GENERIC = ("socket", "ssl", "urllib3", "http", "requests", "httpx", "httpcore", "anyio", "asyncio", "threading", "concurrent", "huggingface_hub", "posthog", "backoff")
THIS_FILE = os.path.realpath(__file__)
STDLIB = os.path.realpath(sysconfig.get_paths()["stdlib"])


def _caller():
    """Who made the attempt: the package of site-packages (by name) or, for any other code outside the standard library and this file, its last two path components."""
    for fr in reversed(traceback.extract_stack()[:-3]):
        fn = os.path.realpath(fr.filename)
        if fn == THIS_FILE or fn.startswith(STDLIB + os.sep) and "site-packages" not in fn:
            continue
        parts = fn.replace("\\", "/").split("/site-packages/")
        if len(parts) == 2:
            pkg = parts[1].split("/")[0].split(".")[0]
            if pkg in GENERIC:
                continue
            return pkg
        return "/".join(fn.replace("\\", "/").split("/")[-2:])
    return None


def _is_external(address):
    return isinstance(address, tuple) and bool(address) and address[0] not in LOOPBACK


class NetworkGuard:
    """Context manager that blocks the network paths of Python sockets: name resolution and connections to anything but loopback and local (unix) sockets raise OSError
    and are recorded (host, port, caller)."""

    def __init__(self):
        self.attempts = []

    def _record(self, host, port):
        self.attempts.append(dict(host=str(host), port=port if isinstance(port, int) else None, caller=_caller()))

    def __enter__(self):
        s = socket.socket
        self._orig = dict(getaddrinfo=socket.getaddrinfo, gethostbyname=socket.gethostbyname, gethostbyname_ex=socket.gethostbyname_ex,
                          connect=s.connect, connect_ex=s.connect_ex, sendto=s.sendto)
        orig, guard = self._orig, self

        def resolve(fn):
            def wrapped(host, *a, **k):
                if host not in LOOPBACK:
                    guard._record(host, a[0] if a and isinstance(a[0], int) else None)
                    raise OSError(f"network blocked by check_offline_load (name resolution of {host})")
                return orig[fn](host, *a, **k)
            return wrapped

        def connect(sock, address):
            if _is_external(address):
                guard._record(address[0], address[1] if len(address) > 1 else None)
                raise OSError(f"network blocked by check_offline_load (connection to {address[0]})")
            return orig["connect"](sock, address)

        def connect_ex(sock, address):
            if _is_external(address):
                guard._record(address[0], address[1] if len(address) > 1 else None)
                return 111
            return orig["connect_ex"](sock, address)

        def sendto(sock, data, *args):
            address = args[-1] if args else None
            if _is_external(address):
                guard._record(address[0], address[1] if len(address) > 1 else None)
                raise OSError(f"network blocked by check_offline_load (datagram to {address[0]})")
            return orig["sendto"](sock, data, *args)

        socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex = resolve("getaddrinfo"), resolve("gethostbyname"), resolve("gethostbyname_ex")
        s.connect, s.connect_ex, s.sendto = connect, connect_ex, sendto
        return self

    def __exit__(self, *exc):
        o, s = self._orig, socket.socket
        socket.getaddrinfo, socket.gethostbyname, socket.gethostbyname_ex = o["getaddrinfo"], o["gethostbyname"], o["gethostbyname_ex"]
        s.connect, s.connect_ex, s.sendto = o["connect"], o["connect_ex"], o["sendto"]
        return False


def _blocked(fn):
    """True if fn raises OSError (the guard's block), False if it succeeds."""
    sock = None
    try:
        sock = fn()
        return False
    except OSError:
        return True
    finally:
        if hasattr(sock, "close"):
            sock.close()


def selftest():
    """The three kinds of paths must be blocked: name resolution, a TCP connection to an address (TEST-NET-1, no name to resolve) and a UDP datagram."""
    def udp():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.sendto(b"x", ("192.0.2.1", 9))
        finally:
            s.close()

    def tcp():
        s = socket.socket()
        try:
            s.connect(("192.0.2.1", 443))
        finally:
            s.close()
    with NetworkGuard() as g:
        res = dict(name_resolution=_blocked(lambda: socket.gethostbyname("huggingface.co")), tcp_connect=_blocked(tcp), udp_sendto=_blocked(udp))
    res["attempts_recorded"] = len(g.attempts) == 3
    return res


def synthetic(seed=0, n_train=200, n_test=40):
    """Small table in the layout of the benchmark wrappers: two categorical columns (integer ids) in front of six numerical columns, a few missing values."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    cat = rng.integers(0, 4, size=(n, 2)).astype("float32")
    cont = rng.normal(size=(n, 6)).astype("float32")
    y = cont[:, 0] - 0.5 * cont[:, 1] + 0.3 * cat[:, 0] + 0.1 * rng.normal(size=n)
    cont[rng.random(cont.shape) < 0.05] = np.nan
    X = np.concatenate([cat, cont], axis=1)
    return X[:n_train], y[:n_train], X[n_train:], [0, 1]


def check_one(name):
    from defectshift.models import foundation as fm
    Xtr, ytr, Xte, cat_idx = synthetic()
    t0 = time.time()
    rec = dict(model=name, status="failed", error=None, seconds=None, version=None)
    with NetworkGuard() as g:
        try:
            rec["version"] = fm.version(name)
            est = fm.build(name, cat_idx, 0)
            est.fit(Xtr, ytr)
            pred = np.asarray(est.predict(Xte), float).reshape(-1)
            if pred.shape != (len(Xte),) or not np.isfinite(pred).all():
                raise RuntimeError(f"prediction has shape {pred.shape} or non-finite values")
            rec["status"] = "ok"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-4:]
        finally:
            fm.release(name)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["network_attempts"] = g.attempts
    return rec


def overall(results):
    """("ok" | "models_failed" | "network_attempts", exit code) from the per-model records."""
    if any(r["status"] != "ok" for r in results):
        return "models_failed", 1
    if any(r["network_attempts"] for r in results):
        return "network_attempts", 2
    return "ok", 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--models", default="TabPFN,TabICL,TabFM,TabDPT,EXAONE,Kumo")
    a = ap.parse_args()
    from defectshift import config
    from defectshift.models import foundation as fm
    from defectshift.utils import free_memory, log
    names = a.models.split(",")
    unknown = sorted(set(names) - set(fm.FOUNDATION_MODELS))
    if unknown:
        raise SystemExit(f"unknown models {unknown}; the foundation models are {list(fm.FOUNDATION_MODELS)}")
    out_dir = os.path.join(config.RESULTS_DIR, "offline_check")
    dst = os.path.join(out_dir, f"{a.run_id}.json")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists (existing records are not overwritten; use another --run-id)")
    st = selftest()
    if not all(st.values()):
        raise SystemExit(f"the network block is not active ({st}); the check would prove nothing")
    results = []
    for name in names:
        log(f"[offline_check] {name} ...")
        r = check_one(name)
        free_memory()
        log(f"[offline_check] {name}: {r['status']} in {r['seconds']} s; network attempts: {len(r['network_attempts'])}" + (f"; {r['error']}" if r["error"] else ""))
        for t in r["network_attempts"]:
            log(f"[offline_check] WARNING {name} tried to reach {t['host']}:{t['port']} (caller: {t['caller']})", level="WARNING")
        results.append(r)
    status, code = overall(results)
    rec = dict(run_id=a.run_id, env=dict(HF_HUB_OFFLINE=os.environ["HF_HUB_OFFLINE"], TRANSFORMERS_OFFLINE=os.environ["TRANSFORMERS_OFFLINE"]), guard_selftest=st,
               models=results, status=status, all_models_ok=all(r["status"] == "ok" for r in results), no_network_attempt=all(not r["network_attempts"] for r in results))
    os.makedirs(out_dir, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(rec, f, indent=1)
    log(f"[offline_check] written: {dst}; status {status}", level="INFO" if code == 0 else "WARNING")
    sys.exit(code)


if __name__ == "__main__":
    main()
