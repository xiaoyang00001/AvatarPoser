import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import websocket


REPO_ROOT = Path(__file__).resolve().parents[1]
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

DATASETS = {
    "BMLrub": (
        "https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/BMLrub.tar.bz2",
        "licenses2024/BMLrub/bmlrub_license_updated.html",
    ),
    "CMU": (
        "https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/CMU.tar.bz2",
        "licenses2024/CMU/cmu_license_updated.html",
    ),
    "HDM05": (
        "https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/HDM05.tar.bz2",
        "licenses2024/HDM05/hdm05_license_updated.html",
    ),
}


class CdpPage:
    def __init__(self, websocket_url):
        self.ws = websocket.create_connection(websocket_url, timeout=30)
        self.next_id = 1
        self.events = []

    def close(self):
        self.ws.close()

    def command(self, method, params=None, timeout=60):
        params = params or {}
        command_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({"id": command_id, "method": method, "params": params}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = json.loads(self.ws.recv())
            if message.get("id") == command_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})
            if "method" in message:
                self.events.append(message)
        raise TimeoutError(method)

    def eval(self, expression, timeout=60):
        result = self.command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
            timeout=timeout,
        )
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    def navigate(self, url):
        self.command("Page.navigate", {"url": url})
        self.wait_for("document.body !== null", timeout=60)

    def wait_for(self, expression, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.eval(expression, timeout=10):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(expression)


def read_json_url(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def find_chrome():
    for path in CHROME_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Chrome or Edge was not found in standard install paths.")


def ensure_browser(port, profile_dir, headed=False):
    try:
        read_json_url(f"http://127.0.0.1:{port}/json/version")
        return None
    except Exception:
        pass

    chrome = find_chrome()
    profile_dir.mkdir(parents=True, exist_ok=True)
    args = [
        chrome,
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--disable-blink-features=AutomationControlled",
        f"--user-agent={USER_AGENT}",
        "--no-first-run",
        "about:blank",
    ]
    if headed:
        args.insert(-1, "--start-minimized")
    else:
        args[1:1] = ["--headless=new", "--disable-gpu", "--disable-extensions"]

    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            read_json_url(f"http://127.0.0.1:{port}/json/version")
            return process
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Chrome remote debugging endpoint did not become available.")


def get_page(port):
    tabs = read_json_url(f"http://127.0.0.1:{port}/json")
    for tab in tabs:
        if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
            return CdpPage(tab["webSocketDebuggerUrl"])
    raise RuntimeError("No debuggable Chrome page found.")


def login(page, username, password):
    page.navigate("https://amass.is.tue.mpg.de/login.php")
    existing_state = page.eval(
        "({hasUsername: !!document.querySelector('#username'), hasLogout: !!document.querySelector('a[href=\"logout.php\"]'), url: location.href})"
    )
    if not existing_state.get("hasUsername"):
        if existing_state.get("hasLogout") or "index.html" in existing_state.get("url", ""):
            return
        raise RuntimeError(f"AMASS login form was not found: {existing_state}")
    page.eval(
        """
        (() => {
          document.querySelector('#username').value = arguments[0];
          document.querySelector('#password').value = arguments[1];
          document.querySelector('form').submit();
          return true;
        })()
        """.replace("arguments[0]", json.dumps(username)).replace("arguments[1]", json.dumps(password))
    )
    page.wait_for("document.readyState === 'interactive' || document.readyState === 'complete'", timeout=60)
    time.sleep(2)
    state = page.eval(
        "({url: location.href, hasLogout: !!document.querySelector('a[href=\"logout.php\"]'), hasPassword: !!document.querySelector('#password'), text: document.body.innerText.slice(0, 300)})"
    )
    if not state.get("hasLogout") and state.get("hasPassword"):
        raise RuntimeError(f"AMASS login did not succeed: {state}")


def trigger_download(page, name, download_url, license_name):
    print(f"{name}: opening AMASS download page", flush=True)
    page.navigate("https://amass.is.tue.mpg.de/download.php")
    page.wait_for("typeof openModalLicense === 'function'", timeout=60)
    page.command("Network.setUserAgentOverride", {"userAgent": USER_AGENT})
    page.command("Emulation.setUserAgentOverride", {"userAgent": USER_AGENT})
    current_ua = page.eval("navigator.userAgent")
    print(f"{name}: browser UA before download: {current_ua}", flush=True)
    page.eval(
        f"openModalLicense({json.dumps(download_url)}, {json.dumps(license_name)}); true",
        timeout=60,
    )
    time.sleep(3)
    modal_state = page.eval(
        "({modal: !!document.querySelector('#licensemodal.show'), accept: !!document.querySelector('#accept_license')})"
    )
    if modal_state.get("accept"):
        print(f"{name}: accepting license", flush=True)
        page.eval("document.querySelector('#accept_license').click(); true", timeout=30)
    else:
        print(f"{name}: license already accepted or direct download started", flush=True)


def wait_for_download(page, output_dir, before_files, timeout=3600):
    deadline = time.time() + timeout
    last_report = 0
    while time.time() < deadline:
        current = set(output_dir.glob("*"))
        new_files = [path for path in current - before_files if path.is_file()]
        partials = [path for path in output_dir.glob("*.crdownload")]
        download_events = [event for event in page.events if event.get("method", "").startswith("Browser.download")]
        now = time.time()
        if now - last_report >= 30:
            for partial in partials:
                print(f"downloading: {partial.name} {partial.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
            if download_events:
                print(f"download events: {download_events[-1]}", flush=True)
            last_report = now
        finished = [path for path in new_files if path.suffix != ".crdownload" and not Path(str(path) + ".crdownload").exists()]
        if finished and not partials:
            return finished
        time.sleep(2)
    raise TimeoutError("Timed out waiting for Chrome download to finish.")


def main():
    parser = argparse.ArgumentParser(description="Download AMASS archives through the official browser flow.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--download-dir", default=str(REPO_ROOT / "amass" / "_archives"))
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument("--timeout", type=int, default=24 * 3600)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    username = os.environ.get("AMASS_USERNAME")
    password = os.environ.get("AMASS_PASSWORD")
    if not username or not password:
        raise SystemExit("AMASS_USERNAME and AMASS_PASSWORD must be set.")

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = REPO_ROOT / f".chrome-amass-{args.port}"
    process = ensure_browser(args.port, profile_dir, headed=args.headed)

    page = get_page(args.port)
    try:
        page.command("Page.enable")
        page.command("Runtime.enable")
        page.command("Network.enable")
        page.command(
            "Network.setUserAgentOverride",
            {"userAgent": USER_AGENT},
        )
        page.command(
            "Emulation.setUserAgentOverride",
            {"userAgent": USER_AGENT},
        )
        page.command(
            "Browser.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": str(download_dir),
                "eventsEnabled": True,
            },
        )
        login(page, username, password)
        for name in args.datasets:
            before = set(download_dir.glob("*"))
            download_url, license_name = DATASETS[name]
            trigger_download(page, name, download_url, license_name)
            finished = wait_for_download(page, download_dir, before, timeout=args.timeout)
            print(f"{name}: finished {', '.join(path.name for path in finished)}", flush=True)
    finally:
        page.close()
        if process is not None:
            process.terminate()


if __name__ == "__main__":
    main()
