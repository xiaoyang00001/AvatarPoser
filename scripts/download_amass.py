import argparse
import os
import sys
import time
from pathlib import Path

import requests


DATASETS = {
    "BMLrub": (
        "amass_per_dataset/smplh/gender_specific/mosh_results/BMLrub.tar.bz2",
        "licenses2024/BMLrub/bmlrub_license_updated.html",
    ),
    "CMU": (
        "amass_per_dataset/smplh/gender_specific/mosh_results/CMU.tar.bz2",
        "licenses2024/CMU/cmu_license_updated.html",
    ),
    "HDM05": (
        "amass_per_dataset/smplh/gender_specific/mosh_results/HDM05.tar.bz2",
        "licenses2024/HDM05/hdm05_license_updated.html",
    ),
}


def download_url(sfile):
    return f"https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile={sfile}"


def login(session, username, password):
    print("Opening login/download page...", flush=True)
    response = session.get("https://amass.is.tue.mpg.de/login.php", timeout=(20, 40))
    response.raise_for_status()
    print("Submitting AMASS login form...", flush=True)
    response = session.post(
        "https://amass.is.tue.mpg.de/login.php",
        data={"username": username, "password": password, "commit": "Log in"},
        timeout=(20, 40),
        allow_redirects=True,
    )
    response.raise_for_status()
    print(f"Login response: status={response.status_code}, content-type={response.headers.get('content-type')}", flush=True)
    return response


def accept_license(session, sfile, license_name):
    from urllib.parse import quote

    filename = quote(sfile, safe="")
    license_path = quote(license_name, safe="")
    url = f"https://amass.is.tue.mpg.de/admin/ajax_setlicenseagreed.php?filename={filename}&licensename={license_path}"
    response = session.get(url, timeout=(20, 40))
    response.raise_for_status()
    if response.text.strip() != "1":
        raise RuntimeError(f"License acceptance failed for {sfile}: {response.text[:200]!r}")


def is_file_response(response):
    content_type = (response.headers.get("content-type") or "").lower()
    disposition = (response.headers.get("content-disposition") or "").lower()
    return "application" in content_type or "octet-stream" in content_type or ".tar.bz2" in disposition


def stream_download(session, name, sfile, license_name, output_dir, probe_bytes=None):
    accept_license(session, sfile, license_name)
    output_path = output_dir / f"{name}.tar.bz2"
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    print(f"{name}: requesting archive...", flush=True)
    with session.get(download_url(sfile), stream=True, headers=headers, timeout=(20, 40)) as response:
        response.raise_for_status()
        print(
            f"{name}: response status={response.status_code}, content-type={response.headers.get('content-type')}, "
            f"content-length={response.headers.get('content-length')}, content-disposition={response.headers.get('content-disposition')}",
            flush=True,
        )
        if not is_file_response(response):
            preview = response.text[:500] if "text" in (response.headers.get("content-type") or "") else ""
            raise RuntimeError(f"{name}: server did not return an archive. content-type={response.headers.get('content-type')!r} preview={preview!r}")

        mode = "ab" if existing_size and response.status_code == 206 else "wb"
        if mode == "wb":
            existing_size = 0

        total_header = response.headers.get("content-length")
        total = int(total_header) + existing_size if total_header and mode == "ab" else int(total_header or 0)
        downloaded = existing_size
        start = time.time()
        last_report = start

        with temp_path.open(mode + "b") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)

                if probe_bytes and downloaded >= probe_bytes:
                    break

                now = time.time()
                if now - last_report >= 30:
                    elapsed = now - start
                    speed = (downloaded - existing_size) / elapsed / 1024 / 1024 if elapsed else 0
                    percent = f" {downloaded / total * 100:.1f}%" if total else ""
                    print(f"{name}: {downloaded / 1024 / 1024:.1f} MB{percent}, {speed:.2f} MB/s", flush=True)
                    last_report = now

        if probe_bytes:
            elapsed = max(time.time() - start, 1e-9)
            speed = (downloaded - existing_size) / elapsed / 1024 / 1024
            print(f"{name} probe: {(downloaded - existing_size) / 1024 / 1024:.1f} MB in {elapsed:.2f}s, {speed:.2f} MB/s")
            return temp_path

    temp_path.replace(output_path)
    print(f"{name}: saved {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download AMASS SMPL+H gender-specific archives.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--output-dir", default="amass/_archives")
    parser.add_argument("--probe-mb", type=int, default=0)
    args = parser.parse_args()

    username = os.environ.get("AMASS_USERNAME")
    password = os.environ.get("AMASS_PASSWORD")
    if not username or not password:
        print("AMASS_USERNAME and AMASS_PASSWORD must be set.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        login(session, username, password)

        for name in args.datasets:
            sfile, license_name = DATASETS[name]
            stream_download(
                session,
                name,
                sfile,
                license_name,
                output_dir,
                probe_bytes=args.probe_mb * 1024 * 1024 if args.probe_mb else None,
            )


if __name__ == "__main__":
    raise SystemExit(main())
