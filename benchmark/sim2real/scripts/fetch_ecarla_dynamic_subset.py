#!/usr/bin/env python3
"""Range-extract selected files from the official eCARLA dynamic ZIP.

This avoids downloading the full 7.5 GB archive. It supports the ZIP64 central
directory used by the official Zenodo artifact and verifies each extracted
member's byte count and CRC32. The archive is never written locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
import zlib
from pathlib import Path

import requests

RECORD_URL = "https://zenodo.org/api/records/14412251/files/ecarla-scenes_dynamic.zip/content"
SCENE = "ecarla-scenes_dynamic/26_cars_town4_forward_cloudy-noon/"
MEMBER_FILES = {
    "props": SCENE + "props.json",
    "events": SCENE + "events.hdf5",
    "gray": SCENE + "gray.hdf5",
    "flow": SCENE + "flow.hdf5",
}
ARCHIVE_MD5 = "154c4562d14fd69a7b9c75883a7e9d79"
EXPECTED_ARCHIVE_SIZE = 7_534_813_423


def fetch_range(session: requests.Session, url: str, start: int, end: int, total: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 120))
    response.raise_for_status()
    expected_header = f"bytes {start}-{end}/{total}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected_header:
        raise RuntimeError(
            f"server did not honor byte range {start}-{end}: "
            f"status={response.status_code}, Content-Range={response.headers.get('Content-Range')!r}"
        )
    if len(response.content) != end - start + 1:
        raise RuntimeError(f"short range response: {len(response.content)} bytes")
    return response.content


def zip64_directory(session: requests.Session, url: str) -> tuple[int, int, list[dict]]:
    probe = fetch_range(session, url, 0, 0, EXPECTED_ARCHIVE_SIZE)
    if len(probe) != 1:
        raise RuntimeError("failed archive length probe")
    tail_size = min(2 * 1024 * 1024, EXPECTED_ARCHIVE_SIZE)
    tail_start = EXPECTED_ARCHIVE_SIZE - tail_size
    tail = fetch_range(session, url, tail_start, EXPECTED_ARCHIVE_SIZE - 1, EXPECTED_ARCHIVE_SIZE)
    locator_rel = tail.rfind(b"PK\x06\x07")
    if locator_rel < 0:
        raise RuntimeError("ZIP64 locator not found")
    _, disk, zip64_offset, disks = struct.unpack_from("<4sIQI", tail, locator_rel)
    if disk != 0 or disks != 1:
        raise RuntimeError("multi-disk ZIP archives are unsupported")
    end_record = fetch_range(session, url, zip64_offset, zip64_offset + 55, EXPECTED_ARCHIVE_SIZE)
    values = struct.unpack("<4sQHHIIQQQQ", end_record)
    signature, _, _, _, disk_no, directory_disk, _, entries, directory_size, directory_offset = values
    if signature != b"PK\x06\x06" or disk_no or directory_disk:
        raise RuntimeError("invalid ZIP64 end record")
    directory = fetch_range(
        session, url, directory_offset, directory_offset + directory_size - 1, EXPECTED_ARCHIVE_SIZE
    )
    rows: list[dict] = []
    pos = 0
    while pos < len(directory):
        if directory[pos : pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory record at offset {pos}")
        vals = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        _, _, _, flags, method, _, _, crc32, comp32, size32, name_len, extra_len, comment_len, _, _, _, local32 = vals
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8")
        extra = directory[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        row = {"name": name, "method": method, "crc32": crc32,
               "compressed_size": comp32, "size": size32, "local_offset": local32}
        p = 0
        while p < len(extra):
            tag, length = struct.unpack_from("<HH", extra, p)
            data = extra[p + 4 : p + 4 + length]
            if tag == 1:
                q = 0
                for key, value in (("size", size32), ("compressed_size", comp32), ("local_offset", local32)):
                    if value == 0xFFFFFFFF:
                        row[key] = struct.unpack_from("<Q", data, q)[0]
                        q += 8
            p += 4 + length
        rows.append(row)
        pos += 46 + name_len + extra_len + comment_len
    if len(rows) != entries:
        raise RuntimeError(f"entry count mismatch: parsed={len(rows)}, recorded={entries}")
    return directory_offset, directory_size, rows


def extract_member(session: requests.Session, url: str, total: int, row: dict, destination: Path) -> dict:
    local = fetch_range(session, url, row["local_offset"], row["local_offset"] + 29, total)
    sig, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3I2H", local)
    if sig != b"PK\x03\x04" or method != row["method"]:
        raise RuntimeError(f"local header mismatch for {row['name']}")
    data_start = row["local_offset"] + 30 + name_len + extra_len
    compressed = fetch_range(
        session, url, data_start, data_start + row["compressed_size"] - 1, total
    )
    if method == 8:
        content = zlib.decompress(compressed, -15)
    elif method == 0:
        content = compressed
    else:
        raise RuntimeError(f"unsupported ZIP method {method} for {row['name']}")
    crc = zlib.crc32(content) & 0xFFFFFFFF
    if len(content) != row["size"] or crc != row["crc32"]:
        raise RuntimeError(
            f"integrity mismatch for {row['name']}: size={len(content)}/{row['size']}, crc={crc:08x}/{row['crc32']:08x}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".partial")
    temp_path.write_bytes(content)
    os.replace(temp_path, destination)
    return {
        "archive_member": row["name"], "archive_member_size": row["size"],
        "archive_member_compressed_size": row["compressed_size"],
        "zip_crc32_verified": f"{crc:08x}", "sha256": hashlib.sha256(content).hexdigest(),
        "bytes_written": len(content), "path": str(destination.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proxy", help="optional HTTP proxy URL, e.g. http://127.0.0.1:7897")
    parser.add_argument("--members", nargs="+", choices=tuple(MEMBER_FILES),
                        default=["props", "events", "gray", "flow"])
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if args.proxy:
        session.trust_env = False
        session.proxies = {"https": args.proxy, "http": args.proxy}
    directory_offset, directory_size, rows = zip64_directory(session, RECORD_URL)
    requested = [MEMBER_FILES[name] for name in args.members]
    all_by_name = {row["name"]: row for row in rows}
    selected = {name: all_by_name[name] for name in requested if name in all_by_name}
    missing = sorted(set(requested) - set(selected))
    if missing:
        raise RuntimeError(f"requested members absent from official archive: {missing}")
    result = {
        "schema": "ecarla_range_subset_manifest_v1",
        "source_record": "https://zenodo.org/records/14412251",
        "source_file": "ecarla-scenes_dynamic.zip",
        "source_file_size": EXPECTED_ARCHIVE_SIZE,
        "source_file_md5_published": ARCHIVE_MD5,
        "source_archive_full_downloaded": False,
        "source_archive_full_md5_verified": False,
        "zip64_central_directory": {"offset": directory_offset, "size": directory_size,
                                     "entries": len(rows)},
        "selected_scene": SCENE,
        "requested_members": requested,
        "members": [],
    }
    for member in requested:
        dest = out / Path(member).name
        if dest.is_file():
            existing = dest.read_bytes()
            row = selected[member]
            crc = zlib.crc32(existing) & 0xFFFFFFFF
            if len(existing) != row["size"] or crc != row["crc32"]:
                raise RuntimeError(f"existing file fails official ZIP integrity check: {dest}")
            if crc != row["crc32"]:
                raise RuntimeError(f"existing file CRC mismatch: {dest}")
        else:
            extract_member(session, RECORD_URL, EXPECTED_ARCHIVE_SIZE, selected[member], dest)
    for name, member in MEMBER_FILES.items():
        dest = out / Path(member).name
        if not dest.is_file():
            continue
        row = all_by_name[member]
        digest = hashlib.sha256()
        crc = 0
        size = 0
        with dest.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                crc = zlib.crc32(block, crc)
                size += len(block)
        crc &= 0xFFFFFFFF
        if size != row["size"] or crc != row["crc32"]:
            raise RuntimeError(f"cached member fails official ZIP integrity check: {dest}")
        result["members"].append({
            "archive_member": member, "archive_member_size": size,
            "archive_member_compressed_size": row["compressed_size"],
            "zip_crc32_verified": f"{crc:08x}", "sha256": digest.hexdigest(),
            "bytes_written": size, "path": str(dest.resolve()), "reused_existing": name not in args.members,
        })
    (out / "subset_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
