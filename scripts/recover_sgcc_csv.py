"""
Recover the COMPLETE SGCC dataset from the spanned archive in data/raw/.

Why this exists (reproducibility, Task 1)
-----------------------------------------
The raw SGCC file ships as a WinZip *spanned* archive split across segments:

    data/raw/data.z01   (first segment, begins with the spanning marker PK\\x07\\x08)
    data/raw/data.z02
    data/raw/data.zip   (LAST segment, holds the central directory + EOCD)

Neither Windows Expand-Archive, Python's zipfile, nor bsdtar can open a spanned
archive directly (zipfile refuses multi-disk archives; the others ignore the
.z01/.z02 segments). This tool rebuilds the single logical ZIP byte-stream and
extracts the CSV WITHOUT trusting the recorded local-header offsets (spanned
archives store per-segment offsets, which are wrong for the rebuilt stream). It
locates each member by scanning for its local-file-header signature and inflates
the contiguous deflate stream in chunks, so peak memory stays low enough for an
8 GB host.

Output: the CSV named by config `paths.raw_sgcc_csv`
(default data/raw/recovered/data.csv). It then stream-counts rows and FLAG==1 to
confirm the dataset is the COMPLETE 42,372-row / 3,615-theft copy (not the
truncated 33,841-row extraction).

Run:
    python scripts/recover_sgcc_csv.py --config config/config.yaml
"""
import argparse
import io
import os
import struct
import sys
import zipfile
import zlib

import yaml

RAW_DIR = "data/raw"
EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"
SPAN_MARKER = b"PK\x07\x08"
EXPECTED_ROWS = 42372
EXPECTED_THEFT = 3615


def find_segments(raw_dir: str):
    """Return the spanned segments in correct order: .z01, .z02, ... , .zip last."""
    zs = sorted(
        f for f in os.listdir(raw_dir)
        if f.lower().startswith("data.z") and f.lower() != "data.zip"
    )
    parts = [os.path.join(raw_dir, f) for f in zs]
    final = os.path.join(raw_dir, "data.zip")
    if os.path.exists(final):
        parts.append(final)
    if not parts:
        raise SystemExit(f"[recover] no data.z* segments found in {raw_dir}")
    return parts


def build_stream(parts):
    """Concatenate segments into the original single-archive byte stream and drop
    the leading temporary spanning marker if present."""
    buf = bytearray()
    for p in parts:
        size = os.path.getsize(p)
        print(f"[recover] + segment {p} ({size:,} bytes)")
        with open(p, "rb") as fh:
            buf += fh.read()
    if bytes(buf[:4]) == SPAN_MARKER:
        print("[recover] stripped leading spanning marker PK\\x07\\x08")
        del buf[:4]
    return bytes(buf)


def parse_central_directory(buf: bytes):
    """Parse the EOCD + central directory. Offsets are captured but NOT trusted for
    data location (see find_local_header)."""
    eocd = buf.rfind(EOCD_SIG)
    if eocd < 0:
        raise SystemExit("[recover] EOCD signature not found - not a zip stream?")
    # EOCD: sig(4) disk(2) diskCD(2) nThis(2) nTotal(2) cdSize(4) cdOff(4) commLen(2)
    (sig, d0, d1, n_this, n_total, cd_size, cd_off, comm_len) = struct.unpack(
        "<IHHHHIIH", buf[eocd:eocd + 22]
    )
    print(f"[recover] EOCD: entries={n_total} cd_size={cd_size} cd_off={cd_off}")

    cd = buf.rfind(CD_SIG) if buf[cd_off:cd_off + 4] != CD_SIG else cd_off
    if cd < 0 or buf[cd:cd + 4] != CD_SIG:
        # fall back: first CD signature at/after the recorded offset region
        cd = buf.find(CD_SIG)
    entries = []
    i = cd
    while i < len(buf) and buf[i:i + 4] == CD_SIG:
        # CD entry fixed part is 46 bytes
        (sig, vmade, vneed, flag, method, mtime, mdate, crc, csize, usize,
         nlen, elen, clen, disk_start, iattr, eattr, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", buf[i:i + 46]
        )
        name = buf[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        entries.append({
            "name": name, "method": method, "csize": csize, "usize": usize,
            "crc": crc, "flag": flag, "lho": lho,
        })
        i += 46 + nlen + elen + clen
    print(f"[recover] central directory members: {[e['name'] for e in entries]}")
    return entries


def find_local_header(buf: bytes, entry):
    """Locate the member's local file header and return the offset where its
    compressed data begins. Tries the recorded offset (+/- the stripped marker)
    then falls back to a signature+filename scan."""
    name_b = entry["name"].encode("utf-8")
    candidates = [entry["lho"], entry["lho"] - 4, entry["lho"] + 4]
    for off in candidates:
        if 0 <= off < len(buf) and buf[off:off + 4] == LFH_SIG:
            nlen, elen = struct.unpack("<HH", buf[off + 26:off + 30])
            if buf[off + 30:off + 30 + nlen] == name_b:
                return off + 30 + nlen + elen
    # scan for LFH whose filename matches
    start = 0
    while True:
        off = buf.find(LFH_SIG, start)
        if off < 0:
            break
        nlen, elen = struct.unpack("<HH", buf[off + 26:off + 30])
        if buf[off + 30:off + 30 + nlen] == name_b:
            return off + 30 + nlen + elen
        start = off + 4
    raise SystemExit(f"[recover] could not locate local header for {entry['name']!r}")


def extract_entry(buf: bytes, entry, out_path: str):
    """Stream-decompress one member to out_path (low peak memory)."""
    data_off = find_local_header(buf, entry)
    method = entry["method"]
    csize = entry["csize"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "wb") as out:
        if method == 0:  # stored
            chunk = buf[data_off:data_off + csize]
            out.write(chunk)
            written = len(chunk)
        elif method == 8:  # deflate
            d = zlib.decompressobj(-zlib.MAX_WBITS)
            pos = data_off
            end = data_off + csize if csize else len(buf)
            step = 1 << 20
            while pos < end:
                block = buf[pos:pos + step]
                pos += step
                out.write(d.decompress(block))
                written += len(d.unconsumed_tail) * 0  # keep linter quiet
            out.write(d.flush())
        else:
            raise SystemExit(f"[recover] unsupported compression method {method}")
    actual = os.path.getsize(out_path)
    print(f"[recover] extracted {entry['name']!r} -> {out_path} "
          f"({actual:,} bytes; expected usize={entry['usize']:,})")
    if entry["usize"] and actual != entry["usize"]:
        print(f"[recover] WARNING: size mismatch (got {actual}, expected {entry['usize']})")
    return out_path


def try_zipfile(buf: bytes, out_path: str):
    """Fast path: if the rebuilt stream opens as a normal zip, use zipfile."""
    try:
        with zipfile.ZipFile(io.BytesIO(buf)) as zf:
            names = zf.namelist()
            print(f"[recover] zipfile opened stream directly; members={names}")
            csv_name = _pick_csv(names)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(csv_name) as src, open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            return out_path
    except Exception as e:
        print(f"[recover] zipfile fast path failed ({type(e).__name__}: {e}); "
              "falling back to manual central-directory parse")
        return None


def _pick_csv(names):
    csvs = [n for n in names if n.lower().endswith(".csv")]
    if not csvs:
        raise SystemExit(f"[recover] no .csv member in archive: {names}")
    # prefer the largest-sounding complete file; else first
    return sorted(csvs)[0]


def confirm_dataset(csv_path: str):
    """Stream-count rows and FLAG==1 (theft). Low memory: one line at a time."""
    rows = 0
    theft = 0
    header_flag_idx = None
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split(",")
        if "FLAG" in header:
            header_flag_idx = header.index("FLAG")
        else:
            print(f"[recover] WARNING: no FLAG column in header: {header[:5]}...")
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            if header_flag_idx is not None:
                parts = line.rstrip("\n").split(",")
                if header_flag_idx < len(parts):
                    try:
                        if int(float(parts[header_flag_idx])) == 1:
                            theft += 1
                    except ValueError:
                        pass
    print(f"[recover] CONFIRM raw rows = {rows:,} (expected {EXPECTED_ROWS:,})")
    print(f"[recover] CONFIRM theft (FLAG==1) = {theft:,} (expected {EXPECTED_THEFT:,})")
    ok = rows == EXPECTED_ROWS and theft == EXPECTED_THEFT
    print(f"[recover] {'OK - COMPLETE dataset' if ok else 'MISMATCH - see notes'}")
    return {"rows": rows, "theft": theft, "expected_rows": EXPECTED_ROWS,
            "expected_theft": EXPECTED_THEFT, "matches_complete_dataset": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--raw_dir", default=RAW_DIR)
    args = ap.parse_args()

    out_path = os.path.join("data", "raw", "recovered", "data.csv")
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        out_path = cfg.get("paths", {}).get("raw_sgcc_csv", out_path)

    parts = find_segments(args.raw_dir)
    buf = build_stream(parts)
    print(f"[recover] rebuilt stream = {len(buf):,} bytes")

    extracted = try_zipfile(buf, out_path)
    if extracted is None:
        entries = parse_central_directory(buf)
        csv_entries = [e for e in entries if e["name"].lower().endswith(".csv")]
        if not csv_entries:
            raise SystemExit(f"[recover] no CSV member in central directory: "
                             f"{[e['name'] for e in entries]}")
        target = max(csv_entries, key=lambda e: e["usize"])
        extract_entry(buf, target, out_path)

    del buf  # free before the (memory-heavy) confirmation pass
    stats = confirm_dataset(out_path)
    print(f"[recover] done -> {out_path}")
    return stats


if __name__ == "__main__":
    main()
