"""
Startup cache cleanup.

Uploaded media, extracted frames, BEV renders, and process_meta/trace JSON
sidecars are all treated as disposable working data for the current
session only. Exported projects (ZIP exports, "Save State" JSON) are
independent, self-contained snapshots the user has explicitly asked to
keep -- they don't live in this folder and are never touched here.

This module is called once automatically at application startup (see
app.py) so nobody has to remember to run a cleanup script manually before
starting a new session.
"""
import os
import time


def clear_uploads(upload_dir, older_than_days=None, dry_run=False, quiet=False):
    """
    Remove all files from the given upload/cache directory.

    older_than_days: if set, only delete files older than N days (useful
        for manual/CLI use); startup calls always pass None so every
        previous session's leftovers are wiped before the app comes up.
    dry_run: preview only, don't delete anything.
    quiet: suppress console logging (startup call keeps this False so the
        cleanup is visible in server logs).
    """
    if not os.path.isdir(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        return {"deleted": 0, "kept": 0, "bytes": 0}

    now = time.time()
    cutoff = now - (older_than_days * 86400) if older_than_days else None

    total_size = 0
    total_count = 0
    kept_count = 0

    for entry in os.scandir(upload_dir):
        if not entry.is_file():
            continue
        stat = entry.stat()
        if cutoff is not None and stat.st_mtime > cutoff:
            kept_count += 1
            continue

        total_size += stat.st_size
        total_count += 1

        if dry_run:
            continue
        try:
            os.remove(entry.path)
        except OSError as e:
            if not quiet:
                print(f"[-] Failed to delete {entry.path}: {e}")

    if not quiet:
        action = "Would delete" if dry_run else "Deleted"
        print(f"[+] Startup cleanup: {action} {total_count} cached file(s) "
              f"({total_size / (1024**3):.2f} GB)")
        if older_than_days:
            print(f"[+] Kept {kept_count} file(s) newer than {older_than_days} day(s)")

    return {"deleted": total_count, "kept": kept_count, "bytes": total_size}
