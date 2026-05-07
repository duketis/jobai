#!/bin/bash
# jobai container entrypoint.
#
# Boot order:
#   1. As root: ensure /data is writable by the jobai user. The
#      named volume from compose may have been created with root
#      ownership on first ``docker compose up`` (before this image
#      switched to a non-root runtime); fix that idempotently here
#      so existing user data survives the upgrade.
#   2. Drop to the jobai user via ``setpriv`` and exec the real
#      command. ``setpriv --reuid jobai --regid jobai
#      --init-groups`` is the closest thing to ``gosu`` that
#      ships in Debian-slim by default.
#
# Why not USER jobai in the Dockerfile alone: a fresh volume mounts
# as root-owned no matter what USER the image declares, so the very
# first ``jobai migrate`` would fail with a read-only-DB error. This
# script makes the upgrade and first-run paths both Just Work.

set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
    chown -R jobai:jobai /data
    exec setpriv --reuid jobai --regid jobai --init-groups -- "$@"
fi

# Already non-root (e.g. compose ``user:`` override). Just exec.
exec "$@"
