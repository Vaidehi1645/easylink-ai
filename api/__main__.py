"""Run the API server:  python -m api"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("EASYLINK_PORT", "8000"))
    # 127.0.0.1 = reachable only from this machine (Phase 2 is local-only;
    # public hosting with HTTPS is Phase 5).
    uvicorn.run("api.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
