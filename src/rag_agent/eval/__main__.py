"""Enable ``python -m rag_agent.eval`` to run the evaluation CLI."""

from __future__ import annotations

from .harness import main

if __name__ == "__main__":
    raise SystemExit(main())
