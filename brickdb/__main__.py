"""Allow ``python -m brickdb`` to launch the interactive shell."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
