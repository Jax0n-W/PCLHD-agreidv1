"""Top-level launcher for the bundled SCHP mask generator.

Run ``python generate_schp_masks.py --help`` from the release root.
"""

from pathlib import Path
import runpy
import sys


def main():
    tool_dir = Path(__file__).resolve().parent / "tools" / "schp"
    tool_entry = tool_dir / "generate_schp_masks.py"
    if not tool_entry.is_file():
        raise FileNotFoundError("Bundled SCHP generator not found: {}".format(tool_entry))
    sys.path.insert(0, str(tool_dir))
    runpy.run_path(str(tool_entry), run_name="__main__")


if __name__ == "__main__":
    main()
