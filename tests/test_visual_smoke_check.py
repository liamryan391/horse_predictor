from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("visual_smoke_check", ROOT / "scripts" / "visual-smoke-check.py")
visual_smoke_check = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(visual_smoke_check)


class VisualSmokeCheckTests(unittest.TestCase):
    def test_reset_browser_flag_is_available_for_daemon_recovery(self) -> None:
        args = visual_smoke_check.parse_args(["--base-url", "http://127.0.0.1:5173", "--reset-browser"])

        self.assertTrue(args.reset_browser)
        self.assertEqual("http://127.0.0.1:5173", args.base_url)


if __name__ == "__main__":
    unittest.main()
