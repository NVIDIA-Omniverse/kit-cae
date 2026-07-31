"""Regression test for README markdown code fence languages."""

import unittest
from pathlib import Path


README = (
    Path(__file__).resolve().parents[1]
    / "templates"
    / "110.1.3"
    / "apps"
    / "usd_composer"
    / "README.md"
)


class TestReadmeCodeBlocks(unittest.TestCase):
    def test_web_viewer_clone_uses_bash_fence(self):
        content = README.read_text(encoding="utf-8")
        self.assertNotIn("\n```base\n", content)
        self.assertIn(
            "\n```bash\ngit clone https://github.com/NVIDIA-Omniverse/web-viewer-sample.git\n```\n",
            content,
        )

