"""Small SONIC provisioning regressions; no network, GPU or model weights."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_sonic


class AssetTests(unittest.TestCase):
    def fetch(self, root, entries, revision=fetch_sonic.SOURCE_REVISION):
        def git(command, **kwargs):
            if "rev-parse" in command:
                return revision + "\n"
            if "ls-tree" in command:
                return "\n".join(f"{mode} blob {'0'*40}\t{name}" for name, (mode, _) in entries.items())
            return entries[command[-1].split(":", 1)[1]][1]

        with patch.object(fetch_sonic.subprocess, "check_output", side_effect=git), \
             patch.object(fetch_sonic, "MODEL_FILES", ()), \
             patch("sys.argv", ["fetch_sonic", "--upstream-root", str(root), "--output", str(root / "out")]):
            fetch_sonic.main()

    def test_preserve_sdk_symlinks(self):
        # Treating the symlink blob as a regular file can overwrite the actual
        # library when overlaid onto an existing source-export symlink.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {"sdk/lib.so": ("100644", b"ELF-test-payload"),
                       "sdk/lib.so.0": ("120000", b"lib.so")}
            self.fetch(root, entries)
            self.fetch(root, entries)
            library = root / "out/assets/sdk/lib.so"
            alias = library.with_name("lib.so.0")
            self.assertTrue(alias.is_symlink())
            self.assertEqual(library.read_bytes(), b"ELF-test-payload")
            self.assertEqual(alias.read_bytes(), library.read_bytes())
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["files"][1]["type"], "symlink")

    def test_reject_escaping_link(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unsafe.*symlink"):
                self.fetch(Path(directory), {"link": ("120000", b"../../../escape")})

    def test_reject_wrong_source_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "expected upstream"):
                self.fetch(Path(directory), {}, revision="wrong")

    def test_reject_corrupt_lfs_download(self):
        payload = b"correct"
        pointer = (f"version https://git-lfs.github.com/spec/v1\n"
                   f"oid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n").encode()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, size): return iter([b"corrupt"])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(fetch_sonic.requests, "get", return_value=Response()):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.fetch(root, {"mesh.stl": ("100644", pointer)})
            self.assertFalse((root / "out/assets/mesh.stl").exists())


if __name__ == "__main__":
    unittest.main()
