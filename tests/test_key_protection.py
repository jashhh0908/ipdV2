"""Protection of Team B's gold answer key (the tier1_eval_contract.md blinding rule).

Everything here works without the key: it checks the *places* the key could leak to, not its content.

* git: the key is ignored (``.gitignore``) and, where a git checkout exists, not tracked;
* the Kaggle notebook builders embed the contents of ``dkmem/`` and ``tests/`` (text, json and jsonl files) and nothing
  else, so no file in those two trees may hold gold labels -- a pair record carrying both a real ``ep_`` id and a
  ``relation``, or an original ``t1_NNNN`` pair id;
* the system code never imports the evaluation package (the key is read only by ``dkmem.eval``, given explicitly);
* no run output or report under ``results/``, ``runs/``, ``outputs/`` or ``validation/`` (JSON or JSONL) carries
  gold labels either.

Run: python -m unittest discover -s tests
"""

import ast
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent
EMBEDDED_TREES = ("dkmem", "tests")  # what validation/*/build_kernel.py embeds
EMBEDDED_SUFFIXES = (".py", ".json", ".jsonl", ".md", ".txt")
OUTPUT_TREES = ("results", "runs", "outputs", "validation")  # where run outputs, scores and reports are written
SYSTEM_PACKAGES = ("backends", "baselines", "memory", "pipeline", "store", "tier1")

REAL_EP_ID = re.compile(r"ep_[0-9a-f]{12}")
RELATION = re.compile(r"\"(?:gold_)?relation\"\s*:\s*\"(?:same|different)\"")
ORIGINAL_PAIR_ID = re.compile(r"\bt1_\d{4}\b")


def embedded_files():
    for top in EMBEDDED_TREES:
        for path in (REPO / top).rglob("*"):
            if path.is_file() and path.suffix in EMBEDDED_SUFFIXES and "__pycache__" not in path.parts:
                yield path


class TestKeyIsNotCommitted(unittest.TestCase):
    def test_gitignore_covers_the_key_wherever_it_is_placed(self):
        lines = {l.strip() for l in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()}
        self.assertIn("teamB_answer_key.jsonl", lines)
        self.assertIn("*answer_key*.jsonl", lines)

    @unittest.skipUnless((REPO / ".git").exists(), "not a git checkout (e.g. the Kaggle copy)")
    def test_no_tracked_file_is_an_answer_key(self):
        try:
            out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=30, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git is not available")
        self.assertEqual([f for f in out.splitlines() if "answer_key" in f.lower()], [])

    @unittest.skipUnless((REPO / ".git").exists(), "not a git checkout (e.g. the Kaggle copy)")
    def test_git_reports_the_key_path_as_ignored(self):
        try:
            r = subprocess.run(["git", "check-ignore", "-q", "teamB_answer_key.jsonl"], cwd=REPO, timeout=30)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("git is not available")
        self.assertEqual(r.returncode, 0)


class TestNothingEmbeddedHoldsGoldLabels(unittest.TestCase):
    def test_no_file_named_like_a_key_is_in_an_embedded_tree(self):
        self.assertEqual([str(p.relative_to(REPO)) for p in embedded_files() if "answer_key" in p.name.lower()], [])

    def test_no_embedded_file_holds_a_gold_record_or_an_original_pair_id(self):
        offenders = []
        for p in embedded_files():
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if (REAL_EP_ID.search(line) and RELATION.search(line)) or ORIGINAL_PAIR_ID.search(line):
                    offenders.append(f"{p.relative_to(REPO)}:{n}")
        self.assertEqual(offenders, [])

    def test_the_builders_embed_only_dkmem_and_tests(self):
        for builder in (REPO / "validation").glob("*/build_kernel.py"):
            src = builder.read_text(encoding="utf-8")
            if "for top in" in src:
                self.assertIn('for top in ("dkmem", "tests")', src, builder)

    def test_run_outputs_carry_no_gold_labels(self):
        # every place run outputs and reports land (gitignored or committed), JSON and JSONL alike
        offenders = []
        for top in OUTPUT_TREES:
            for p in (REPO / top).rglob("*"):
                if not p.is_file() or p.suffix not in (".json", ".jsonl"):
                    continue
                if "src" in p.relative_to(REPO / top).parts:  # embedded source copies
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                if (RELATION.search(text) and REAL_EP_ID.search(text)) or ORIGINAL_PAIR_ID.search(text):
                    offenders.append(str(p.relative_to(REPO)))
        self.assertEqual(offenders, [])


class TestSystemCodeNeverReadsTheKey(unittest.TestCase):
    def test_system_packages_do_not_import_the_evaluation_package(self):
        offenders = []
        for pkg in SYSTEM_PACKAGES:
            for path in (REPO / "dkmem" / pkg).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    if any(n == "dkmem.eval" or n.startswith("dkmem.eval.") for n in names):
                        offenders.append(f"{path.relative_to(REPO)}: {names}")
        self.assertEqual(offenders, [])

    def test_the_evaluation_package_is_the_only_reader_of_a_key(self):
        readers = []
        for path in (REPO / "dkmem").rglob("*.py"):
            rel = path.relative_to(REPO / "dkmem").parts
            if rel[0] == "eval":
                continue
            if "answer_key" in path.read_text(encoding="utf-8").lower() and "tier1/io.py" != "/".join(rel):
                readers.append("/".join(rel))
        self.assertEqual(readers, [])  # tier1/io.py only names the file in a docstring


if __name__ == "__main__":
    unittest.main()
