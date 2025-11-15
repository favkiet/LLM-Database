import sys
import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from src.state import ChatState
from src.nodes import text2sql_node

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


@dataclass
class SampleResult:
    exact_match: bool
    valid: bool
    ves: float
    details: Dict[str, float]


class Text2SQLEvaluator:
    """
    Independent evaluator for Text-to-SQL outputs.
    Metrics:
    - EM (Exact Match): String match after normalization (case/space/semicolon).
    - VES (Valid Efficiency Score): 0 if invalid SQL (by heuristic). Otherwise,
      an efficiency ratio based on token length relative to reference.
        VES = valid_flag * min(len_ref, len_pred) / max(len_ref, len_pred)
      This favors queries similar in complexity/length to the reference while
      remaining independent of execution or external parsers.
    """

    def __init__(
        self,
        collapse_whitespace: bool = True,
        lowercase: bool = True,
        strip_semicolon: bool = True,
        remove_redundant_spaces_around_symbols: bool = True,
    ) -> None:
        self.collapse_whitespace = collapse_whitespace
        self.lowercase = lowercase
        self.strip_semicolon = strip_semicolon
        self.remove_redundant_spaces_around_symbols = remove_redundant_spaces_around_symbols

        # Basic SQL keywords for validation heuristic
        self._sql_starters = {
            "select",
            "insert",
            "update",
            "delete",
            "create",
            "drop",
            "alter",
            "with",
            "replace",
        }

    # ------------- Normalization -------------
    def normalize_sql(self, sql: str) -> str:
        if sql is None:
            return ""
        text = sql.strip()
        if self.strip_semicolon:
            text = text[:-1] if text.endswith(";") else text
        if self.lowercase:
            text = text.lower()
        if self.remove_redundant_spaces_around_symbols:
            # Normalize spaces around common SQL symbols and commas
            text = re.sub(r"\s*,\s*", ", ", text)
            text = re.sub(r"\s*\(\s*", "(", text)
            text = re.sub(r"\s*\)\s*", ")", text)
            text = re.sub(r"\s*=\s*", " = ", text)
            text = re.sub(r"\s*>\s*", " > ", text)
            text = re.sub(r"\s*<\s*", " < ", text)
            text = re.sub(r"\s*>=\s*", " >= ", text)
            text = re.sub(r"\s*<=\s*", " <= ", text)
            text = re.sub(r"\s*<>\s*", " <> ", text)
            text = re.sub(r"\s*!=\s*", " != ", text)
            text = re.sub(r"\s*\+\s*", " + ", text)
            text = re.sub(r"\s*-\s*", " - ", text)
            text = re.sub(r"\s*/\s*", " / ", text)
            text = re.sub(r"\s*\*\s*", " * ", text)
        if self.collapse_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, sql: str) -> List[str]:
        normalized = self.normalize_sql(sql)
        # Split on word boundaries and common SQL symbols
        return re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[(),.*=<>!+\-\/]", normalized
        )

    # ------------- Metrics -------------
    def exact_match(self, pred_sql: str, ref_sql: str) -> bool:
        return self.normalize_sql(pred_sql) == self.normalize_sql(ref_sql)

    def is_valid_sql(self, sql: str) -> bool:
        """
        Heuristic syntactic sanity checks without external dependencies:
        - Non-empty, balanced parentheses, balanced quotes (single/double/backticks).
        - Starts with a common SQL starter keyword.
        - No multiple statements separated by semicolons.
        """
        if sql is None:
            return False
        text = sql.strip()
        if not text:
            return False
        # No multiple statements
        if text.count(";") > 1 or (text.endswith(";") and text[:-1].count(";") > 0):
            return False

        norm = self.normalize_sql(text)
        # Starter keyword
        starter = norm.split(" ", 1)[0] if norm else ""
        if starter not in self._sql_starters:
            return False

        # Balanced parentheses
        if not self._balanced_parens(norm):
            return False

        return bool(self._balanced_quotes(text))

    def ves(self, pred_sql: str, ref_sql: str) -> float:
        valid = self.is_valid_sql(pred_sql)
        if not valid:
            return 0.0
        pred_tokens = self.tokenize(pred_sql)
        ref_tokens = self.tokenize(ref_sql)
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            # If either is empty after tokenization, treat as invalid/zero efficiency
            return 0.0
        efficiency_ratio = min(len(ref_tokens), len(pred_tokens)) / max(len(ref_tokens), len(pred_tokens))
        return float(efficiency_ratio)

    # ------------- Helpers -------------
    def _balanced_parens(self, text: str) -> bool:
        depth = 0
        for ch in text:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    def _balanced_quotes(self, text: str) -> bool:
        # Track single ('), double ("), and backtick (`) quotes, ignoring escaped ones.
        single = double = backtick = False
        i = 0
        n = len(text)
        for i in range(n):
            ch = text[i]
            prev_escape = i > 0 and text[i - 1] == "\\"
            if ch == "'" and not prev_escape and not double and not backtick:
                single = not single
            elif ch == '"' and not prev_escape and not single and not backtick:
                double = not double
            elif ch == "`" and not prev_escape and not single and not double:
                backtick = not backtick
            i += 1
        return not single and not double and not backtick

    # ------------- Public API -------------
    def evaluate_pair(self, pred_sql: str, ref_sql: str) -> SampleResult:
        em = self.exact_match(pred_sql, ref_sql)
        valid = self.is_valid_sql(pred_sql)
        ves_score = self.ves(pred_sql, ref_sql)
        return SampleResult(
            exact_match=em,
            valid=valid,
            ves=ves_score,
            details={
                "pred_tokens": float(len(self.tokenize(pred_sql))),
                "ref_tokens": float(len(self.tokenize(ref_sql))),
            },
        )

    def evaluate(
        self,
        predictions: Iterable[str],
        references: Iterable[str],
        return_per_sample: bool = False,
    ) -> Dict[str, object]:
        preds = list(predictions)
        refs = list(references)
        if len(preds) != len(refs):
            raise ValueError("predictions and references must have the same length")

        results: List[SampleResult] = [self.evaluate_pair(p, r) for p, r in zip(preds, refs)]
        em_scores = [1.0 if r.exact_match else 0.0 for r in results]
        valid_flags = [1.0 if r.valid else 0.0 for r in results]
        ves_scores = [r.ves for r in results]

        def avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        summary: Dict[str, object] = {
            "num_samples": len(results),
            "EM": avg(em_scores),
            "Valid_Ratio": avg(valid_flags),
            "VES": avg(ves_scores),
        }
        if return_per_sample:
            summary["per_sample"] = [
                {
                    "exact_match": r.exact_match,
                    "valid": r.valid,
                    "ves": r.ves,
                    "details": r.details,
                }
                for r in results
            ]
        return summary

    # ------------- I/O Utilities -------------
    @staticmethod
    def _read_jsonl(path: str, pred_key_candidates: List[str], ref_key_candidates: List[str]) -> Tuple[List[str], List[str]]:
        predictions: List[str] = []
        references: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pred = Text2SQLEvaluator._first_present_key(obj, pred_key_candidates)
                ref = Text2SQLEvaluator._first_present_key(obj, ref_key_candidates)
                if pred is None or ref is None:
                    raise KeyError(
                        "Missing prediction/reference keys in JSONL. "
                        f"Looked for pred keys {pred_key_candidates}, ref keys {ref_key_candidates}."
                    )
                predictions.append(str(pred))
                references.append(str(ref))
        return predictions, references

    @staticmethod
    def _read_csv(
        path: str,
        pred_col: str,
        ref_col: str,
    ) -> Tuple[List[str], List[str]]:
        predictions: List[str] = []
        references: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append(str(row[pred_col]))
                references.append(str(row[ref_col]))
        return predictions, references

    @staticmethod
    def _read_parallel_text(pred_path: str, ref_path: str) -> Tuple[List[str], List[str]]:
        with open(pred_path, "r", encoding="utf-8") as fp:
            preds = [line.rstrip("\n") for line in fp]
        with open(ref_path, "r", encoding="utf-8") as fr:
            refs = [line.rstrip("\n") for line in fr]
        return preds, refs

    @staticmethod
    def _first_present_key(obj: Dict[str, object], candidates: List[str]) -> Optional[object]:
        return next((obj[key] for key in candidates if key in obj), None)

    @staticmethod
    def _read_single_text(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    @staticmethod
    def _read_test_text2sql_txt(path: str) -> List[str]:
        """
        Parse a test_text2sql.txt-style file:
        - Lines alternate: a numbered question line, followed by a SQL line, then a blank line.
        - Extract and return the SQL lines as references in order.
        """
        references: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            # Detect numbered prompt like "1." or "1.\t..."
            if re.match(r"^\d+\.\s", line) or re.match(r"^\d+\.\t", line) or re.match(r"^\d+\.", line):
                # Next non-empty line should be the SQL reference
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n:
                    references.append(lines[j].strip())
                    # Advance i beyond the SQL and any following blank line
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        return references

    @staticmethod
    def _read_test_text2sql_pairs(path: str) -> List[Tuple[str, str]]:
        """
        Return list of (question, reference_sql) pairs from test_text2sql.txt-style file.
        """
        pairs: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if re.match(r"^\d+\.", line):
                # Extract question text after "n."
                question = re.sub(r"^\d+\.\s*", "", line).strip()
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n:
                    sql_line = lines[j].strip()
                    pairs.append((question, sql_line))
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        return pairs


def _save_json(path: str, data: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    evaluator = Text2SQLEvaluator()

    # Fixed path: data/test_text2sql.txt relative to project root (this file's directory)
    project_root = Path(__file__).resolve().parent
    testfile_path = project_root / "data" / "test_text2sql.txt"
    db_path = project_root / "data" / "retails" / "retails.sqlite"
    out_path = project_root / "data" / "evaluation" / "result_text2sql_rag.json"
    pairs = evaluator._read_test_text2sql_pairs(str(testfile_path))

    predictions: List[str] = []
    references: List[str] = []
    pred_results: List[str] = []
    ref_results: List[str] = []

    def execute_sql(sql: str) -> str:
        sql_clean = (sql or "").strip()
        if not sql_clean:
            return ""
        try:
            with sqlite3.connect(db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql_clean)
                rows = cursor.fetchall()
                if not rows:
                    return ""
                df = pd.DataFrame([dict(row) for row in rows])
                return "\n".join(
                    [", ".join(f"{col}={row[col]}" for col in df.columns) for _, row in df.iterrows()]
                )
        except Exception as e:
            return f"DB error: {e}"

    for question, ref_sql in pairs:
        # Build initial state and invoke text2sql_node to get predicted SQL
        state = ChatState(
            human_messages=question,
            ai_messages="",
            query_sql="",
            result_query="",
            conversation=[],
        )
        state = text2sql_node(state)
        pred_sql = state.query_sql or ""
        predictions.append(pred_sql)
        references.append(ref_sql)
        pred_results.append(execute_sql(pred_sql))
        ref_results.append(execute_sql(ref_sql))

    # Compute metrics and assemble detailed per-sample report
    results = [evaluator.evaluate_pair(p, r) for p, r in zip(predictions, references)]
    em_scores = [1.0 if r.exact_match else 0.0 for r in results]
    valid_flags = [1.0 if r.valid else 0.0 for r in results]
    ves_scores = [r.ves for r in results]

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    per_sample = []
    for idx, ((question, ref_sql), pred_sql, res, pr, rr) in enumerate(
        zip(pairs, predictions, results, pred_results, ref_results), start=1
    ):
        per_sample.append(
            {
                "id": idx,
                "question": question,
                "pred_sql": pred_sql,
                "ref_sql": ref_sql,
                "pred_result": pr,
                "ref_result": rr,
                "exact_match": res.exact_match,
                "valid": res.valid,
                "ves": res.ves,
                "details": res.details,
            }
        )

    summary: Dict[str, object] = {
        "num_samples": len(results),
        "EM": avg(em_scores),
        "Valid_Ratio": avg(valid_flags),
        "VES": avg(ves_scores),
        "per_sample": per_sample,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

