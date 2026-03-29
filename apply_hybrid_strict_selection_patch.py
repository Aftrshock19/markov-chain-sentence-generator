#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys

STRICT_PUBLISHABLE_OLD = '''def candidate_is_hybrid_publishable(self, candidate: Optional[Candidate]) -> bool:
        if not candidate or not candidate.sentence:
            return False
        if self.candidate_is_general_publishable(candidate):
            return not self._hybrid_policy_reasons(candidate)
        return self._safe_template_publishable_override(candidate)
'''

STRICT_PUBLISHABLE_NEW = '''def candidate_is_hybrid_publishable(self, candidate: Optional[Candidate]) -> bool:
        if not candidate or not candidate.sentence:
            return False
        if not self.candidate_is_general_publishable(candidate):
            return False
        return not self._hybrid_policy_reasons(candidate)
'''

CHOOSE_FROM_POOL_OLD = '''def _choose_from_pool(
        self,
        pool: List[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
        target: Lexeme,
    ) -> Candidate:
        trimmed = self.dedupe_candidates(pool)[: self.max_candidates_to_keep]
        winner: Optional[Candidate] = None
        family = self.normalized_pos_family(target)
        if trimmed:
            ranked = sorted(trimmed, key=lambda candidate: self._candidate_bucket(candidate, family))
            top_bucket = self._candidate_bucket(ranked[0], family)
            bucket_candidates = [
                candidate for candidate in ranked
                if self._candidate_bucket(candidate, family) == top_bucket
            ]
            winner = self._choose_best_in_bucket(bucket_candidates)

        if winner is not None and self._candidate_is_valid(winner):
            return winner
        if best_valid is not None:
            return best_valid
        if best_any is not None and best_any.sentence:
            return best_any
        return self._no_candidate_result(target)
'''

CHOOSE_FROM_POOL_NEW = '''def _choose_from_pool(
        self,
        pool: List[Candidate],
        best_valid: Optional[Candidate],
        best_any: Optional[Candidate],
        target: Lexeme,
    ) -> Candidate:
        trimmed = self.dedupe_candidates(pool)[: self.max_candidates_to_keep]
        publishable_candidates = [
            candidate for candidate in trimmed
            if self.candidate_is_hybrid_publishable(candidate)
        ]
        if publishable_candidates:
            return self._choose_best_in_bucket(publishable_candidates)
        return self._no_candidate_result(target)
'''

HYBRID_METADATA_OLD = '''        failure_reason = ""
        if not publishable:
            failure_reason = "; ".join(self._hybrid_failure_reasons(candidate))
'''

HYBRID_METADATA_NEW = '''        failure_reason = ""
        if candidate.source_method == "no_candidate_found":
            failure_reason = "no_publishable_candidate"
        elif not publishable:
            failure_reason = "; ".join(self._hybrid_failure_reasons(candidate))
'''

REPLACEMENTS = [
    (STRICT_PUBLISHABLE_OLD, STRICT_PUBLISHABLE_NEW, 'strict hybrid publishable gate'),
    (CHOOSE_FROM_POOL_OLD, CHOOSE_FROM_POOL_NEW, 'publishable-only final selection'),
    (HYBRID_METADATA_OLD, HYBRID_METADATA_NEW, 'clear no-candidate failure reason'),
]


def apply_patch(path: Path, dry_run: bool) -> None:
    text = path.read_text(encoding='utf-8')
    original = text
    applied = []
    for old, new, label in REPLACEMENTS:
        if old not in text:
            raise RuntimeError(f'{label} block not found in {path}')
        text = text.replace(old, new, 1)
        applied.append(label)
    if text == original:
        print(f'No changes needed in {path}')
        return
    if dry_run:
        print(f'[dry-run] Would patch {path} with: {", ".join(applied)}')
        return
    backup = path.with_suffix(path.suffix + '.bak')
    shutil.copy2(path, backup)
    path.write_text(text, encoding='utf-8')
    print(f'Patched {path}')
    print(f'Backup  {backup}')
    print('Applied ' + ', '.join(applied))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hybrid-generator', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    apply_patch(Path(args.hybrid_generator), args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Patch failed: {exc}', file=sys.stderr)
        sys.exit(1)
