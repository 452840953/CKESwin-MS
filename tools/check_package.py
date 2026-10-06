"""Validate a source-only package without importing ML libraries or writing files."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {'.py', '.md', '.json', '.txt'}
ALLOWED_JSON = {'configs/paths.json', 'configs/visual.json', 'configs/rf.json'}

def main():
    failures = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.name not in {'.gitignore', 'LICENSE'} and path.suffix not in ALLOWED_SUFFIXES:
            failures.append(f'Non-source artifact: {relative}')
        if path.suffix == '.json' and relative not in ALLOWED_JSON:
            failures.append(f'Unexpected JSON artifact: {relative}')
        if path.suffix == '.txt' and not path.name.startswith('requirements'):
            failures.append(f'Unexpected text data: {relative}')
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=relative)
            except SyntaxError as exc:
                failures.append(str(exc))
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {len(files)} source and configuration files.')

if __name__ == '__main__':
    main()
