"""Read-only integrity and dependency check for the public PRR archive."""
import ast
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
record=json.loads((root/'MANIFEST.json').read_text())
for name,info in record['files'].items():
    p=root/name
    assert p.is_file(),name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256'],name
for f in root.rglob('*.py'):
    ast.parse(f.read_text(),filename=str(f))
for f in (root/'manuscript').glob('*.tex'):
    text=f.read_text()
    for figure in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
        assert (f.parent/figure).is_file(),figure
    for part in re.findall(r'\\input\{([^}]+)\}',text):
        assert (f.parent/(part+'.tex')).is_file(),part
for name in record['files']:
    assert not re.search(r'(?:signature|cover_letter|pasted-text|\.DS_Store|\.git/)',name,re.I),name
for pattern in ('*signature*','*cover_letter*','*.nwb','*.parquet','*.mat'):
    assert not list(root.rglob(pattern)),pattern
assert len(json.loads((root/'manuscript/strengthening_results.json').read_text())['recurrent']['rows'])==48
assert (root/'manuscript/local_gain_loop_results.json').is_file()
assert (root/'manuscript/recurrent_feedback_results.json').is_file()
print('PASS:',len(record['files']),'hashed files; S9/S10 and TeX dependencies included.')
