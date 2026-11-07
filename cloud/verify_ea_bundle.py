"""Verify the additive delivery and the unchanged shared experiment files."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/'ea_extension_bundle.json').read_text(encoding='utf-8'))
    checked=0
    for section in ('new_files','shared_files'):
        for name,expected in manifest[section].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise RuntimeError('Missing/invalid path: '+name)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):
                    h.update(block)
            if h.hexdigest()!=expected:
                raise RuntimeError('File hash mismatch: '+name)
            checked+=1
    print(f'EA_BUNDLE_VERIFIED: {checked} files; existing training sources unchanged; no Test evaluation')
    return manifest


if __name__=='__main__':
    verify()
