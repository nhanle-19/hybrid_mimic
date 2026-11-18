"""Package the active input motions and exact T1 dependencies for a GPU server."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ASSET = Path('source/whole_body_tracking/whole_body_tracking/assets/booster/t1')
SOURCE = Path('retargeted_motion/training_list')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/force2_spider_h100.tar.gz')
    args = parser.parse_args()
    repo = ROOT / 'spider'
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', 'HEAD', '--', 'spider', 'LICENSE'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--', 'spider'], text=True)
    files = {Path('spider') / name for name in tracked.split('\0') if name.endswith('.py')}
    files.update(Path(name) for name in (
        'scripts/retarget_force2_spider.py', 'scripts/dsms_t1_model.py',
        'scripts/setup_force2_spider.sh', 'configs/spider_runtime.txt', 'README.md',
        'spider/LICENSE', 'spider/pyproject.toml',
        'source/whole_body_tracking/whole_body_tracking/robots/t1.py',
        'eval_data/reference_walk_20261016/hdm_bk_settled_env_geometry.npz',
    ))
    files.add(ASSET / 'wbc_acc_dynamics.json')
    dynamics = json.loads((ROOT / ASSET / 'wbc_acc_dynamics.json').read_text())
    for name, digest in dynamics['source_layers_sha256'].items():
        matches = list((ROOT / ASSET).rglob(name))
        assert len(matches) == 1 and hashlib.sha256(matches[0].read_bytes()).hexdigest() == digest, name
        files.add(matches[0].relative_to(ROOT))
    manifest = json.loads((ROOT / SOURCE / 'manifest.json').read_text())
    files.add(SOURCE / 'manifest.json')
    for motion in manifest['motions']:
        path = SOURCE / motion['file']
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == motion['sha256'], str(path)
        files.add(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(files)}
    metadata = dict(upstream_commit=commit, motions=len(manifest['motions']), files_sha256=hashes)
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in sorted(files):
            archive.add(ROOT / path, arcname=str(Path('force2_spider') / path), recursive=False)
        for name, data in (
            ('spider/UPSTREAM_COMMIT', (commit + '\n').encode()),
            ('transfer_manifest.json', (json.dumps(metadata, indent=2) + '\n').encode()),
        ):
            entry = tarfile.TarInfo('force2_spider/' + name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    print(f'{args.output}: {args.output.stat().st_size / 1024**2:.1f} MiB; {metadata["motions"]} motions; SPIDER {commit}')


if __name__ == '__main__':
    main()
