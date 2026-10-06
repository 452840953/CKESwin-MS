"""Portable entry points for the CKESwin-MS core implementation."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', default=str(ROOT / 'configs/paths.json'))
    parser.add_argument('command', choices=['train', 'evaluate', 'fuse', 'build-graphs', 'train-rf'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    remaining = args.arguments
    paths = json.loads(Path(args.paths).read_text(encoding='utf-8'))
    for key, value in paths.items():
        # Environment variables override the portable example configuration.
        os.environ.setdefault(key, str(value))
    sys.path.insert(0, str(ROOT / 'src'))
    sys.argv = [f'run.py {args.command}', *remaining]
    if args.command == 'train':
        p = argparse.ArgumentParser(description='Train CKESwin using the archived paper configuration.')
        p.add_argument('--config', default=str(ROOT / 'configs/visual.json'))
        opts = p.parse_args(remaining)
        config = json.loads(Path(opts.config).read_text(encoding='utf-8'))
        for key in ['test_file', 'swin_ckpt_path', 'rf_csv']:
            if not Path(config[key]).is_file():
                p.error(f'Missing {key}: {config[key]}')
        if not Path(config['test_file']).with_name('test.txt').is_file():
            p.error('Provide val.txt and test.txt in the same split directory.')
        module = importlib.import_module('train_tri_modal_swin_fusion_v3_2')
        cfg = module.TrainConfig(**config)
        module.main(cfg)
    elif args.command == 'build-graphs':
        p = argparse.ArgumentParser(description='Build vessel graphs from external images and detector weights.')
        p.parse_args(remaining)
        from dograph import WoodCellsGraphDataset
        WoodCellsGraphDataset(root=os.environ['SAVEROOT'], data_root=os.environ['IMAGEROOT'])
    else:
        name = {'evaluate': 'evaluation.visual', 'fuse': 'evaluation.fusion', 'train-rf': 'chemistry.train_rf'}[args.command]
        importlib.import_module(name).main()

if __name__ == '__main__':
    main()
