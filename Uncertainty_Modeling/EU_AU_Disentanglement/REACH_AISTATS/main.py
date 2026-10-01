"""Single entry point for REACH experiments, cached evaluation, and tests."""
import argparse
from importlib import import_module
from pathlib import Path
import sys
import unittest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'separation', 'matched', 'conditionals',
                                            'evaluate', 'test'),
                        help='train: synthetic fitting/acquisition experiments; '
                             'separation: weak embedding-separation sweep; '
                             'matched: matched total width with unequal acquisition benefit; '
                             'conditionals: estimated references, reference error, shift stress; '
                             'evaluate: fit reference/log cache and evaluate held-out intervals; '
                             'test: run numerical regression tests')
    if not argv or argv[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    if args.command in ('train', 'separation', 'matched', 'conditionals', 'evaluate'):
        module = {'train': 'train.experiments', 'separation': 'train.separation',
                  'matched': 'train.matched', 'conditionals': 'train.conditionals',
                  'evaluate': 'train.cache'}[args.command]
        run = getattr(import_module(module), 'main')
        run(argv[1:])
    else:
        if argv[1:]:
            parser.error('test does not accept additional arguments')
        root = Path(__file__).resolve().parent
        suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
