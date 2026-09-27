"""Every flatout module imports and every CLI subcommand parses (catches syntax / wiring errors)."""
import contextlib
import importlib
import io
import pkgutil
import sys
import unittest

import flatout


class CliSmoke(unittest.TestCase):
    def test_all_modules_import(self):
        for m in pkgutil.iter_modules(flatout.__path__):
            with self.subTest(module=m.name):
                importlib.import_module(f'flatout.{m.name}')

    def test_every_subcommand_help(self):
        from flatout import cli
        for cmd in ('sync', 'build', 'train', 'predict', 'evaluate', 'backtest', 'calibrate', 'nested',
                    'promote', 'snapshot', 'unseen', 'ratings', 'status', 'export', 'weekend'):
            with self.subTest(cmd=cmd):
                argv, sys.argv = sys.argv, ['flatout', cmd, '--help']
                try:
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as e:
                        cli.main()
                    self.assertEqual(e.exception.code, 0)
                finally:
                    sys.argv = argv


if __name__ == '__main__':
    unittest.main()
