import tempfile
import unittest
from pathlib import Path

from parser.dat_parser import load_dat


class IncludeTests(unittest.TestCase):
    def test_nested_deck_is_one_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'parts').mkdir()
            (root / 'main.bdf').write_text(
                "SOL 103\nCEND\nTITLE = Assembly\nBEGIN BULK\n"
                "INCLUDE 'parts/nodes.bdf' $ mesh\n"
                'include,"parts/ele\nments.bdf"\nENDDATA\n')
            (root / 'parts' / 'nodes.bdf').write_text(
                "GRID,1,,0.,0.,0.\nINCLUDE 'more.bdf'\n")
            (root / 'parts' / 'more.bdf').write_text('GRID,2,,1.,0.,0.\n')
            (root / 'parts' / 'elements.bdf').write_text('CROD,10,20,1,2\n')
            model = load_dat(str(root / 'main.bdf'))
            self.assertEqual(set(model.nodes), {1, 2})
            self.assertEqual(model.elements[10].nodes, [1, 2])
            self.assertEqual(model.title, 'Assembly')

    def test_missing_include_reports_path(self):
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / 'main.bdf'
            main.write_text("INCLUDE 'missing.bdf'\n")
            with self.assertRaisesRegex(OSError, 'missing.bdf'):
                load_dat(str(main))

    def test_cycle_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            main = Path(directory) / 'main.bdf'
            main.write_text("INCLUDE 'child.bdf'\n")
            (main.parent / 'child.bdf').write_text("INCLUDE 'main.bdf'\n")
            with self.assertRaisesRegex(ValueError, 'Circular INCLUDE'):
                load_dat(str(main))


if __name__ == '__main__':
    unittest.main()
