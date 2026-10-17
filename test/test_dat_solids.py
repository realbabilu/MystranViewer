"""Regression checks for large-field cards and quadratic solid connectivity."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser.dat_parser import load_dat


def large(name, values, label=''):
    return f'{name:<8}' + ''.join(f'{v:>16}' for v in values) + f'{label:<8}'


def small(name, values, label=''):
    return f'{name:<8}' + ''.join(f'{v:>8}' for v in values) + f'{label:<8}'


class DatSolidTests(unittest.TestCase):
    def parse(self, lines):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'model.dat'
            path.write_text('BEGIN BULK\n' + '\n'.join(lines) + '\nENDDATA\n')
            return load_dat(str(path))

    def test_large_grid_and_continuation_labels(self):
        model = self.parse([
            large('GRID*', [600001, '', '1.25', '-2.5'], '*A'),
            large('*A', ['3.75', '', '', '']),
            large('GRID*', [600002, '', '4.', '5.']),
            large('*', ['6.', '', '', '']),
        ])
        self.assertEqual(set(model.nodes), {600001, 600002})
        self.assertEqual(model.nodes[600001].xyz.tolist(), [1.25, -2.5, 3.75])
        self.assertEqual(model.nodes[600002].xyz.tolist(), [4., 5., 6.])

    def test_quadratic_solids_all_formats(self):
        for card, count in [('CHEXA', 20), ('CPENTA', 15), ('CTETRA', 10), ('CPYRAM', 13)]:
            values = [101, 1] + list(range(1, count + 1))
            for form in ('free', 'small', 'large'):
                with self.subTest(card=card, form=form):
                    lines = []
                    width = 4 if form == 'large' else 8
                    for i in range(0, len(values), width):
                        chunk = values[i:i+width]
                        if form == 'free':
                            lines.append(','.join(map(str, [card if i == 0 else '+'] + chunk)))
                        else:
                            chunk += [''] * (width - len(chunk))
                            if form == 'large':
                                lines.append(large(card+'*' if i == 0 else '*', chunk))
                            else:
                                lines.append(small(card if i == 0 else '', chunk))
                    model = self.parse(lines)
                    self.assertEqual(model.elements[101].nodes, list(range(1, count + 1)))

    def test_linear_solid_with_blank_optional_nodes(self):
        model = self.parse(['CTETRA,1,1,1,2,3,4,0,0', '+,0,0,0,0'])
        self.assertEqual(model.elements[1].nodes, [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
