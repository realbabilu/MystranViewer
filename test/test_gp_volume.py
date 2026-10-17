"""GPSTRESS volume invariants and UI selection regression checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from parser.f06_parser import F06Results, ElementGPStress
from parser.gp_volume import read_gp_volume_stress
from gui.panels import ViewerState, _current_result_label


class GPVolumeTests(unittest.TestCase):
    def table(self, volume, tensor, cid=0):
        return SimpleNamespace(ogs_id=volume, refid=cid, data_code={}, node=np.array([42]),
            data=np.array([[tensor]], dtype=float),
            get_headers=lambda: ['ox','oy','oz','txy','tyz','txz','pressure','ovm'])

    def test_shared_grid_volumes_do_not_overwrite_surface_or_each_other(self):
        results = F06Results()
        surface = ElementGPStress(1, 42, ovm=99)
        results.gp_stresses[1] = {(42, 'MID'): surface}
        op = SimpleNamespace(grid_point_stresses_volume_direct={
            (1, 1, 1, 0, 1): self.table(1, [3,2,1,0,0,0,-2,3**0.5]),
            (1, 1, 1, 0, 2): self.table(2, [0,0,0,4,0,0,0,48**0.5], cid=7)})
        self.assertEqual(read_gp_volume_stress(op, results), 2)
        self.assertIs(results.gp_stresses[1][42, 'MID'], surface)
        gp = results.gp_stresses[1][42, 'V1']
        self.assertEqual((gp.s1, gp.s2, gp.s3), (3,2,1))
        gp = results.gp_stresses[1][42, 'V2']
        np.testing.assert_allclose([gp.s1,gp.s2,gp.s3], [4,0,-4])
        self.assertEqual(gp.txy, 4)
        self.assertEqual(results.gp_volume_cids[1], {1:0, 2:7})
        state = ViewerState(); state.results = results
        state.result_type = 'gp_s3_v2'
        self.assertTrue(state.has_gp_stress())
        self.assertEqual(state.gp_stress_components(), {42:gp})
        self.assertIn('Volume 2 (CID 7)', _current_result_label(state))

    def test_empty_volume_table_preserves_other_results(self):
        results = F06Results(subcases=[3])
        self.assertEqual(read_gp_volume_stress(SimpleNamespace(), results), 0)
        self.assertEqual(results.subcases, [3])
        self.assertEqual(results.gp_stresses, {})


if __name__ == '__main__':
    unittest.main()
