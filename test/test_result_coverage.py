"""Missing element stress must remain distinct from zero stress and GPSTRESS."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
from parser.dat_parser import Element,MystranModel
from parser.f06_parser import ElementStress,F06Results,ElementGPStress
from parser.result_coverage import missing_element_stresses,element_id_ranges
from gui.panels import ViewerState,_RESULTS
import main


class ResultCoverageTests(unittest.TestCase):
    def setUp(self):
        self.model=MystranModel(elements={
            500001:Element(500001,'CHEXA',list(range(1,9))),
            600001:Element(600001,'CHEXA',list(range(1,21))),
            800001:Element(800001,'CPYRAM',list(range(1,14)))})
        self.results=F06Results(subcases=[1],stresses={1:{
            600001:ElementStress(1,600001,'CHEXA',{'von_mises':0},0)}})

    def test_zero_is_present_gp_volume_does_not_fill_absent_element_results(self):
        self.results.gp_stresses[1]={(1,'V5'):ElementGPStress(1,1,ovm=123)}
        self.assertEqual(missing_element_stresses(self.model,self.results,1),
                         {'CHEXA8':[500001],'CPYRAM13':[800001]})
        self.assertNotIn(500001,self.results.stresses[1])
        self.assertEqual(element_id_ranges([500003,500001,500002,500008]),'500001-500003, 500008')

    def test_shell_fibers_are_not_offered_for_solids(self):
        state=ViewerState();state.results=self.results
        self.assertFalse(state.has_shell_stress_fibers())
        self.results.stresses[1][3]=ElementStress(1,3,'CQUAD4',{'von_mises_top':2})
        self.assertTrue(state.has_shell_stress_fibers())
        self.assertEqual(dict(_RESULTS)['von_mises'],'Element Stress - VM')

    def test_missing_element_hover_explains_absence(self):
        state=ViewerState();state.results=self.results;state.model=self.model
        state.display_mode='contour';state.result_type='von_mises'
        state.mouse_x=400;state.mouse_y=200
        with patch.object(main,'_nearest_elem_hover',return_value=(500001,self.model.elements[500001],0)), \
             patch.object(main,'_draw_hover_box') as box:
            main._draw_result_hover_info(self.model,None,1600,900,False,state)
        lines=box.call_args.args[0]
        self.assertIn('unavailable',lines[1])
        self.assertTrue(any('GPSTRESS Volume' in line for line in lines))
        self.assertFalse(any('=0' in line for line in lines))


if __name__=='__main__':
    unittest.main()
