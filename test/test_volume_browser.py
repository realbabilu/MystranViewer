"""Case-control recovery volumes and model-browser selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import MagicMock,patch
from parser.dat_parser import DatParser,MystranModel
from parser.f06_parser import F06Results,ElementGPStress
from gui.panels import ViewerState,_draw_model_tree


class VolumeBrowserTests(unittest.TestCase):
    def test_ranges_and_forward_set_reference(self):
        parser=DatParser()
        parser._parse_case_control(['VOLUME 2, SET 12, SYSTEM 7',
            'SET 12 = 10 THRU 16 BY 2, 20',
            'SET 13 = 100 THRU 102','VOLUME 8, SET 13, SYSTEM BASIC'])
        self.assertEqual(parser.model.volumes,{2:([10,12,14,16,20],'7'),8:([100,101,102],'BASIC')})

    def test_click_volume_selects_gp_and_right_click_clears(self):
        state=ViewerState();state.model=MystranModel(volumes={1:([101],'BASIC'),2:([102,103],'BASIC')})
        state.results=F06Results(gp_stresses={1:{(4,'V2'):ElementGPStress(1,4)}})
        state.selected_surface=8;state.result_type='gp_szz_v1'
        fake=MagicMock();fake.tree_node.side_effect=lambda label: label.startswith('Volumes')
        current=[''];labels=[]
        def selectable(label,*args):
            current[0]=label;labels.append(label);return False,False
        fake.selectable.side_effect=selectable
        fake.is_item_hovered.return_value=False
        fake.MouseButton_.left=0;fake.MouseButton_.right=1
        fake.is_item_clicked.side_effect=lambda button: current[0].startswith('Volume 2 ') and button==0
        with patch('gui.panels.imgui',fake):_draw_model_tree(state)
        self.assertEqual(labels,['Volume 1  (1 solids)','Volume 2  (2 solids)'])
        self.assertEqual(state.selected_volume,2);self.assertEqual(state.selected_surface,0)
        self.assertEqual(state.result_type,'gp_szz_v2');self.assertTrue(state.request_rebuild)
        fake.is_item_clicked.side_effect=lambda button: current[0].startswith('Volume 2 ') and button==1
        with patch('gui.panels.imgui',fake):_draw_model_tree(state)
        self.assertEqual(state.selected_volume,0)
        self.assertEqual(state.result_type,'gp_szz_v2')

    def test_volume_notation_detection_and_gp_only_nodes(self):
        from parser.gp_volume import volume_node_groups
        state=ViewerState();state.model=MystranModel()
        self.assertFalse(state.has_volumes())
        state.results=F06Results(gp_stresses={1:{(44,'V8'):ElementGPStress(1,44)}})
        self.assertTrue(state.has_volumes())
        self.assertEqual(volume_node_groups(state.model,state.results),{8:{44}})
        state.subcase=2
        self.assertFalse(state.has_volumes())
        state.model.volumes[3]=([99],'BASIC')
        self.assertTrue(state.has_volumes())


if __name__=='__main__':unittest.main()
