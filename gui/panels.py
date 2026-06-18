"""ImGui panels for MYSTRAN Viewer."""

import os
from typing import Optional
from imgui_bundle import imgui
try:
    from imgui_bundle import portable_file_dialogs as pfd
    _HAS_PFD = True
except Exception:
    _HAS_PFD = False

from parser.dat_parser import MystranModel
from parser.f06_parser  import F06Results
from renderer.contour   import legend_ticks, COLORMAPS

# View presets
_VIEW_PRESETS = [
    ("X-Z",   0, -89, "Front: X right, Z up"),
    ("Y-Z",  90,   0, "Side: Y right, Z up"),
    ("X-Y",   0,   0, "Top: X right, Y up"),
]


class ViewerState:
    def __init__(self):
        self.dat_path: str = ""
        self.f06_path: str = ""
        self.display_mode: str  = "solid"
        self.result_type: str   = "displacement"
        self.cmap_name: str     = "rainbow"
        self.subcase: int       = 1
        self.deform_scale: float= 0.0
        self.show_edges: bool   = True
        self.show_spc:   bool   = True
        self.show_nodes:       bool = False
        self.show_node_nums:   bool = False
        self.show_elem_nums:   bool = False
        self.show_node_values: bool = False
        self.show_elem_values: bool = False
        self.nodal_result_source: str = "solver_first"
        self.averaging_scope: str = "property"
        self.average_same_family: bool = True
        self.average_transform_shell: bool = True
        self.average_angle_deg: float = 20.0
        self.contour_notice: str = ""
        self.open_contour_notice: bool = False
        self.show_reaction_forces: bool = False
        self.show_reaction_moments: bool = False
        self.reaction_engineering: bool = False
        self.show_constraints: bool = True
        # Local axes
        self.show_local_axes:  bool = False
        self.local_axis_shell: bool = True
        self.local_axis_frame: bool = True
        self.local_axis_solid: bool = False
        # Loads
        self.show_forces:   bool = False
        self.show_moments:  bool = False
        self.show_pressure: bool = False
        self.active_load_case: int = 0  # 0 = all cases
        self.load_filter_mode: str = "subcase"  # subcase | all | manual
        self.active_spc_case: int = 0
        self.spc_filter_mode: str = "subcase"  # subcase | all | manual
        self.visible_dims: list = [True, True, True]
        self.show_deformed: bool = True
        self.show_undeformed: bool = False
        self.ortho_mode:  bool  = True
        # Beam diagrams
        self.show_beam_panel:    bool  = False
        self.beam_result_key:    str   = 'bm1'
        self.beam_scale:         float = 0.0
        self.beam_alpha:         float = 0.28
        self.show_station_nodes: bool  = False
        self.beam_diagram              = None
        self.beam_source_path:   str   = ""
        self.beam_loaded_subcase:int   = 0
        self.request_export_mp4: bool  = False
        self.request_export_png: bool  = False
        self.exporting_mp4: bool       = False
        self.export_progress: float    = 0.0
        self.request_beam_rebuild:bool = False
        self.model:   Optional[MystranModel] = None
        self.results: Optional[F06Results]   = None
        self.selected_node: int = -1
        self.selected_property_id: int = -1
        self.selected_material_id: int = -1
        self.show_property_window: bool = False
        self.show_material_window: bool = False
        self.hover_status: str = ""
        self.mouse_x: float = -1.0
        self.mouse_y: float = -1.0
        self.fps: float        = 0.0
        self.frame_time: float = 0.0
        self.request_load_dat: bool = False
        self.request_load_f06: bool = False
        self.request_fit:      bool = False
        self.request_rebuild:  bool = False
        self.request_quit:     bool = False
        self._dat_dialog = None
        self._f06_dialog = None

    def visible_dim_tuple(self):
        dims=[]
        if self.visible_dims[0]: dims.append('1d')
        if self.visible_dims[1]: dims.append('2d')
        if self.visible_dims[2]: dims.append('3d')
        return tuple(dims)

    def _stress_store(self):
        if not self.results or self.subcase not in self.results.stresses:
            return {}
        return self.results.stresses[self.subcase]

    def _force_store(self):
        if not self.results or self.subcase not in getattr(self.results, 'forces', {}):
            return {}
        return self.results.forces[self.subcase]

    def nodal_stress_components(self):
        st = self._stress_store()
        return st.get('_derived_nodal_avg_components', {})

    def nodal_force_components(self):
        fc = self._force_store()
        if self.nodal_result_source == "derived":
            return fc.get('_derived_nodal_avg', {})
        return fc.get('_solver_nodal_avg', {}) or fc.get('_derived_nodal_avg', {})

    def has_solver_nodal_stress(self) -> bool:
        return bool(self._stress_store().get('_solver_nodal_avg_components', {}))

    def has_derived_nodal_stress(self) -> bool:
        return bool(self._stress_store().get('_derived_nodal_avg_components', {}))

    def has_corner_stress_data(self) -> bool:
        return bool(self._stress_store().get('_solver_nodal_avg_components', {}))

    def has_solver_nodal_force(self) -> bool:
        return bool(self._force_store().get('_solver_nodal_avg', {}))

    def has_derived_nodal_force(self) -> bool:
        return bool(self._force_store().get('_derived_nodal_avg', {}))


def _open_dialog(title, filters):
    if _HAS_PFD:
        return pfd.open_file(title, filters=filters)
    return None

def _dialog_result(dlg):
    if dlg is None: return None
    try:
        if dlg.ready():
            r = dlg.result()
            return r[0] if r else None
    except Exception:
        pass
    return None


def _subcase_display_label(state: ViewerState, sc: int) -> str:
    eigen = getattr(state, 'eigen_data', {})
    if sc in eigen:
        ed = eigen[sc]
        if ed.get('is_buckling'):
            eig = ed.get('eigenvalue', ed.get('load_factor', 0.0))
            return f"Buckling - Eigenvalue {eig:.4g}"
        return f"Mode {ed['mode']} - {ed['freq_hz']:.3g} Hz"
    return f"SC {sc}"


def _nodal_source_suffix(state: ViewerState, kind: str) -> str:
    if kind == 'stress':
        derived = state.has_derived_nodal_stress()
    else:
        solver = state.has_solver_nodal_force()
        derived = state.has_derived_nodal_force()
    if kind == 'stress':
        return ' [Viewer avg]' if derived else ' [Nodal]'
    if derived:
        return ' [Derived]'
    return ' [Nodal]'


def _nodal_source_title(state: ViewerState, kind: str) -> str:
    if kind == 'stress':
        derived = state.has_derived_nodal_stress()
    else:
        solver = state.has_solver_nodal_force()
        derived = state.has_derived_nodal_force()
    if kind == 'stress':
        return ' [Viewer avg]' if derived else ' [Nodal]'
    if derived:
        return ' [Viewer avg]'
    return ' [Nodal]'


def _format_legend_tick(tick: float) -> str:
    if tick == 0.0:
        return "0.000"
    if abs(tick) < 1e-2:
        return f"{tick:.3e}"
    text = f"{tick:.3f}"
    if len(text) <= 10:
        return text
    return f"{tick:.3e}"


def _current_result_label(state: ViewerState):
    if state.results is None:
        return None
    sc = state.subcase
    r = state.results
    rt = state.result_type
    if rt == 'displacement' and sc in r.displacements:
        return "Total Translation"
    if rt in ('t1','t2','t3') and sc in r.displacements:
        return {'t1':'T1 (X)','t2':'T2 (Y)','t3':'T3 (Z)'}[rt]
    if rt in ('von_mises','oxx','oyy','txy','omax','omin',
              'von_mises_top','von_mises_bottom') and sc in r.stresses:
        base = {'von_mises':'Von Mises [Elem]','oxx':'Sxx [Elem]','oyy':'Syy [Elem]',
                'txy':'Sxy [Elem]','omax':'S1 [Elem]','omin':'S3 [Elem]',
                'von_mises_top':'Von Mises Top [Elem]',
                'von_mises_bottom':'Von Mises Bottom [Elem]'}.get(rt, 'Stress [Elem]')
        if not state.has_corner_stress_data():
            base += ' [center only]'
        return base
    if rt in ('sxc','sxd','sxe','sxf','smax','smin'):
        return {
            'sxc': 'Beam Stress C [Solver]',
            'sxd': 'Beam Stress D [Solver]',
            'sxe': 'Beam Stress E [Solver]',
            'sxf': 'Beam Stress F [Solver]',
            'smax': 'Beam Stress Smax [Solver]',
            'smin': 'Beam Stress Smin [Solver]',
        }[rt]
    if rt == 'stress3d':
        return 'Beam Stress 3D [CDEF interp]'
    if rt in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin') and sc in r.stresses:
        base = {
            'nodal_vm':'Von Mises',
            'noxx':'Sxx',
            'noyy':'Syy',
            'ntxy':'Sxy',
            'nomax':'S1',
            'nomin':'S3',
        }[rt]
        return base + _nodal_source_title(state, 'stress')
    if rt in ('fx','fy','fxy','mx','my','mxy','qx','qy') and hasattr(r, 'forces') and sc in r.forces:
        return {'fx':'FX [Elem]','fy':'FY [Elem]','fxy':'FXY [Elem]','mx':'MX [Elem]',
                'my':'MY [Elem]','mxy':'MXY [Elem]','qx':'QX [Elem]','qy':'QY [Elem]'}.get(rt, rt.upper())
    if rt in ('nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy') and hasattr(r, 'forces') and sc in r.forces:
        base_rt = {'nfx':'FX','nfy':'FY','nfxy':'FXY','nmx':'MX','nmy':'MY',
                   'nmxy':'MXY','nqx':'QX','nqy':'QY'}.get(rt, rt.upper())
        return base_rt + _nodal_source_title(state, 'force')
    return None


def _status_context_label(state: ViewerState):
    bits = []
    if state.results is not None:
        bits.append(_subcase_display_label(state, state.subcase))
    load_sid = _subcase_load_sid(state) if state.load_filter_mode == "subcase" else state.active_load_case
    if load_sid > 0:
        bits.append(f"Load {load_sid}")
    spc_sid = _subcase_spc_sid(state) if state.spc_filter_mode == "subcase" else state.active_spc_case
    if spc_sid > 0:
        bits.append(f"BC {spc_sid}")
    if (state.display_mode == 'contour'
            and state.result_type in ('von_mises','oxx','oyy','txy','omax','omin',
                                      'von_mises_top','von_mises_bottom',
                                      'nodal_vm','noxx','noyy','ntxy','nomax','nomin')
            and state.model is not None):
        e1d, e2d, e3d = state.model.elements_by_dim()
        if e1d and (e2d or e3d):
            bits.append("Frame stress not displayed")
    beam_note = _beam_mesh_note(state)
    if beam_note:
        bits.append(beam_note)
    return "  |  ".join(bits)


def _beam_mesh_note(state: ViewerState):
    model = getattr(state, 'model', None)
    if model is None:
        return ""
    e1d, _, _ = model.elements_by_dim()
    beam_eids = {eid for eid, elem in e1d.items() if elem.type in ('CBAR', 'CBEAM')}
    if not beam_eids:
        return ""

    load_sid = _subcase_load_sid(state) if state.load_filter_mode == "subcase" else state.active_load_case

    def _has_active_pload(seq):
        for pl in seq:
            try:
                eid = int(pl.get('eid', 0))
                sid = int(pl.get('sid', 0))
            except Exception:
                continue
            if eid not in beam_eids:
                continue
            if load_sid > 0 and sid != load_sid:
                continue
            return True
        return False

    if (_has_active_pload(getattr(model, 'pload1s', []))
            or _has_active_pload(getattr(model, 'pload2s', []))
            or _has_active_pload(getattr(model, 'pload4s', []))):
        return "For CBEAM/CBAR with PLOAD, run with sufficient mesh"
    return ""


def draw_status_strip(state: ViewerState, width: int, height: int):
    if state.model is None:
        return
    strip_x = 275
    strip_w = max(440, width - strip_x - 258)
    imgui.set_next_window_pos((strip_x, height-58), imgui.Cond_.always)
    imgui.set_next_window_size((strip_w, 54), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.78)
    flags=(imgui.WindowFlags_.no_title_bar|imgui.WindowFlags_.no_resize|
           imgui.WindowFlags_.no_move|imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##statusstrip", None, flags)
    title_s=(state.model.title if state.model and state.model.title else "")
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0,0.9,0.4,1.0))
    if title_s: imgui.text(title_s)
    imgui.pop_style_color()
    if state.deform_scale > 0 and state.show_deformed:
        def_str=f"Deformed({state.deform_scale:.2g}x)"
    else:
        def_str="Undeformed"
    label = ""
    if state.display_mode in ("beam", "beam_v2"):
        diag = getattr(state, 'beam_diagram', None)
        label_map = dict(getattr(diag, 'available_results', lambda: [])()) if diag is not None else {}
        base = label_map.get(getattr(state, 'beam_result_key', ''), "Beam Diagram")
        prefix = "Beam Diagram Viewer Approximation" if state.display_mode == "beam_v2" else "Beam Diagram Solver Reference"
        label = f"{prefix} | {base}"
    elif state.display_mode == "contour":
        label = _current_result_label(state) or "Contour"
    elif state.display_mode == "wireframe":
        label = "Wireframe"
    else:
        label = "Hidden"
    ctx = _status_context_label(state)
    hover = (getattr(state, 'hover_status', '') or '').strip()
    parts = [p for p in [ctx, def_str, label, hover] if p]
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.8,0.9,1.0,1.0))
    imgui.text("  |  ".join(parts))
    imgui.pop_style_color()
    imgui.end()


def _available_load_sids(state: ViewerState):
    if state.model is None:
        return []
    sids = set()
    sids.update(getattr(f, 'sid', 0) for f in getattr(state.model, 'forces', []))
    sids.update(getattr(m, 'sid', 0) for m in getattr(state.model, 'moments', []))
    sids.update(pl.get('sid', 0) for pl in getattr(state.model, 'pload1s', []))
    sids.update(pl.get('sid', 0) for pl in getattr(state.model, 'pload2s', []))
    sids.update(pl.get('sid', 0) for pl in getattr(state.model, 'pload4s', []))
    return sorted(s for s in sids if s > 0)


def _subcase_load_sid(state: ViewerState) -> int:
    if state.model is None:
        return 0
    return int(getattr(state.model, 'subcase_loads', {}).get(state.subcase, 0) or 0)


def _available_spc_sids(state: ViewerState):
    if state.model is None:
        return []
    return sorted({int(spc.id) for spc in getattr(state.model, 'spcs', []) if int(spc.id) > 0})


def _subcase_spc_sid(state: ViewerState) -> int:
    if state.model is None:
        return 0
    return int(getattr(state.model, 'subcase_spcs', {}).get(state.subcase, 0) or 0)


def _fmt_meta(val: float) -> str:
    try:
        x = float(val)
    except Exception:
        return str(val)
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.4f}".rstrip('0').rstrip('.')
    if s in ("-0", "+0", ""):
        s = "0"
    if len(s) <= 12:
        return s
    return f"{x:.4e}"


def _is_beam_stress_obj(es) -> bool:
    return getattr(es, 'elem_type', '').upper() in ('CBAR', 'CBEAM')


def _draw_model_tree(state: ViewerState):
    model = state.model
    if model is None:
        return
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 1.0, 0.7, 1.0))
    imgui.text("Library")
    imgui.pop_style_color()
    if imgui.tree_node(f"Materials ({len(model.materials)})##mats"):
        for mid in sorted(model.materials):
            mat = model.materials[mid]
            label = f"MAT {mid}  {mat.type}"
            selected = state.selected_material_id == mid and state.show_material_window
            if imgui.selectable(label, selected)[0]:
                state.selected_material_id = mid
                state.show_material_window = True
        imgui.tree_pop()
    if imgui.tree_node(f"Properties ({len(model.properties)})##props"):
        for pid in sorted(model.properties):
            prop = model.properties[pid]
            sec = getattr(prop, 'section', None)
            if prop.type in ('PBAR', 'PBEAM', 'PROD'):
                if prop.type == 'PROD':
                    a_txt = _fmt_meta(float(prop.params.get('f3', 0) or 0)) if 'f3' in prop.params else "?"
                    suffix = f"  A={a_txt}"
                else:
                    a_txt = _fmt_meta(float(prop.params.get('f3', 0) or 0)) if 'f3' in prop.params else "?"
                    suffix = f"  A={a_txt}"
            else:
                suffix = f"  {getattr(sec, 'label', '')}" if sec is not None else ""
            label = f"PID {pid}  {prop.type}{suffix}"
            selected = state.selected_property_id == pid and state.show_property_window
            if imgui.selectable(label, selected)[0]:
                state.selected_property_id = pid
                state.show_property_window = True
        imgui.tree_pop()


def draw_model_browser_windows(state: ViewerState, width: int, height: int):
    model = state.model
    if model is None:
        return
    if state.show_material_window and state.selected_material_id in model.materials:
        mat = model.materials[state.selected_material_id]
        imgui.set_next_window_pos((282, 72), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((320, 145), imgui.Cond_.first_use_ever)
        expanded, opened = imgui.begin(f"Material {mat.id}", True)
        state.show_material_window = bool(opened)
        if expanded:
            imgui.text(f"Type: {mat.type}")
            imgui.separator()
            imgui.text(f"E   = {_fmt_meta(mat.E)}")
            imgui.text(f"G   = {_fmt_meta(mat.G)}")
            imgui.text(f"nu  = {_fmt_meta(mat.nu)}")
            imgui.text(f"rho = {_fmt_meta(mat.rho)}")
            imgui.end()
        else:
            imgui.end()
    if state.show_property_window and state.selected_property_id in model.properties:
        prop = model.properties[state.selected_property_id]
        sec = getattr(prop, 'section', None)
        elem_types = sorted({elem.type for elem in model.elements.values() if int(getattr(elem, 'pid', 0)) == int(prop.id)})
        imgui.set_next_window_pos((282, 228), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((390, 220), imgui.Cond_.first_use_ever)
        expanded, opened = imgui.begin(f"Property {prop.id}", True)
        state.show_property_window = bool(opened)
        if expanded:
            imgui.text(f"Type: {prop.type}")
            imgui.text(f"MID : {prop.mid}")
            if elem_types:
                imgui.text(f"Formulation: {', '.join(elem_types)}")
            if sec is not None:
                imgui.separator()
                dims = list(getattr(sec, 'dims', []) or [])
                shp = str(getattr(sec, 'shape', '')).upper()
                if prop.type in ('PBAR', 'PBEAM', 'PROD'):
                    if prop.type == 'PROD':
                        if 'f3' in prop.params:
                            imgui.text(f"Area   : {_fmt_meta(float(prop.params.get('f3', 0) or 0))}")
                        if 'f4' in prop.params:
                            imgui.text(f"J      : {_fmt_meta(float(prop.params.get('f4', 0) or 0))}")
                    elif prop.type == 'PBAR':
                        if 'f3' in prop.params:
                            imgui.text(f"A      : {_fmt_meta(float(prop.params.get('f3', 0) or 0))}")
                        if 'f4' in prop.params:
                            imgui.text(f"I1     : {_fmt_meta(float(prop.params.get('f4', 0) or 0))}")
                        if 'f5' in prop.params:
                            imgui.text(f"I2     : {_fmt_meta(float(prop.params.get('f5', 0) or 0))}")
                        if 'f6' in prop.params:
                            imgui.text(f"J      : {_fmt_meta(float(prop.params.get('f6', 0) or 0))}")
                    elif prop.type == 'PBEAM':
                        if 'f3' in prop.params:
                            imgui.text(f"A      : {_fmt_meta(float(prop.params.get('f3', 0) or 0))}")
                        if 'f4' in prop.params:
                            imgui.text(f"I1     : {_fmt_meta(float(prop.params.get('f4', 0) or 0))}")
                        if 'f5' in prop.params:
                            imgui.text(f"I2     : {_fmt_meta(float(prop.params.get('f5', 0) or 0))}")
                        if 'f7' in prop.params:
                            imgui.text(f"J      : {_fmt_meta(float(prop.params.get('f7', 0) or 0))}")
                elif shp == 'I' and len(dims) >= 6:
                    imgui.text(f"Section: {getattr(sec, 'label', sec.shape)}")
                    imgui.text(f"Area   : {_fmt_meta(getattr(sec, 'area', 0.0))}")
                    H, bf1, bf2, tf1, tf2, tw = dims[:6]
                    imgui.text(f"H={_fmt_meta(H)}  bf1={_fmt_meta(bf1)}  bf2={_fmt_meta(bf2)}")
                    imgui.text(f"tf1={_fmt_meta(tf1)}  tf2={_fmt_meta(tf2)}  tw={_fmt_meta(tw)}")
                elif shp == 'RECT' and len(dims) >= 2:
                    imgui.text(f"Section: {getattr(sec, 'label', sec.shape)}")
                    imgui.text(f"Area   : {_fmt_meta(getattr(sec, 'area', 0.0))}")
                    imgui.text(f"b={_fmt_meta(dims[0])}  h={_fmt_meta(dims[1])}")
                elif shp == 'CIRCLE' and len(dims) >= 1:
                    imgui.text(f"Section: {getattr(sec, 'label', sec.shape)}")
                    imgui.text(f"Area   : {_fmt_meta(getattr(sec, 'area', 0.0))}")
                    imgui.text(f"d={_fmt_meta(dims[0])}")
                elif shp == 'PIPE' and len(dims) >= 2:
                    imgui.text(f"Section: {getattr(sec, 'label', sec.shape)}")
                    imgui.text(f"Area   : {_fmt_meta(getattr(sec, 'area', 0.0))}")
                    imgui.text(f"do={_fmt_meta(dims[0])}  di={_fmt_meta(dims[1])}")
                elif shp == 'BOX' and len(dims) >= 4:
                    imgui.text(f"Section: {getattr(sec, 'label', sec.shape)}")
                    imgui.text(f"Area   : {_fmt_meta(getattr(sec, 'area', 0.0))}")
                    imgui.text(f"b={_fmt_meta(dims[0])}  h={_fmt_meta(dims[1])}")
                    imgui.text(f"t1={_fmt_meta(dims[2])}  t2={_fmt_meta(dims[3])}")
            if prop.params:
                imgui.separator()
                if imgui.tree_node("Raw Params"):
                    for k, v in sorted(prop.params.items()):
                        imgui.text(f"{k}: {v}")
                    imgui.tree_pop()
            imgui.end()
        else:
            imgui.end()


def draw_menu_bar(state: ViewerState):
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            if imgui.menu_item("Open .dat...", "", False)[0]:
                state._dat_dialog = _open_dialog("Open .dat", ["*.dat *.nas *.bdf"])
            if imgui.menu_item("Open .f06...", "", False)[0]:
                state._f06_dialog = _open_dialog("Open .f06 / .op2", ["*.f06 *.op2 *.neu"])
            imgui.separator()
            if imgui.menu_item("Quit", "Esc", False)[0]:
                state.request_quit = True
            imgui.end_menu()
        if imgui.begin_menu("View"):
            if imgui.menu_item("Fit Model", "F", False)[0]: state.request_fit=True
            imgui.end_menu()
        imgui.end_main_menu_bar()
    if state._dat_dialog:
        r=_dialog_result(state._dat_dialog)
        if r is not None: state.dat_path=r; state.request_load_dat=True; state._dat_dialog=None
    if state._f06_dialog:
        r=_dialog_result(state._f06_dialog)
        if r is not None: state.f06_path=r; state.request_load_f06=True; state._f06_dialog=None


def draw_left_panel(state: ViewerState, height: int):
    imgui.set_next_window_pos((0,20), imgui.Cond_.always)
    imgui.set_next_window_size((270, max(200, height-20)), imgui.Cond_.always)
    flags=(imgui.WindowFlags_.no_title_bar|imgui.WindowFlags_.no_resize|
           imgui.WindowFlags_.no_move|imgui.WindowFlags_.no_scroll_with_mouse)
    imgui.begin("##left", None, flags)

    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5,0.85,1.0,1.0))
    imgui.text("MYSTRAN VIEWER"); imgui.pop_style_color()
    imgui.separator()

    imgui.text("DAT:")
    imgui.set_next_item_width(-40)
    _,state.dat_path=imgui.input_text("##dat",state.dat_path)
    imgui.same_line()
    if imgui.button("...##dat"): state._dat_dialog=_open_dialog("Open .dat",["*.dat *.nas *.bdf"])
    if imgui.button("Load .dat",(-1,0)): state.request_load_dat=True

    imgui.spacing()
    imgui.text("F06/OP2:")
    imgui.set_next_item_width(-40)
    _,state.f06_path=imgui.input_text("##f06",state.f06_path)
    imgui.same_line()
    if imgui.button("...##f06"): state._f06_dialog=_open_dialog("Open results",["*.f06 *.op2 *.neu"])
    if imgui.button("Load results",(-1,0)): state.request_load_f06=True

    imgui.separator()
    if state.model:
        m=state.model
        imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(1.0,0.85,0.3,1.0))
        imgui.text("Model"); imgui.pop_style_color()
        if m.title: imgui.text(f"  {m.title[:28]}")
        imgui.text(f"  Nodes:    {len(m.nodes)}")
        e1d,e2d,e3d=m.elements_by_dim()
        imgui.text(f"  1D elem:  {len(e1d)}")
        imgui.text(f"  2D elem:  {len(e2d)}")
        imgui.text(f"  3D elem:  {len(e3d)}")
        imgui.text(f"  Materials:{len(m.materials)}")
        imgui.text(f"  Properties:{len(m.properties)}")
        imgui.text(f"  SPCs:     {len(m.spcs)}")
        imgui.text(f"  Forces:   {len(m.forces)}")
        imgui.separator()
        _draw_model_tree(state)
        imgui.separator()
    if state.results:
        r=state.results
        imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(1.0,0.85,0.3,1.0))
        imgui.text("Results"); imgui.pop_style_color()
        imgui.text(f"  Subcases: {r.subcases}")
        if state.subcase in r.displacements:
            imgui.text(f"  Max |u|:  {r.max_displacement(state.subcase):.4e}")
        imgui.separator()

    if imgui.button("Fit View [F]",(-1,0)): state.request_fit=True
    has_model = state.model is not None
    has_results = state.results is not None and any(
        len(v) > 0 for v in state.results.displacements.values())
    if has_model:
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.15,0.45,0.15,1.0))
        if imgui.button("Export PNG##snap_left", (-1,0)):
            state.request_export_png = True
        if has_results:
            if imgui.button("Export MP4##anim_left", (-1,0)):
                state.request_export_mp4 = True
        imgui.pop_style_color()
        if has_results and getattr(state, 'exporting_mp4', False):
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1,0.8,0,1))
            pct = getattr(state, 'export_progress', 0)
            imgui.text(f"Exporting {pct:.0f}%...")
            imgui.pop_style_color()

    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,0.9,0.7,1.0))
    imgui.text("Visible Elements"); imgui.pop_style_color()
    rebuild=False
    for label,idx in [("1D (beam/rod)",0),("2D (shell)",1),("3D (solid)",2)]:
        ch,state.visible_dims[idx]=imgui.checkbox(label,state.visible_dims[idx])
        if ch: rebuild=True
    if rebuild: state.request_rebuild=True

    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.5,0.5,0.5,1.0))
    imgui.text(f"FPS: {state.fps:.1f}   {state.frame_time*1000:.1f}ms")
    imgui.pop_style_color()
    imgui.end()


def draw_right_panel(state: ViewerState, camera, width: int, height: int):
    pw=255
    imgui.set_next_window_pos((width-pw,20),imgui.Cond_.always)
    imgui.set_next_window_size((pw,height-20),imgui.Cond_.always)
    flags=(imgui.WindowFlags_.no_title_bar|imgui.WindowFlags_.no_resize|imgui.WindowFlags_.no_move)
    imgui.begin("##right",None,flags)

    # View presets
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.5,0.85,1.0,1.0))
    imgui.text("View"); imgui.pop_style_color()
    imgui.push_style_var(imgui.StyleVar_.frame_padding,imgui.ImVec2(2,2))
    item_gap = float(imgui.get_style().item_spacing.x)
    avail_w = float(imgui.get_content_region_avail().x)
    bw = max(40.0, (avail_w - item_gap) * 0.5)
    iso_gap = item_gap
    iso_w = max(34.0, (bw - iso_gap) * 0.5)
    for i,(label,yaw,pitch,tip) in enumerate(_VIEW_PRESETS[:2]):
        if i%2==1: imgui.same_line()
        if imgui.button(label,(bw,18)):
            camera._yaw=float(yaw); camera._pitch=float(pitch)
            camera.set_up_axis('z')
        if imgui.is_item_hovered(): imgui.set_tooltip(tip)
    if imgui.button("X-Y",(bw,18)):
        camera._yaw = 0.0
        camera._pitch = 0.0
        camera.set_up_axis('y')
    if imgui.is_item_hovered():
        imgui.set_tooltip("Top: X right, Y up")
    imgui.same_line()
    if imgui.button("ISO Z",(iso_w,18)):
        camera._yaw = 45.0
        camera._pitch = 25.0
        camera.set_up_axis('z')
    if imgui.is_item_hovered():
        imgui.set_tooltip("Isometric, Z-up oriented")
    imgui.same_line(0.0, float(iso_gap))
    if imgui.button("ISO Y",(iso_w,18)):
        camera._yaw = 45.0
        camera._pitch = -25.0
        camera.set_up_axis('y')
    if imgui.is_item_hovered():
        imgui.set_tooltip("Isometric, Y-up oriented")
    imgui.pop_style_var()
    imgui.spacing()

    # Display mode
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.5,0.85,1.0,1.0))
    imgui.text("Display Mode"); imgui.pop_style_color()
    imgui.separator()
    rebuild=False
    if imgui.radio_button("Wireframe [W]", state.display_mode=="wireframe"):
        if state.display_mode!="wireframe": state.display_mode="wireframe"; rebuild=True
    imgui.same_line()
    if imgui.radio_button("Orthographic", state.ortho_mode):
        state.ortho_mode = True
    if imgui.radio_button("Hidden    [S]", state.display_mode=="solid"):
        if state.display_mode!="solid": state.display_mode="solid"; rebuild=True
    imgui.same_line()
    if imgui.radio_button("Perspective", not state.ortho_mode):
        state.ortho_mode = False
    if imgui.radio_button("Contour   [C]", state.display_mode=="contour"):
        if state.display_mode!="contour": state.display_mode="contour"; rebuild=True
    if state.deform_scale > 0:
        imgui.same_line()
        if imgui.radio_button("Undeformed", not state.show_deformed):
            if state.show_deformed:
                state.show_deformed = False
                rebuild = True
    beam_ready = bool(getattr(state, 'beam_diagram', None) and getattr(state.beam_diagram, 'beam_data', {}))
    if beam_ready:
        if imgui.radio_button("Beam Diag.", state.display_mode=="beam"):
            if state.display_mode!="beam":
                state.display_mode="beam"; state.beam_scale = 0.0; rebuild=True; state.request_fit = True; state.request_beam_rebuild = True
        if state.deform_scale > 0:
            imgui.same_line()
            imgui.dummy((14, 0))
            imgui.same_line()
            if imgui.radio_button("Deformed", state.show_deformed):
                if not state.show_deformed:
                    state.show_deformed = True
                    rebuild = True
        if imgui.radio_button("Beam Diag. v2", state.display_mode=="beam_v2"):
            if state.display_mode!="beam_v2":
                state.display_mode="beam_v2"; state.beam_scale = 0.0; rebuild=True; state.request_fit = True; state.request_beam_rebuild = True
    elif state.deform_scale > 0:
        if imgui.radio_button("Deformed", state.show_deformed):
            if not state.show_deformed:
                state.show_deformed = True
                rebuild = True
    imgui.spacing()

    if state.results:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7,1.0,0.7,1.0))
        imgui.text("Result Set"); imgui.pop_style_color()
        imgui.text("Case")
        imgui.set_next_item_width(-1)
        case_items = ["Case 1"]
        imgui.combo("##case_right", 0, case_items)
        imgui.text("Subcase")
        sc_list = [_subcase_display_label(state, s) for s in state.results.subcases]
        cur_sc = state.results.subcases.index(state.subcase) if state.subcase in state.results.subcases else 0
        imgui.set_next_item_width(-1)
        ch_sc, idx_sc = imgui.combo("##sc_right", cur_sc, sc_list)
        if ch_sc:
            state.subcase = state.results.subcases[idx_sc]
            rebuild = True
        imgui.spacing()

    if state.display_mode=="contour":
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,1.0,0.7,1.0))
        imgui.text("Result"); imgui.pop_style_color()
        stress_types = ['von_mises','oxx','oyy','txy','omax','omin','sxc','sxd','sxe','sxf','smax','smin','stress3d']
        disp_types   = ['displacement','t1','t2','t3']
        cur_is_stress = state.result_type in stress_types
        cur_is_disp   = state.result_type in disp_types or not cur_is_stress
        if imgui.radio_button("Displacement", cur_is_disp):
            if not cur_is_disp: state.result_type='displacement'; rebuild=True
        if imgui.radio_button("Stress/Strain", cur_is_stress):
            if not cur_is_stress: state.result_type='von_mises'; rebuild=True
        cmaps=list(COLORMAPS.keys())
        cur=list(cmaps).index(state.cmap_name) if state.cmap_name in cmaps else 0
        imgui.set_next_item_width(-1)
        ch,idx=imgui.combo("##cmap",cur,list(cmaps))
        if ch: state.cmap_name=list(cmaps)[idx]; rebuild=True
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,1.0,0.7,1.0))
        imgui.text("Viewer Nodal Averaging"); imgui.pop_style_color()
        scopes = ["property", "material", "all"]
        scope_labels = ["Same property only", "Same material only", "Compatible all"]
        cur_scope = scopes.index(state.averaging_scope) if state.averaging_scope in scopes else 0
        imgui.set_next_item_width(-1)
        ch, idx = imgui.combo("##avg_scope", cur_scope, scope_labels)
        if ch:
            state.averaging_scope = scopes[idx]
            rebuild = True
        ch, state.average_same_family = imgui.checkbox("No family mixing", state.average_same_family)
        if ch: rebuild = True
        ch, state.average_transform_shell = imgui.checkbox("Transform shell frame", state.average_transform_shell)
        if ch: rebuild = True
        imgui.text("Angle betw.")
        imgui.same_line()
        imgui.set_next_item_width(120)
        ch, ang = imgui.slider_float("##avg_ang", state.average_angle_deg, 0.0, 90.0, "%.0f deg")
        if ch:
            state.average_angle_deg = ang
            rebuild = True

    imgui.spacing(); imgui.separator()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,1.0,0.7,1.0))
    imgui.text("Deformation Scale"); imgui.pop_style_color()
    imgui.set_next_item_width(-1)
    is_eigen = state.results is not None and state.subcase in getattr(state, 'eigen_data', {})
    if is_eigen:
        def_max = 1.0
        def_fmt = "%.3f x"
    else:
        def_max = max(200.0, float(state.deform_scale) * 1.25 if state.deform_scale > 0 else 200.0)
        def_fmt = "%.1f x" if def_max < 1.0e4 else "%.3g x"
    ch,nd=imgui.slider_float("##def",state.deform_scale,0.0,def_max,def_fmt)
    if ch: state.deform_scale=nd; rebuild=True

    imgui.spacing(); imgui.separator()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,1.0,0.7,1.0))
    imgui.text("Overlay"); imgui.pop_style_color()
    _,state.show_edges=imgui.checkbox("Show Edges",state.show_edges)
    _,state.show_spc  =imgui.checkbox("Show BCs (SPCs)",state.show_spc)

    if state.display_mode in ("beam", "beam_v2"):
        imgui.spacing(); imgui.separator()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.3,0.9,1.0,1.0))
        imgui.text("Beam Diagram"); imgui.pop_style_color()
        diag = getattr(state, 'beam_diagram', None)
        if diag is not None and getattr(diag, 'beam_data', {}):
            results = diag.available_results()
            keys   = [r[0] for r in results]
            labels = [r[1] for r in results]
            cur = keys.index(state.beam_result_key) if state.beam_result_key in keys else 0
            imgui.set_next_item_width(-1)
            ch, idx = imgui.combo("##bres_inline", cur, labels)
            if ch:
                state.beam_result_key = keys[idx]
                state.beam_scale = 0.0
                state.request_beam_rebuild = True
                state.request_fit = True
            imgui.text("Diagram scale")
            imgui.set_next_item_width(-1)
            ch2, sv = imgui.slider_float("##bscale_inline", state.beam_scale, 0.0, 1.0, "%.2fx")
            if ch2:
                state.beam_scale = sv
                state.request_beam_rebuild = True
            mv = diag.max_value(state.beam_result_key)
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.8,1.0,0.6,1.0))
            imgui.text(f"Max: {_format_legend_tick(mv)}")
            imgui.pop_style_color()
            if not getattr(diag, 'beam_stress_available', False):
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0,0.78,0.42,1.0))
                imgui.text_wrapped("Beam stress not present in solver output")
                imgui.pop_style_color()
        else:
            imgui.text("No beam OP2 loaded")

    imgui.spacing(); imgui.separator()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.7,1.0,0.7,1.0))
    imgui.text("Notation (when idle)"); imgui.pop_style_color()
    _,state.show_nodes       = imgui.checkbox("Nodes (dots)",       state.show_nodes)
    _,state.show_node_nums   = imgui.checkbox("Node numbers",       state.show_node_nums)
    _,state.show_elem_nums   = imgui.checkbox("Element numbers",    state.show_elem_nums)
    if state.display_mode == "contour":
        _,state.show_node_values = imgui.checkbox("Node result values", state.show_node_values)
    if state.display_mode in ("contour", "beam", "beam_v2"):
        _,state.show_elem_values = imgui.checkbox("Element result values", state.show_elem_values)
    _,state.show_reaction_forces = imgui.checkbox("Reaction TX/TY/TZ", state.show_reaction_forces)
    _,state.show_reaction_moments = imgui.checkbox("Reaction RX/RY/RZ", state.show_reaction_moments)
    if state.show_reaction_forces or state.show_reaction_moments:
        _,state.reaction_engineering = imgui.checkbox("Engineering values", state.reaction_engineering)
    _,state.show_constraints = imgui.checkbox("Constraints + dirs", state.show_constraints)
    if state.show_beam_panel and state.display_mode == "beam":
        _,state.show_station_nodes = imgui.checkbox("Beam station dots", state.show_station_nodes)

    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.5,0.85,1.0,1.0))
    imgui.text("Local Axes"); imgui.pop_style_color()
    _,state.show_local_axes = imgui.checkbox("Show local axes", state.show_local_axes)
    if state.show_local_axes:
        imgui.indent(10)
        _,state.local_axis_frame = imgui.checkbox("Frame##ax", state.local_axis_frame)
        imgui.same_line()
        _,state.local_axis_shell = imgui.checkbox("Shell##ax", state.local_axis_shell)
        imgui.same_line()
        _,state.local_axis_solid = imgui.checkbox("Solid##ax", state.local_axis_solid)
        imgui.unindent(10)

    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(0.5,0.85,1.0,1.0))
    imgui.text("Loads"); imgui.pop_style_color()
    _,state.show_forces   = imgui.checkbox("Point forces",  state.show_forces)
    _,state.show_moments  = imgui.checkbox("Moments",       state.show_moments)
    _,state.show_pressure = imgui.checkbox("Element loads", state.show_pressure)
    if any([state.show_forces, state.show_moments, state.show_pressure]):
        load_sids = _available_load_sids(state)
        subcase_sid = _subcase_load_sid(state)
        imgui.text("Load set")
        if subcase_sid > 0 and state.results is not None:
            state.load_filter_mode = "subcase"
            state.active_load_case = subcase_sid
            imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.18,0.22,0.18,1.0))
            imgui.set_next_item_width(-1)
            imgui.combo("##lc_subcase_only", 0, [f"Load {subcase_sid} (from subcase)"])
            imgui.pop_style_color()
        else:
            items = ["All loads"] + [f"Load {sid}" for sid in load_sids]
            cur = 0
            if state.load_filter_mode == "manual" and state.active_load_case > 0 and state.active_load_case in load_sids:
                cur = load_sids.index(state.active_load_case) + 1
            imgui.set_next_item_width(-1)
            ch_lc, idx_lc = imgui.combo("##lc", cur, items)
            if ch_lc:
                if idx_lc == 0:
                    state.load_filter_mode = "all"
                    state.active_load_case = 0
                else:
                    state.load_filter_mode = "manual"
                    state.active_load_case = load_sids[idx_lc - 1]

    if any([state.show_spc, state.show_constraints, state.show_reaction_forces, state.show_reaction_moments]):
        spc_sids = _available_spc_sids(state)
        subcase_spc = _subcase_spc_sid(state)
        imgui.text("BC set")
        if subcase_spc > 0 and state.results is not None:
            state.spc_filter_mode = "subcase"
            state.active_spc_case = subcase_spc
            imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.18,0.22,0.18,1.0))
            imgui.set_next_item_width(-1)
            imgui.combo("##spc_subcase_only", 0, [f"BC {subcase_spc} (from subcase)"])
            imgui.pop_style_color()
        elif spc_sids:
            items = ["All BCs"] + [f"BC {sid}" for sid in spc_sids]
            cur = 0
            if state.spc_filter_mode == "manual" and state.active_spc_case > 0 and state.active_spc_case in spc_sids:
                cur = spc_sids.index(state.active_spc_case) + 1
            imgui.set_next_item_width(-1)
            ch_spc, idx_spc = imgui.combo("##spc", cur, items)
            if ch_spc:
                if idx_spc == 0:
                    state.spc_filter_mode = "all"
                    state.active_spc_case = 0
                else:
                    state.spc_filter_mode = "manual"
                    state.active_spc_case = spc_sids[idx_spc - 1]

    if rebuild: state.request_rebuild=True
    imgui.end()


def _draw_maxmin_markers(state, dl, bar_x, bar_y, bar_h, bar_w):
    """Draw MAX/MIN text below colorbar with location info."""
    if state.results is None or state.model is None: return
    import numpy as np
    sc = state.subcase; r = state.results; rt = state.result_type
    max_val = min_val = None
    max_loc = min_loc = ""

    def _eloc(eid):
        elem = state.model.elements.get(eid); s = f"Elem {eid}"
        if elem:
            ns = [state.model.nodes.get(n) for n in elem.nodes[:3] if state.model.nodes.get(n)]
            if ns:
                c = np.mean([n.xyz for n in ns], axis=0)
                s += f"  ({c[0]:.3g},{c[1]:.3g},{c[2]:.3g})"
        return s

    def _nloc(nid):
        n = state.model.nodes.get(nid); s = f"Node {nid}"
        if n: s += f"  ({n.xyz[0]:.3g},{n.xyz[1]:.3g},{n.xyz[2]:.3g})"
        return s

    if rt in ('displacement','t1','t2','t3') and sc in r.displacements:
        disp = r.displacements[sc]
        if rt == 'displacement':
            items = {nid: float(np.linalg.norm(d.translation)) for nid,d in disp.items()}
        else:
            comp = {'t1':0,'t2':1,'t3':2}[rt]
            items = {nid: float(d.translation[comp]) for nid,d in disp.items()}
        if items:
            max_nid = max(items, key=items.get); min_nid = min(items, key=items.get)
            max_val = items[max_nid]; min_val = items[min_nid]
            max_loc = _nloc(max_nid); min_loc = _nloc(min_nid)
            if rt == 'displacement':
                dmax = disp.get(max_nid)
                dmin = disp.get(min_nid)
                if dmax is not None:
                    max_loc += f"  TX={_format_legend_tick(float(dmax.t1))} TY={_format_legend_tick(float(dmax.t2))} TZ={_format_legend_tick(float(dmax.t3))}"
                if dmin is not None:
                    min_loc += f"  TX={_format_legend_tick(float(dmin.t1))} TY={_format_legend_tick(float(dmin.t2))} TZ={_format_legend_tick(float(dmin.t3))}"

    elif rt in ('von_mises','oxx','oyy','txy','omax','omin',
                'von_mises_top','von_mises_bottom') and sc in r.stresses:
        st = r.stresses[sc]
        def _gv(es):
            return es.von_mises if rt=='von_mises' else es.values.get(rt, es.von_mises)
        elem_st = {k:v for k,v in st.items()
                   if k not in ('_nodal_avg', '_nodal_avg_components', '_nodal_acc',
                                '_solver_nodal_avg', '_solver_nodal_avg_components',
                                '_derived_nodal_avg', '_derived_nodal_avg_components',
                                '_shell_corner_contribs')
                   and hasattr(v,'values') and hasattr(v, 'von_mises')
                   and not _is_beam_stress_obj(v)}
        if elem_st:
            max_eid = max(elem_st, key=lambda k: _gv(elem_st[k]))
            min_eid = min(elem_st, key=lambda k: _gv(elem_st[k]))
            max_val = _gv(elem_st[max_eid]); min_val = _gv(elem_st[min_eid])
            max_loc = _eloc(max_eid); min_loc = _eloc(min_eid)

    elif rt in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin') and sc in r.stresses:
        nav = state.nodal_stress_components()
        key = _ELEMENT_STRESS_MAP.get(rt, 'von_mises')
        if isinstance(nav,dict) and nav:
            max_nid = max(nav, key=lambda k: nav[k].get(key, 0.0))
            min_nid = min(nav, key=lambda k: nav[k].get(key, 0.0))
            max_val = nav[max_nid].get(key, 0.0); min_val = nav[min_nid].get(key, 0.0)
            max_loc = _nloc(max_nid); min_loc = _nloc(min_nid)

    elif hasattr(r,'forces') and sc in r.forces:
        _inv = {'nfx':'fx','nfy':'fy','nfxy':'fxy','nmx':'mx','nmy':'my',
                'nmxy':'mxy','nqx':'qx','nqy':'qy'}
        base = _inv.get(rt, rt)
        is_nodal = rt in _inv
        fc_raw = r.forces[sc]
        if is_nodal:
            nav = state.nodal_force_components()
            if isinstance(nav,dict) and nav:
                max_nid = max(nav, key=lambda k: nav[k].get(base,0))
                min_nid = min(nav, key=lambda k: nav[k].get(base,0))
                max_val = nav[max_nid].get(base,0); min_val = nav[min_nid].get(base,0)
                max_loc = _nloc(max_nid); min_loc = _nloc(min_nid)
        else:
            fc = {k:v for k,v in fc_raw.items()
                  if k not in ('_nodal_avg', '_solver_nodal_avg', '_derived_nodal_avg')
                  and hasattr(v,'values')}
            if fc:
                max_eid = max(fc, key=lambda k: fc[k].values.get(base,0))
                min_eid = min(fc, key=lambda k: fc[k].values.get(base,0))
                max_val = fc[max_eid].values.get(base,0)
                min_val = fc[min_eid].values.get(base,0)
                max_loc = _eloc(max_eid); min_loc = _eloc(min_eid)

    if max_val is None: return

    text_y = bar_y + bar_h + 20
    dl.add_text((bar_x, text_y),      0xFF4499FF, f"MAX {_format_legend_tick(max_val)}")
    dl.add_text((bar_x, text_y + 14), 0xFF88AAFF, f"  {max_loc}")
    dl.add_text((bar_x, text_y + 30), 0xFF9944FF, f"MIN {_format_legend_tick(min_val)}")
    dl.add_text((bar_x, text_y + 44), 0xFFAABBFF, f"  {min_loc}")


def draw_legend(state: ViewerState, width: int, height: int):
    """Colorbar drawn directly on foreground draw list - no window clipping."""
    if state.display_mode not in ("contour", "beam", "beam_v2") or state.results is None: return
    import numpy as np
    from renderer.contour import COLORMAPS

    sc = state.subcase; r = state.results; rt = state.result_type
    cmap = COLORMAPS[state.cmap_name]

    if state.display_mode in ("beam", "beam_v2"):
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'diagram_data', lambda: getattr(diag, 'beam_data', {}))() if diag is not None else {}
        if not beam_data:
            return
        label_map = dict(getattr(diag, 'available_results', lambda: [])())
        key = state.beam_result_key
        vals = []
        max_info = None
        min_info = None
        for eid, bd in beam_data.items():
            if not bd.stations:
                continue
            bvals = [float(getattr(st, key, 0.0)) for st in bd.stations]
            vals.extend(bvals)
            local_max = max(range(len(bvals)), key=lambda i: bvals[i])
            local_min = min(range(len(bvals)), key=lambda i: bvals[i])
            if max_info is None or bvals[local_max] > max_info[2]:
                max_info = (eid, bd.stations[local_max].sd, bvals[local_max])
            if min_info is None or bvals[local_min] < min_info[2]:
                min_info = (eid, bd.stations[local_min].sd, bvals[local_min])
        if not vals or max_info is None or min_info is None:
            return
        dl = imgui.get_foreground_draw_list()
        bar_x = 276
        bar_y = 200
        beam_lbl = label_map.get(key, key)
        beam_lbl = (f"Beam Diagram Viewer Approximation | {beam_lbl}"
                    if state.display_mode == "beam_v2"
                    else f"Beam Diagram Solver Reference | {beam_lbl}")
        dl.add_text((bar_x, bar_y - 28), 0xFFFFCC44, beam_lbl)
        dl.add_text((bar_x, bar_y + 0),  0xFF4499FF, f"MAX {_format_legend_tick(max_info[2])}")
        dl.add_text((bar_x, bar_y + 14), 0xFF88AAFF, f"  Beam {max_info[0]}  s={max_info[1]:.3f}")
        dl.add_text((bar_x, bar_y + 30), 0xFF9944FF, f"MIN {_format_legend_tick(min_info[2])}")
        dl.add_text((bar_x, bar_y + 44), 0xFFAABBFF, f"  Beam {min_info[0]}  s={min_info[1]:.3f}")
        return

    if state.display_mode == "contour" and rt == 'stress3d':
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'beam_data', {}) if diag is not None else {}
        vals = []
        max_info = None
        min_info = None
        max_corner = None
        min_corner = None
        for eid, bd in beam_data.items():
            if not getattr(bd, 'stations', None):
                continue
            for st in bd.stations:
                cdef_vals = {
                    'C': float(getattr(st, 'sxc', 0.0)),
                    'D': float(getattr(st, 'sxd', 0.0)),
                    'E': float(getattr(st, 'sxe', 0.0)),
                    'F': float(getattr(st, 'sxf', 0.0)),
                }
                vals.extend(cdef_vals.values())
                cmax = max(cdef_vals, key=cdef_vals.get)
                cmin = min(cdef_vals, key=cdef_vals.get)
                if max_info is None or cdef_vals[cmax] > max_info[2]:
                    max_info = (eid, st.sd, cdef_vals[cmax])
                    max_corner = cmax
                if min_info is None or cdef_vals[cmin] < min_info[2]:
                    min_info = (eid, st.sd, cdef_vals[cmin])
                    min_corner = cmin
        if vals and max_info is not None and min_info is not None:
            vmin = min(vals)
            vmax = max(vals)
            label = 'Beam Stress 3D [CDEF interp]'
            dl = imgui.get_foreground_draw_list()
            bar_w = 16
            bar_h = min(360, height - 220)
            bar_x = 276
            bar_y = 200
            dl.add_text((bar_x, bar_y - 28), 0xFFFFCC44, label)

            def to_col(c):
                return (0xFF000000 | (int(np.clip(c[2], 0, 1) * 255) << 16) |
                        (int(np.clip(c[1], 0, 1) * 255) << 8) |
                        int(np.clip(c[0], 0, 1) * 255))

            n_seg = 64
            for i in range(n_seg):
                t0 = 1.0 - i / n_seg
                t1 = 1.0 - (i + 1) / n_seg
                c0 = to_col(cmap[int(np.clip(t0, 0, 1) * 255)])
                c1 = to_col(cmap[int(np.clip(t1, 0, 1) * 255)])
                seg_h = bar_h / n_seg
                dl.add_rect_filled_multi_color(
                    (bar_x, bar_y + i * seg_h), (bar_x + bar_w, bar_y + (i + 1) * seg_h),
                    c0, c0, c1, c1)
            dl.add_rect((bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 0xFF444444, 0, 1.0)

            lx = bar_x + bar_w + 6
            tick_count = 20 if bar_h >= 320 else 10
            for tick in legend_ticks(vmin, vmax, tick_count):
                t = np.clip((tick - vmin) / (vmax - vmin + 1e-30), 0, 1)
                ty = bar_y + (1.0 - t) * bar_h
                dl.add_line((bar_x + bar_w, ty), (bar_x + bar_w + 4, ty), 0xFF666666, 1.0)
                dl.add_text((lx, ty - 6), 0xFFBBBBBB, _format_legend_tick(tick))

            base_y = bar_y + bar_h + 18
            dl.add_text((bar_x, base_y + 0),  0xFF4499FF, f"MAX {_format_legend_tick(max_info[2])}")
            dl.add_text((bar_x, base_y + 14), 0xFF88AAFF,
                        f"  Beam {max_info[0]}  s={max_info[1]:.3f}  corner {max_corner}")
            dl.add_text((bar_x, base_y + 30), 0xFF9944FF, f"MIN {_format_legend_tick(min_info[2])}")
            dl.add_text((bar_x, base_y + 44), 0xFFAABBFF,
                        f"  Beam {min_info[0]}  s={min_info[1]:.3f}  corner {min_corner}")
            return

    if state.display_mode == "contour" and rt in _BEAM_STRESS_KEYS:
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'beam_data', {}) if diag is not None else {}
        vals = []
        max_info = None
        min_info = None
        max_corner = None
        min_corner = None
        for eid, bd in beam_data.items():
            if not getattr(bd, 'stations', None):
                continue
            bvals = [float(getattr(st, rt, 0.0)) for st in bd.stations]
            if not bvals:
                continue
            vals.extend(bvals)
            local_max = max(range(len(bvals)), key=lambda i: bvals[i])
            local_min = min(range(len(bvals)), key=lambda i: bvals[i])
            if max_info is None or bvals[local_max] > max_info[2]:
                max_info = (eid, bd.stations[local_max].sd, bvals[local_max])
            if min_info is None or bvals[local_min] < min_info[2]:
                min_info = (eid, bd.stations[local_min].sd, bvals[local_min])
        if vals and max_info is not None and min_info is not None:
            vmin = min(vals)
            vmax = max(vals)
            label = {
                'sxc': 'Beam Stress C [Solver]',
                'sxd': 'Beam Stress D [Solver]',
                'sxe': 'Beam Stress E [Solver]',
                'sxf': 'Beam Stress F [Solver]',
                'smax': 'Beam Stress Smax [Solver]',
                'smin': 'Beam Stress Smin [Solver]',
            }[rt]
            dl = imgui.get_foreground_draw_list()
            bar_w = 16
            bar_h = min(360, height - 220)
            bar_x = 276
            bar_y = 200
            dl.add_text((bar_x, bar_y - 28), 0xFFFFCC44, label)

            def to_col(c):
                return (0xFF000000 | (int(np.clip(c[2], 0, 1) * 255) << 16) |
                        (int(np.clip(c[1], 0, 1) * 255) << 8) |
                        int(np.clip(c[0], 0, 1) * 255))

            n_seg = 64
            for i in range(n_seg):
                t0 = 1.0 - i / n_seg
                t1 = 1.0 - (i + 1) / n_seg
                c0 = to_col(cmap[int(np.clip(t0, 0, 1) * 255)])
                c1 = to_col(cmap[int(np.clip(t1, 0, 1) * 255)])
                seg_h = bar_h / n_seg
                dl.add_rect_filled_multi_color(
                    (bar_x, bar_y + i * seg_h), (bar_x + bar_w, bar_y + (i + 1) * seg_h),
                    c0, c0, c1, c1)
            dl.add_rect((bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 0xFF444444, 0, 1.0)

            lx = bar_x + bar_w + 6
            tick_count = 20 if bar_h >= 320 else 10
            for tick in legend_ticks(vmin, vmax, tick_count):
                t = np.clip((tick - vmin) / (vmax - vmin + 1e-30), 0, 1)
                ty = bar_y + (1.0 - t) * bar_h
                dl.add_line((bar_x + bar_w, ty), (bar_x + bar_w + 4, ty), 0xFF666666, 1.0)
                dl.add_text((lx, ty - 6), 0xFFBBBBBB, _format_legend_tick(tick))

            base_y = bar_y + bar_h + 18
            dl.add_text((bar_x, base_y + 0),  0xFF4499FF, f"MAX {_format_legend_tick(max_info[2])}")
            dl.add_text((bar_x, base_y + 14), 0xFF88AAFF, f"  Beam {max_info[0]}  s={max_info[1]:.3f}")
            dl.add_text((bar_x, base_y + 30), 0xFF9944FF, f"MIN {_format_legend_tick(min_info[2])}")
            dl.add_text((bar_x, base_y + 44), 0xFFAABBFF, f"  Beam {min_info[0]}  s={min_info[1]:.3f}")
            return

    # Compute value range
    if rt == 'displacement' and sc in r.displacements:
        vmin, vmax = 0.0, r.max_displacement(sc); label = "Total Translation"
    elif rt in ('t1','t2','t3') and sc in r.displacements:
        comp = {'t1':0,'t2':1,'t3':2}[rt]
        vals = [float(d.translation[comp]) for d in r.displacements[sc].values()]
        vmin,vmax = (min(vals),max(vals)) if vals else (0,1)
        label = {'t1':'T1 (X)','t2':'T2 (Y)','t3':'T3 (Z)'}[rt]
    elif rt in ('von_mises','oxx','oyy','txy','omax','omin',
                'von_mises_top','von_mises_bottom') and sc in r.stresses:
        st = r.stresses[sc]
        def _gv(es):
            if rt=='von_mises': return es.von_mises
            return es.values.get(rt, es.von_mises)
        vals = [_gv(v) for k,v in st.items()
                if k not in ('_nodal_avg', '_nodal_avg_components', '_nodal_acc',
                             '_solver_nodal_avg', '_solver_nodal_avg_components',
                             '_derived_nodal_avg', '_derived_nodal_avg_components',
                             '_shell_corner_contribs')
                and hasattr(v,'values') and hasattr(v, 'von_mises')
                and not _is_beam_stress_obj(v)]
        vmin,vmax = (min(vals),max(vals)) if vals else (0,1)
        label = {'von_mises':'Von Mises [Elem]','oxx':'Sxx [Elem]','oyy':'Syy [Elem]','txy':'Sxy [Elem]',
                 'omax':'S1 [Elem]','omin':'S3 [Elem]',
                 'von_mises_top':'Von Mises Top [Elem]',
                 'von_mises_bottom':'Von Mises Bottom [Elem]'}.get(rt,'Stress [Elem]')
    elif rt in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin') and sc in r.stresses:
        nav = state.nodal_stress_components()
        if not nav: return
        base_rt = _ELEMENT_STRESS_MAP.get(rt, 'von_mises')
        vals = [v.get(base_rt, 0.0) for v in nav.values()]
        vmin,vmax = min(vals),max(vals)
        label = {
            'nodal_vm':'Von Mises',
            'noxx':'Sxx',
            'noyy':'Syy',
            'ntxy':'Sxy',
            'nomax':'S1',
            'nomin':'S3',
        }[rt] + _nodal_source_title(state, 'stress')
    elif rt in ('fx','fy','fxy','mx','my','mxy','qx','qy','nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy'):
        if not hasattr(r,'forces') or sc not in r.forces or not r.forces[sc]: return
        fc = r.forces[sc]
        inv_map = {'nfx':'fx','nfy':'fy','nfxy':'fxy','nmx':'mx','nmy':'my','nmxy':'mxy','nqx':'qx','nqy':'qy'}
        base_rt = inv_map.get(rt, rt)
        is_nod = rt in inv_map
        real_fc = {k:v for k,v in fc.items()
                   if k not in ('_nodal_avg', '_solver_nodal_avg', '_derived_nodal_avg')
                   and hasattr(v,'values')}
        if is_nod:
            nav = state.nodal_force_components()
            svals = [v.get(base_rt,0) for v in nav.values()] if isinstance(nav,dict) else []
        else:
            svals = [v.values.get(base_rt,0) for v in real_fc.values()]
        if not svals: return
        vmin,vmax = min(svals),max(svals)
        suffix = _nodal_source_title(state, 'force') if is_nod else ' [Elem]'
        label = {'fx':'FX','fy':'FY','fxy':'FXY','mx':'MX','my':'MY',
                 'mxy':'MXY','qx':'QX','qy':'QY'}.get(base_rt, base_rt.upper()) + suffix
    else:
        return

    dl = imgui.get_foreground_draw_list()

    # ── Layout ────────────────────────────────────────────────────
    # Toolbar: y=22 h=28 -> bottom y=50. Left panel: x=0..270.
    # Bar starts just right of left panel, below toolbar.
    bar_w  = 16
    bar_h  = min(360, height - 220)
    bar_x  = 276   # 6px right of left panel
    bar_y  = 200   # well below toolbar and any overlap

    # ── Draw ──────────────────────────────────────────────────────
    # Title above bar
    dl.add_text((bar_x, bar_y - 28), 0xFFFFCC44, label)

    def to_col(c):
        return (0xFF000000|(int(np.clip(c[2],0,1)*255)<<16)|
                (int(np.clip(c[1],0,1)*255)<<8)|int(np.clip(c[0],0,1)*255))

    # Gradient
    n_seg = 64
    for i in range(n_seg):
        t0=1.0-i/n_seg; t1=1.0-(i+1)/n_seg
        c0=to_col(cmap[int(np.clip(t0,0,1)*255)])
        c1=to_col(cmap[int(np.clip(t1,0,1)*255)])
        seg_h = bar_h/n_seg
        dl.add_rect_filled_multi_color(
            (bar_x, bar_y+i*seg_h),(bar_x+bar_w, bar_y+(i+1)*seg_h),
            c0,c0,c1,c1)
    dl.add_rect((bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h), 0xFF444444, 0, 1.0)

    # Tick marks + labels RIGHT of bar
    # Use a denser legend for contour reading.
    lx = bar_x + bar_w + 6
    tick_count = 20 if bar_h >= 320 else 10
    for tick in legend_ticks(vmin, vmax, tick_count):
        t  = np.clip((tick-vmin)/(vmax-vmin+1e-30), 0, 1)
        ty = bar_y + (1.0-t)*bar_h
        dl.add_line((bar_x+bar_w, ty),(bar_x+bar_w+4, ty), 0xFF666666, 1.0)
        dl.add_text((lx, ty-6), 0xFFBBBBBB, _format_legend_tick(tick))

    # Title BELOW the bar (avoids all overlap with tick labels)

    # Max/Min location (2 lines below title)
    _draw_maxmin_markers(state, dl, bar_x, bar_y, bar_h, bar_w)
    if (state.model is not None
            and rt in ('von_mises','oxx','oyy','txy','omax','omin',
                       'von_mises_top','von_mises_bottom',
                       'nodal_vm','noxx','noyy','ntxy','nomax','nomin')):
        e1d, e2d, e3d = state.model.elements_by_dim()
        if e1d and (e2d or e3d):
            note_y = bar_y + bar_h + 62
            dl.add_text((bar_x, note_y), 0xFFAAAA88, "Frame stress not displayed")

    # Result info strip at bottom


_CMAPS = ['rainbow','jet','coolwarm','grayscale']
_FORCE_KEYS = {'fx','fy','fxy','mx','my','mxy','qx','qy'}
_RESULTS = [
    ('displacement', 'Displacement (Total)'),
    ('t1',           'Displacement T1'),
    ('t2',           'Displacement T2'),
    ('t3',           'Displacement T3'),
    ('von_mises',    'Stress - Von Mises'),
    ('oxx',          'Stress - Sxx'),
    ('oyy',          'Stress - Syy'),
    ('txy',          'Stress - Sxy'),
    ('omax',         'Stress - S1'),
    ('omin',         'Stress - S3'),
    ('von_mises_top','Stress - VM Top'),
    ('von_mises_bottom','Stress - VM Bottom'),
    ('sxc',          'Beam Stress C'),
    ('sxd',          'Beam Stress D'),
    ('sxe',          'Beam Stress E'),
    ('sxf',          'Beam Stress F'),
    ('smax',         'Beam Stress Smax'),
    ('smin',         'Beam Stress Smin'),
    ('stress3d',     'Beam Stress 3D'),
    ('nodal_vm',     'Stress Nodal VM'),
    ('noxx',         'Stress Nodal Sxx'),
    ('noyy',         'Stress Nodal Syy'),
    ('ntxy',         'Stress Nodal Sxy'),
    ('nomax',        'Stress Nodal S1'),
    ('nomin',        'Stress Nodal S3'),
    ('fx',  'Force FX (Membrane X)'),
    ('fy',  'Force FY (Membrane Y)'),
    ('fxy', 'Force FXY (Shear)'),
    ('mx',  'Force MX (Bending X)'),
    ('my',  'Force MY (Bending Y)'),
    ('mxy', 'Force MXY (Twist)'),
    ('qx',  'Force QX (Shear X)'),
    ('qy',  'Force QY (Shear Y)'),
    ('nfx', 'Force Nodal FX'),
    ('nfy', 'Force Nodal FY'),
    ('nfxy','Force Nodal FXY'),
    ('nmx', 'Force Nodal MX'),
    ('nmy', 'Force Nodal MY'),
    ('nmxy','Force Nodal MXY'),
    ('nqx', 'Force Nodal QX'),
    ('nqy', 'Force Nodal QY'),
]

_NODAL_STRESS_MAP = {
    'von_mises': 'nodal_vm',
    'oxx': 'noxx',
    'oyy': 'noyy',
    'txy': 'ntxy',
    'omax': 'nomax',
    'omin': 'nomin',
}
_ELEMENT_STRESS_MAP = {v: k for k, v in _NODAL_STRESS_MAP.items()}
_BEAM_STRESS_KEYS = ('sxc','sxd','sxe','sxf','smax','smin')
_NODAL_FORCE_MAP = {
    'fx': 'nfx', 'fy': 'nfy', 'fxy': 'nfxy', 'mx': 'nmx',
    'my': 'nmy', 'mxy': 'nmxy', 'qx': 'nqx', 'qy': 'nqy',
}
_ELEMENT_FORCE_MAP = {v: k for k, v in _NODAL_FORCE_MAP.items()}


def _stress_nodal_components(state: ViewerState):
    return state.nodal_stress_components()


def _has_nodal_stress(state: ViewerState) -> bool:
    return bool(state.nodal_stress_components())


def _has_nodal_force(state: ViewerState) -> bool:
    return bool(state.nodal_force_components())

def draw_contour_toolbar(state: ViewerState, width: int, height: int):
    """Small toolbar at top for contour/deform options."""
    if state.results is None: return

    tw = 620; th = 28
    imgui.set_next_window_pos(((width-tw)//2, 22), imgui.Cond_.always)
    imgui.set_next_window_size((tw, th), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.82)
    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##ctoolbar", None, flags)
    rebuild = False

    if state.display_mode == 'contour':
        # Build available result list dynamically based on loaded results
        avail_keys = ['displacement','t1','t2','t3']  # always show displacement
        if state.results and state.subcase in state.results.stresses:
            st = state.results.stresses[state.subcase]
            elem_st = [v for k,v in st.items()
                       if k not in ('_nodal_avg', '_nodal_avg_components', '_nodal_acc',
                                    '_solver_nodal_avg', '_solver_nodal_avg_components',
                                    '_derived_nodal_avg', '_derived_nodal_avg_components',
                                    '_shell_corner_contribs')
                       and hasattr(v,'values') and hasattr(v, 'von_mises')
                       and not _is_beam_stress_obj(v)]
            if elem_st:
                # Check across elements for non-zero values
                for k in ('von_mises','oxx','oyy','txy','omax','omin',
                          'von_mises_top','von_mises_bottom'):
                    if any(abs(es.values.get(k,0)) > 1e-15 for es in elem_st):
                        avail_keys.append(k)
            nav = state.nodal_stress_components()
            if nav:
                for k in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin'):
                    if k not in avail_keys:
                        avail_keys.append(k)
            diag = getattr(state, 'beam_diagram', None)
            if diag is not None and getattr(diag, 'beam_stress_available', False):
                for k in _BEAM_STRESS_KEYS:
                    if k not in avail_keys:
                        avail_keys.append(k)
                if 'stress3d' not in avail_keys:
                    avail_keys.append('stress3d')
        if hasattr(state.results, 'forces') and state.subcase in state.results.forces:
            fc = state.results.forces[state.subcase]
            if fc:
                for k in ('fx','fy','fxy','mx','my','mxy','qx','qy'):
                    if any(hasattr(v, 'values') and abs(v.values.get(k,0)) > 1e-15
                           for v in fc.values()):
                        avail_keys.append(k)
                navf = state.nodal_force_components()
                if navf:
                    for k in ('nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy'):
                        avail_keys.append(k)
        avail_keys = list(dict.fromkeys(avail_keys))

        res_map = dict(_RESULTS)
        avail_labels = [res_map.get(k, k) for k in avail_keys]
        if avail_keys and state.result_type not in avail_keys:
            state.result_type = avail_keys[0]

        imgui.text("Result:")
        imgui.same_line()
        imgui.set_next_item_width(160)
        cur = avail_keys.index(state.result_type) if state.result_type in avail_keys else 0
        ch, idx = imgui.combo("##rt", cur, avail_labels)
        if ch: state.result_type = avail_keys[idx]; rebuild = True
        imgui.same_line()

        # Elem/Nodal toggle
        _STRESS_TYPES = tuple(_NODAL_STRESS_MAP.keys()) + tuple(_NODAL_STRESS_MAP.values())
        _ETYPE_KEYS   = tuple(_NODAL_FORCE_MAP.keys())
        _NTYPE_KEYS   = tuple(_NODAL_FORCE_MAP.values())
        is_stress = state.result_type in _STRESS_TYPES
        is_etype  = state.result_type in _ETYPE_KEYS
        is_ntype  = state.result_type in _NTYPE_KEYS
        if is_stress:
            is_nodal = state.result_type in _ELEMENT_STRESS_MAP
            lbl_en = "[Nodal]" if is_nodal else "[Elem]"
            if imgui.button(lbl_en):
                if is_nodal:
                    state.result_type = _ELEMENT_STRESS_MAP[state.result_type]
                    rebuild = True
                else:
                    if _has_nodal_stress(state):
                        state.result_type = _NODAL_STRESS_MAP.get(state.result_type, 'nodal_vm')
                        rebuild = True
                    else:
                        state.contour_notice = "Only center element stress is available. No averaged nodal stress was produced."
                        state.open_contour_notice = True
            imgui.same_line()
        elif is_etype or is_ntype:
            lbl_fn = "[Nodal]" if is_ntype else "[Elem]"
            if _has_nodal_force(state):
                if imgui.button(lbl_fn + "##f"):
                    if is_ntype: state.result_type = _ELEMENT_FORCE_MAP[state.result_type]
                    else:        state.result_type = _NODAL_FORCE_MAP[state.result_type]
                    rebuild = True
            imgui.same_line()

        imgui.text("Cmap:")
        imgui.same_line()
        imgui.set_next_item_width(90)
        cur_cm = _CMAPS.index(state.cmap_name) if state.cmap_name in _CMAPS else 0
        ch2, idx2 = imgui.combo("##cm", cur_cm, _CMAPS)
        if ch2: state.cmap_name = _CMAPS[idx2]; rebuild = True
        imgui.same_line()
        if state.deform_scale > 0:
            ghost_lbl = "Ghost: ON" if state.show_undeformed else "Ghost: off"
            if imgui.button(ghost_lbl):
                state.show_undeformed = not state.show_undeformed
                rebuild = True

    elif state.display_mode in ('beam', 'beam_v2'):
        diag = getattr(state, 'beam_diagram', None)
        if diag is not None and getattr(diag, 'beam_data', {}):
            results = diag.available_results()
            keys = [r[0] for r in results]
            labels = [
                (f"{r[1]} [Approx]" if state.display_mode == 'beam_v2' else r[1])
                for r in results
            ]
            if state.beam_result_key not in keys and keys:
                state.beam_result_key = keys[0]
            imgui.text("Diagram:")
            imgui.same_line()
            imgui.set_next_item_width(200)
            cur = keys.index(state.beam_result_key) if state.beam_result_key in keys else 0
            ch, idx = imgui.combo("##beamrt", cur, labels)
            if ch:
                state.beam_result_key = keys[idx]
                state.beam_scale = 0.0
                state.request_beam_rebuild = True
                state.request_fit = True
            imgui.same_line()
            if state.results and len(state.results.subcases) > 1:
                sc_list = [_subcase_display_label(state, s) for s in state.results.subcases]
                cur_sc = state.results.subcases.index(state.subcase) if state.subcase in state.results.subcases else 0
                imgui.set_next_item_width(170)
                ch_sc, idx_sc = imgui.combo("##beamsc", cur_sc, sc_list)
                if ch_sc:
                    state.subcase = state.results.subcases[idx_sc]
                    rebuild = True
            imgui.same_line()
            if state.deform_scale > 0:
                ghost_lbl = "Ghost: ON" if state.show_undeformed else "Ghost: off"
                if imgui.button(ghost_lbl + "##beamghost"):
                    state.show_undeformed = not state.show_undeformed
                    rebuild = True

    if rebuild: state.request_rebuild = True
    if state.open_contour_notice:
        imgui.open_popup("Stress Notice")
        state.open_contour_notice = False
    if imgui.begin_popup_modal("Stress Notice", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text_wrapped(state.contour_notice or "No nodal stress is available.")
        if imgui.button("OK", (120, 0)):
            imgui.close_current_popup()
        imgui.end_popup()
    imgui.end()


def draw_beam_diagram_panel(state: ViewerState, width: int, height: int):
    """Beam internal force diagram controls."""
    return


def draw_node_info(state: ViewerState, width: int, height: int):
    if state.selected_node<0 or state.model is None: return
    node=state.model.nodes.get(state.selected_node)
    if not node: return
    imgui.set_next_window_pos((275,height-120),imgui.Cond_.always)
    imgui.set_next_window_size((230,95),imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.82)
    flags=(imgui.WindowFlags_.no_title_bar|imgui.WindowFlags_.no_resize|imgui.WindowFlags_.no_move)
    imgui.begin("##ninfo",None,flags)
    imgui.push_style_color(imgui.Col_.text,imgui.ImVec4(1.0,0.85,0.3,1.0))
    imgui.text(f"Node {state.selected_node}"); imgui.pop_style_color()
    imgui.text(f"  X={node.xyz[0]:.5g}  Y={node.xyz[1]:.5g}  Z={node.xyz[2]:.5g}")
    if state.results and state.subcase in state.results.displacements:
        d=state.results.displacements[state.subcase].get(state.selected_node)
        if d: imgui.text(f"  |u|={d.magnitude:.5e}")
    imgui.end()


