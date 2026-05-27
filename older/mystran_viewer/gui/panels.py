"""
ImGui panels for MYSTRAN Viewer.
"""

import os
import numpy as np
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
import renderer.mesh_renderer as mr   # to edit color arrays live


class ViewerState:
    def __init__(self):
        self.dat_path: str = ""
        self.f06_path: str = ""

        self.display_mode: str  = "wireframe"
        self.result_type: str   = "displacement"
        self.cmap_name: str     = "rainbow"
        self.subcase: int       = 1
        self.deform_scale: float = 0.0
        self.show_edges: bool   = True
        self.show_spc:   bool   = True
        self.show_nodes: bool   = False
        self.visible_dims: list = [True, True, True]   # 1d, 2d, 3d

        # Background
        self.bg_color: list     = [0.22, 0.23, 0.28, 1.0]

        # Per-dim wire colors (list so imgui can mutate)
        self.c_wire_1d: list = list(mr.C_1D_WIRE)
        self.c_wire_2d: list = list(mr.C_2D_WIRE)
        self.c_wire_3d: list = list(mr.C_3D_WIRE)
        self.c_fill_1d: list = list(mr.C_1D_FILL)
        self.c_fill_2d: list = list(mr.C_2D_FILL)
        self.c_fill_3d: list = list(mr.C_3D_FILL)
        self.c_spc:     list = list(mr.C_SPC)

        self.model:   Optional[MystranModel] = None
        self.results: Optional[F06Results]   = None
        self.selected_node: int = -1
        self.selected_elem: int = -1

        self.request_load_dat: bool  = False
        self.request_load_f06: bool  = False
        self.request_fit:      bool  = False
        self.request_rebuild:  bool  = False
        self.request_quit:     bool  = False

        self.fps: float        = 0.0
        self.frame_time: float = 0.0

        self._dat_dialog = None
        self._f06_dialog = None

    def push_colors_to_renderer(self):
        """Copy GUI color state → renderer module arrays."""
        import numpy as np
        mr.C_1D_WIRE[:] = self.c_wire_1d
        mr.C_2D_WIRE[:] = self.c_wire_2d
        mr.C_3D_WIRE[:] = self.c_wire_3d
        mr.C_1D_FILL[:] = self.c_fill_1d
        mr.C_2D_FILL[:] = self.c_fill_2d
        mr.C_3D_FILL[:] = self.c_fill_3d
        mr.C_SPC[:]     = self.c_spc

    def visible_dim_tuple(self):
        dims = []
        if self.visible_dims[0]: dims.append('1d')
        if self.visible_dims[1]: dims.append('2d')
        if self.visible_dims[2]: dims.append('3d')
        return tuple(dims)


# ---------------------------------------------------------------------------

def _open_file_dialog(title, filters):
    if _HAS_PFD:
        return pfd.open_file(title, filters=filters)
    return None

def _dialog_result(dlg):
    if dlg is None: return None
    if _HAS_PFD and dlg.ready():
        r = dlg.result()
        return r[0] if r else None
    return None


# ---------------------------------------------------------------------------

def draw_menu_bar(state: ViewerState):
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            if imgui.menu_item("Open .dat...", "", False)[0]:
                state._dat_dialog = _open_file_dialog("Open .dat", ["*.dat *.nas *.bdf"])
            if imgui.menu_item("Open .f06...", "", False)[0]:
                state._f06_dialog = _open_file_dialog("Open .f06", ["*.f06"])
            imgui.separator()
            if imgui.menu_item("Quit", "Esc", False)[0]:
                state.request_quit = True
            imgui.end_menu()

        if imgui.begin_menu("View"):
            if imgui.menu_item("Fit to Model", "F", False)[0]:
                state.request_fit = True
            imgui.separator()
            ch, state.show_edges = imgui.menu_item("Show Edges", "", state.show_edges)
            ch, state.show_spc   = imgui.menu_item("Show BCs",   "", state.show_spc)
            imgui.end_menu()

        imgui.end_main_menu_bar()

    # File dialog results
    if state._dat_dialog:
        r = _dialog_result(state._dat_dialog)
        if r is not None:
            state.dat_path = r
            state.request_load_dat = True
            state._dat_dialog = None

    if state._f06_dialog:
        r = _dialog_result(state._f06_dialog)
        if r is not None:
            state.f06_path = r
            state.request_load_f06 = True
            state._f06_dialog = None


# ---------------------------------------------------------------------------

def draw_left_panel(state: ViewerState):
    imgui.set_next_window_pos((0, 20), imgui.Cond_.always)
    imgui.set_next_window_size((270, 620), imgui.Cond_.always)

    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_scroll_with_mouse)
    imgui.begin("##left", None, flags)

    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 0.85, 1.0, 1.0))
    imgui.text("MYSTRAN VIEWER")
    imgui.pop_style_color()
    imgui.separator()

    # DAT
    imgui.text("DAT file:")
    imgui.set_next_item_width(-40)
    _, state.dat_path = imgui.input_text("##dat", state.dat_path)
    imgui.same_line()
    if imgui.button("...##dat"):
        state._dat_dialog = _open_file_dialog("Open .dat", ["*.dat *.nas *.bdf"])
    if imgui.button("Load .dat", (-1, 0)):
        state.request_load_dat = True

    imgui.spacing()
    imgui.text("F06 file:")
    imgui.set_next_item_width(-40)
    _, state.f06_path = imgui.input_text("##f06", state.f06_path)
    imgui.same_line()
    if imgui.button("...##f06"):
        state._f06_dialog = _open_file_dialog("Open .f06", ["*.f06"])
    if imgui.button("Load .f06", (-1, 0)):
        state.request_load_f06 = True

    imgui.separator()

    # Model info
    if state.model:
        m = state.model
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.85, 0.3, 1.0))
        imgui.text("Model Info")
        imgui.pop_style_color()
        if m.title:
            imgui.text(f"  {m.title[:30]}")
        imgui.text(f"  Nodes:     {len(m.nodes)}")
        e1d, e2d, e3d = m.elements_by_dim()
        imgui.text(f"  1D elem:   {len(e1d)}")
        imgui.text(f"  2D elem:   {len(e2d)}")
        imgui.text(f"  3D elem:   {len(e3d)}")
        imgui.text(f"  SPCs:      {len(m.spcs)}")
        imgui.text(f"  Forces:    {len(m.forces)}")
        imgui.text(f"  Materials: {len(m.materials)}")
        imgui.separator()

    # Results info
    if state.results:
        r = state.results
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.85, 0.3, 1.0))
        imgui.text("Results")
        imgui.pop_style_color()
        imgui.text(f"  Subcases:  {r.subcases}")
        if state.subcase in r.displacements:
            imgui.text(f"  Max |u|:   {r.max_displacement(state.subcase):.4e}")
        imgui.separator()

    if imgui.button("Fit View  [F]", (-1, 0)):
        state.request_fit = True

    # Visibility toggles
    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 0.9, 0.7, 1.0))
    imgui.text("Visible")
    imgui.pop_style_color()
    dims = [("1D (beam/rod)", 0), ("2D (shell)", 1), ("3D (solid)", 2)]
    rebuild = False
    for label, idx in dims:
        ch, state.visible_dims[idx] = imgui.checkbox(label, state.visible_dims[idx])
        if ch: rebuild = True
    if rebuild:
        state.request_rebuild = True

    imgui.spacing()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
    imgui.text(f"FPS: {state.fps:.1f}   {state.frame_time*1000:.1f} ms")
    imgui.pop_style_color()

    imgui.end()


# ---------------------------------------------------------------------------

def draw_right_panel(state: ViewerState, width: int, height: int):
    pw = 255
    imgui.set_next_window_pos((width - pw, 20), imgui.Cond_.always)
    imgui.set_next_window_size((pw, height - 20), imgui.Cond_.always)

    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move)
    imgui.begin("##right", None, flags)

    # ---- Display mode ----
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 0.85, 1.0, 1.0))
    imgui.text("Display Mode")
    imgui.pop_style_color()
    imgui.separator()

    rebuild = False
    modes = [("Wireframe  [W]", "wireframe"),
             ("Solid      [S]", "solid"),
             ("Contour    [C]", "contour")]
    for label, mode in modes:
        if imgui.radio_button(label, state.display_mode == mode):
            if state.display_mode != mode:
                state.display_mode = mode
                rebuild = True

    # ---- Contour options ----
    if state.display_mode == "contour":
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 1.0, 0.7, 1.0))
        imgui.text("Result")
        imgui.pop_style_color()

        for label, rt in [("Displacement","displacement"),("Von Mises","von_mises")]:
            if imgui.radio_button(label, state.result_type == rt):
                if state.result_type != rt:
                    state.result_type = rt
                    rebuild = True

        imgui.spacing()
        imgui.text("Colormap")
        cmaps = list(COLORMAPS.keys())
        cur = cmaps.index(state.cmap_name) if state.cmap_name in cmaps else 0
        imgui.set_next_item_width(-1)
        ch, idx = imgui.combo("##cmap", cur, cmaps)
        if ch:
            state.cmap_name = cmaps[idx]
            rebuild = True

        if state.results and len(state.results.subcases) > 1:
            imgui.text("Subcase")
            sc_list = [str(s) for s in state.results.subcases]
            cur_sc = state.results.subcases.index(state.subcase) if state.subcase in state.results.subcases else 0
            imgui.set_next_item_width(-1)
            ch, idx = imgui.combo("##sc", cur_sc, sc_list)
            if ch:
                state.subcase = state.results.subcases[idx]
                rebuild = True

    # ---- Deformation ----
    imgui.spacing()
    imgui.separator()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 1.0, 0.7, 1.0))
    imgui.text("Deformation Scale")
    imgui.pop_style_color()
    imgui.set_next_item_width(-1)
    ch, nd = imgui.slider_float("##def", state.deform_scale, 0.0, 200.0, "%.1f x")
    if ch:
        state.deform_scale = nd
        rebuild = True

    # ---- Overlay ----
    imgui.spacing()
    imgui.separator()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 1.0, 0.7, 1.0))
    imgui.text("Overlay")
    imgui.pop_style_color()
    imgui.checkbox("Show Edges", state.show_edges)
    imgui.same_line()
    _, state.show_edges = imgui.checkbox("##edges", state.show_edges)
    _, state.show_spc   = imgui.checkbox("Show BCs (SPCs)", state.show_spc)

    # ---- Options / Colors ----
    imgui.spacing()
    imgui.separator()
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 0.85, 1.0, 1.0))
    imgui.text("Options")
    imgui.pop_style_color()

    if imgui.tree_node("Background Color"):
        imgui.set_next_item_width(-1)
        imgui.color_edit3("##bg", state.bg_color)
        imgui.tree_pop()

    if imgui.tree_node("Element Colors"):
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.4, 0.85, 1.0, 1.0))
        imgui.text("1D (beam/rod)")
        imgui.pop_style_color()
        imgui.set_next_item_width(-1)
        ch1, state.c_wire_1d = imgui.color_edit3("Wire##1d", state.c_wire_1d)
        imgui.set_next_item_width(-1)
        ch2, state.c_fill_1d = imgui.color_edit3("Fill##1d", state.c_fill_1d)

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.3, 1.0, 0.65, 1.0))
        imgui.text("2D (shell/plate)")
        imgui.pop_style_color()
        imgui.set_next_item_width(-1)
        ch3, state.c_wire_2d = imgui.color_edit3("Wire##2d", state.c_wire_2d)
        imgui.set_next_item_width(-1)
        ch4, state.c_fill_2d = imgui.color_edit3("Fill##2d", state.c_fill_2d)

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.75, 0.2, 1.0))
        imgui.text("3D (solid)")
        imgui.pop_style_color()
        imgui.set_next_item_width(-1)
        ch5, state.c_wire_3d = imgui.color_edit3("Wire##3d", state.c_wire_3d)
        imgui.set_next_item_width(-1)
        ch6, state.c_fill_3d = imgui.color_edit3("Fill##3d", state.c_fill_3d)

        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.3, 0.15, 1.0))
        imgui.text("SPC markers")
        imgui.pop_style_color()
        imgui.set_next_item_width(-1)
        ch7, state.c_spc = imgui.color_edit3("Color##spc", state.c_spc)

        if any([ch1,ch2,ch3,ch4,ch5,ch6,ch7]):
            state.push_colors_to_renderer()
            rebuild = True

        imgui.tree_pop()

    if rebuild:
        state.request_rebuild = True

    imgui.end()


# ---------------------------------------------------------------------------

def draw_legend(state: ViewerState, width: int, height: int):
    if state.display_mode != "contour" or state.results is None:
        return

    lw, lh = 340, 48
    lx = (width - lw) // 2
    ly = height - lh - 8

    imgui.set_next_window_pos((lx, ly), imgui.Cond_.always)
    imgui.set_next_window_size((lw, lh), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.75)

    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##legend", None, flags)

    sc = state.subcase
    r  = state.results

    if state.result_type == "displacement" and sc in r.displacements:
        dmax = r.max_displacement(sc)
        vmin, vmax = 0.0, dmax
        label = "Displacement"
    elif state.result_type == "von_mises" and sc in r.stresses:
        _, vms = r.von_mises_array(sc)
        vmin = float(vms.min()) if len(vms) else 0.0
        vmax = float(vms.max()) if len(vms) else 1.0
        label = "Von Mises"
    else:
        imgui.end()
        return

    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.9, 0.4, 1.0))
    imgui.text(f"{label}")
    imgui.pop_style_color()
    imgui.same_line()
    ticks = legend_ticks(vmin, vmax, 5)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 1.0, 0.7, 1.0))
    imgui.text("   ".join([f"{t:.3e}" for t in ticks]))
    imgui.pop_style_color()
    imgui.end()


# ---------------------------------------------------------------------------

def draw_node_info(state: ViewerState, width: int, height: int):
    if state.selected_node < 0 or state.model is None:
        return
    node = state.model.nodes.get(state.selected_node)
    if not node:
        return

    imgui.set_next_window_pos((275, height - 120), imgui.Cond_.always)
    imgui.set_next_window_size((230, 95), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.82)
    flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_move)
    imgui.begin("##nodeinfo", None, flags)

    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.85, 0.3, 1.0))
    imgui.text(f"Node {state.selected_node}")
    imgui.pop_style_color()
    imgui.text(f"  X={node.xyz[0]:.5g}  Y={node.xyz[1]:.5g}  Z={node.xyz[2]:.5g}")

    if state.results and state.subcase in state.results.displacements:
        d = state.results.displacements[state.subcase].get(state.selected_node)
        if d:
            imgui.text(f"  |u|= {d.magnitude:.5e}")
            imgui.text(f"  ux={d.t1:.3e}  uy={d.t2:.3e}  uz={d.t3:.3e}")
    imgui.end()


# ---------------------------------------------------------------------------
# View Cube / Preset buttons
# ---------------------------------------------------------------------------

_VIEW_PRESETS = [
    # label,   yaw,    pitch,  tooltip
    ("+X",      90,      0,   "View from +X (YZ plane)"),
    ("-X",     270,      0,   "View from -X"),
    ("+Y",       0,     90,   "View from +Y (XZ plane, top)"),
    ("-Y",       0,    -90,   "View from -Y (bottom)"),
    ("+Z",       0,      0,   "View from +Z (XY plane, front)"),
    ("-Z",     180,      0,   "View from -Z (back)"),
    ("ISO",     45,     25,   "Isometric view"),
]

def draw_view_cube(state, camera, width: int, height: int):
    """
    Small floating panel with face-click view presets.
    Mimics the ViewCube in CAD software.
    Placed top-right, above the axis widget.
    """
    pw, ph = 108, 162
    px = width - pw - 8
    py = 28   # just below menu bar

    imgui.set_next_window_pos((px, py), imgui.Cond_.always)
    imgui.set_next_window_size((pw, ph), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.55)

    flags = (imgui.WindowFlags_.no_title_bar  |
             imgui.WindowFlags_.no_resize      |
             imgui.WindowFlags_.no_move        |
             imgui.WindowFlags_.no_scrollbar)
    imgui.begin("##viewcube", None, flags)

    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.85, 1.0, 1.0))
    imgui.text(" View")
    imgui.pop_style_color()

    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(2, 2))

    for label, yaw, pitch, tip in _VIEW_PRESETS:
        bw = (pw - 16) // 2
        if imgui.button(label, (bw, 18)):
            camera._yaw   = float(yaw)
            camera._pitch = float(pitch)
        if imgui.is_item_hovered():
            imgui.set_tooltip(tip)
        if label not in ("+X", "+Y", "+Z", "ISO"):
            imgui.same_line()

    imgui.pop_style_var()
    imgui.end()
