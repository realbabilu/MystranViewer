"""
ImGui panel definitions for MYSTRAN Viewer.
Uses imgui_bundle.
"""

import os
import numpy as np
from typing import Optional, List

from imgui_bundle import imgui, hello_imgui, portable_file_dialogs as pfd

from parser.dat_parser import MystranModel
from parser.f06_parser  import F06Results
from renderer.contour   import legend_ticks, COLORMAPS


class ViewerState:
    """Mutable state shared between GUI and main loop."""
    def __init__(self):
        # File paths
        self.dat_path: str = ""
        self.f06_path: str = ""

        # Display
        self.display_mode: str = "wireframe"   # wireframe / solid / contour
        self.result_type: str  = "displacement" # displacement / von_mises
        self.cmap_name: str    = "rainbow"
        self.subcase: int      = 1
        self.deform_scale: float = 0.0
        self.show_edges: bool  = True
        self.show_spc: bool    = True
        self.show_nodes: bool  = False
        self.bg_color: list    = [0.12, 0.13, 0.15, 1.0]

        # Derived / result
        self.model: Optional[MystranModel] = None
        self.results: Optional[F06Results] = None
        self.selected_node: int = -1
        self.selected_elem: int = -1

        # Requests (set by GUI, consumed by main loop)
        self.request_load_dat: bool   = False
        self.request_load_f06: bool   = False
        self.request_fit:      bool   = False
        self.request_rebuild:  bool   = False   # rebuild GPU buffers

        # Stats
        self.fps: float = 0.0
        self.frame_time: float = 0.0

        # File dialog state
        self._dat_dialog = None
        self._f06_dialog = None


# ---------------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------------

def draw_menu_bar(state: ViewerState):
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            if imgui.menu_item("Open .dat...", "")[0]:
                state._dat_dialog = pfd.open_file(
                    "Open MYSTRAN .dat", filters=["*.dat", "*.nas", "*.bdf"])
            if imgui.menu_item("Open .f06...", "")[0]:
                state._f06_dialog = pfd.open_file(
                    "Open MYSTRAN .f06", filters=["*.f06"])
            imgui.separator()
            if imgui.menu_item("Quit", "Alt+F4")[0]:
                import glfw
                # signal quit via state
                state.request_quit = True
            imgui.end_menu()

        if imgui.begin_menu("View"):
            if imgui.menu_item("Fit to Model", "F")[0]:
                state.request_fit = True
            imgui.separator()
            _, state.show_edges = imgui.menu_item("Show Edges",   "", state.show_edges)
            _, state.show_spc   = imgui.menu_item("Show BCs",     "", state.show_spc)
            _, state.show_nodes = imgui.menu_item("Show Nodes",   "", state.show_nodes)
            imgui.end_menu()

        imgui.end_main_menu_bar()

    # Handle file dialog results
    if state._dat_dialog and state._dat_dialog.is_ready():
        r = state._dat_dialog.result()
        if r:
            state.dat_path = r[0]
            state.request_load_dat = True
        state._dat_dialog = None

    if state._f06_dialog and state._f06_dialog.is_ready():
        r = state._f06_dialog.result()
        if r:
            state.f06_path = r[0]
            state.request_load_f06 = True
        state._f06_dialog = None


def draw_left_panel(state: ViewerState):
    """Left sidebar: model tree + properties."""
    imgui.set_next_window_pos((0, 20), imgui.Cond_.always)
    imgui.set_next_window_size((280, 600), imgui.Cond_.always)

    flags = (imgui.WindowFlags_.no_title_bar |
             imgui.WindowFlags_.no_resize    |
             imgui.WindowFlags_.no_move      |
             imgui.WindowFlags_.no_scrollbar_with_mouse)

    imgui.begin("##left_panel", None, flags)

    # --- File section ---
    imgui.push_style_color(imgui.Col_.text, (0.6, 0.9, 1.0, 1.0))
    imgui.text("MYSTRAN VIEWER")
    imgui.pop_style_color()
    imgui.separator()

    imgui.text("DAT:")
    imgui.same_line()
    imgui.set_next_item_width(-40)
    _, state.dat_path = imgui.input_text("##dat", state.dat_path)
    imgui.same_line()
    if imgui.button("...##dat"):
        state._dat_dialog = pfd.open_file("Open .dat", filters=["*.dat *.nas *.bdf"])
    if imgui.button("Load .dat", (-1, 0)):
        state.request_load_dat = True

    imgui.spacing()
    imgui.text("F06:")
    imgui.same_line()
    imgui.set_next_item_width(-40)
    _, state.f06_path = imgui.input_text("##f06", state.f06_path)
    imgui.same_line()
    if imgui.button("...##f06"):
        state._f06_dialog = pfd.open_file("Open .f06", filters=["*.f06"])
    if imgui.button("Load .f06", (-1, 0)):
        state.request_load_f06 = True

    imgui.separator()

    # --- Model info ---
    if state.model:
        m = state.model
        imgui.push_style_color(imgui.Col_.text, (0.9, 0.8, 0.4, 1.0))
        imgui.text("Model")
        imgui.pop_style_color()

        imgui.text(f"  Title:    {m.title[:28] if m.title else '(none)'}")
        imgui.text(f"  Nodes:    {len(m.nodes)}")
        e1d, e2d, e3d = m.elements_by_dim()
        imgui.text(f"  1D elem:  {len(e1d)}")
        imgui.text(f"  2D elem:  {len(e2d)}")
        imgui.text(f"  3D elem:  {len(e3d)}")
        imgui.text(f"  SPCs:     {len(m.spcs)}")
        imgui.text(f"  Forces:   {len(m.forces)}")
        imgui.text(f"  Materials:{len(m.materials)}")
        imgui.separator()

    # --- Results info ---
    if state.results:
        r = state.results
        imgui.push_style_color(imgui.Col_.text, (0.9, 0.8, 0.4, 1.0))
        imgui.text("Results")
        imgui.pop_style_color()

        imgui.text(f"  Subcases: {r.subcases}")
        sc = state.subcase
        if sc in r.displacements:
            dmax = r.max_displacement(sc)
            imgui.text(f"  Max disp: {dmax:.4e}")
        imgui.separator()

    if imgui.button("Fit View [F]", (-1, 0)):
        state.request_fit = True

    imgui.end()


def draw_right_panel(state: ViewerState, width: int, height: int):
    """Right sidebar: display settings."""
    pw = 240
    imgui.set_next_window_pos((width - pw, 20), imgui.Cond_.always)
    imgui.set_next_window_size((pw, height - 20), imgui.Cond_.always)

    flags = (imgui.WindowFlags_.no_title_bar |
             imgui.WindowFlags_.no_resize    |
             imgui.WindowFlags_.no_move)

    imgui.begin("##right_panel", None, flags)

    imgui.push_style_color(imgui.Col_.text, (0.9, 0.8, 0.4, 1.0))
    imgui.text("Display")
    imgui.pop_style_color()
    imgui.separator()

    # Display mode radio buttons
    modes = [("Wireframe", "wireframe"), ("Solid", "solid"), ("Contour", "contour")]
    for label, mode in modes:
        if imgui.radio_button(label, state.display_mode == mode):
            if state.display_mode != mode:
                state.display_mode = mode
                state.request_rebuild = True

    imgui.separator()

    # Contour options (only when contour active)
    if state.display_mode == "contour":
        imgui.push_style_color(imgui.Col_.text, (0.7, 0.9, 0.7, 1.0))
        imgui.text("Result Type")
        imgui.pop_style_color()

        rt_modes = [("Displacement", "displacement"), ("Von Mises", "von_mises")]
        for label, rt in rt_modes:
            if imgui.radio_button(label, state.result_type == rt):
                if state.result_type != rt:
                    state.result_type = rt
                    state.request_rebuild = True

        imgui.spacing()
        imgui.text("Colormap")
        cmaps = list(COLORMAPS.keys())
        cur_idx = cmaps.index(state.cmap_name) if state.cmap_name in cmaps else 0
        imgui.set_next_item_width(-1)
        changed, new_idx = imgui.combo("##cmap", cur_idx, cmaps)
        if changed:
            state.cmap_name = cmaps[new_idx]
            state.request_rebuild = True

        # Subcase selector
        if state.results and len(state.results.subcases) > 1:
            imgui.spacing()
            imgui.text("Subcase")
            sc_list = [str(s) for s in state.results.subcases]
            cur_sc = state.results.subcases.index(state.subcase) if state.subcase in state.results.subcases else 0
            imgui.set_next_item_width(-1)
            changed_sc, new_sc_idx = imgui.combo("##subcase", cur_sc, sc_list)
            if changed_sc:
                state.subcase = state.results.subcases[new_sc_idx]
                state.request_rebuild = True

        imgui.spacing()
        imgui.separator()

    # Deformation scale
    imgui.push_style_color(imgui.Col_.text, (0.7, 0.9, 0.7, 1.0))
    imgui.text("Deformation")
    imgui.pop_style_color()
    imgui.set_next_item_width(-1)
    changed_def, new_def = imgui.slider_float("##deform", state.deform_scale, 0.0, 100.0, "%.1fx")
    if changed_def:
        state.deform_scale = new_def
        state.request_rebuild = True

    imgui.separator()

    # Overlay toggles
    imgui.push_style_color(imgui.Col_.text, (0.7, 0.9, 0.7, 1.0))
    imgui.text("Overlay")
    imgui.pop_style_color()

    ch_e, state.show_edges = imgui.checkbox("Show Edges", state.show_edges)
    ch_b, state.show_spc   = imgui.checkbox("Show BCs",   state.show_spc)
    ch_n, state.show_nodes = imgui.checkbox("Show Nodes", state.show_nodes)
    if ch_e or ch_b or ch_n:
        pass  # no rebuild needed, handled in draw call

    imgui.separator()

    # Background color
    imgui.push_style_color(imgui.Col_.text, (0.7, 0.9, 0.7, 1.0))
    imgui.text("Background")
    imgui.pop_style_color()
    imgui.set_next_item_width(-1)
    imgui.color_edit3("##bg", state.bg_color)

    # Performance
    imgui.spacing()
    imgui.separator()
    imgui.push_style_color(imgui.Col_.text, (0.5, 0.5, 0.5, 1.0))
    imgui.text(f"FPS: {state.fps:.1f}")
    imgui.text(f"Frame: {state.frame_time*1000:.2f} ms")
    imgui.pop_style_color()

    imgui.end()


def draw_legend(state: ViewerState, width: int, height: int):
    """Colorbar legend at bottom."""
    if state.display_mode != "contour" or state.results is None:
        return
    if state.model is None:
        return

    lw, lh = 300, 50
    lx = (width - lw) // 2
    ly = height - lh - 10

    imgui.set_next_window_pos((lx, ly), imgui.Cond_.always)
    imgui.set_next_window_size((lw, lh), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.7)

    flags = (imgui.WindowFlags_.no_title_bar     |
             imgui.WindowFlags_.no_resize         |
             imgui.WindowFlags_.no_move           |
             imgui.WindowFlags_.no_scrollbar)

    imgui.begin("##legend", None, flags)

    sc = state.subcase
    r  = state.results

    if state.result_type == "displacement" and sc in r.displacements:
        dmax = r.max_displacement(sc)
        vmin, vmax = 0.0, dmax
        label = "Displacement"
    elif state.result_type == "von_mises" and sc in r.stresses:
        ids, vms = r.von_mises_array(sc)
        vmin = float(vms.min()) if len(vms) else 0.0
        vmax = float(vms.max()) if len(vms) else 1.0
        label = "Von Mises Stress"
    else:
        imgui.end()
        return

    imgui.text(f"{label}    [{vmin:.3e} — {vmax:.3e}]")
    ticks = legend_ticks(vmin, vmax, 5)
    tl    = "  ".join([f"{t:.2e}" for t in ticks])
    imgui.push_style_color(imgui.Col_.text, (0.7,0.9,0.7,1.0))
    imgui.text(tl)
    imgui.pop_style_color()

    imgui.end()


def draw_node_info(state: ViewerState, width: int, height: int):
    """Floating tooltip with selected node/element info."""
    if state.selected_node < 0 or state.model is None:
        return
    node = state.model.nodes.get(state.selected_node)
    if not node:
        return

    imgui.set_next_window_pos((290, height - 130), imgui.Cond_.always)
    imgui.set_next_window_size((220, 100), imgui.Cond_.always)
    imgui.set_next_window_bg_alpha(0.8)
    flags = imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
    imgui.begin("##nodeinfo", None, flags)

    imgui.push_style_color(imgui.Col_.text, (1.0,0.8,0.4,1.0))
    imgui.text(f"Node {state.selected_node}")
    imgui.pop_style_color()

    imgui.text(f"  X = {node.xyz[0]:.5g}")
    imgui.text(f"  Y = {node.xyz[1]:.5g}")
    imgui.text(f"  Z = {node.xyz[2]:.5g}")

    sc = state.subcase
    if state.results and sc in state.results.displacements:
        d = state.results.displacements[sc].get(state.selected_node)
        if d:
            imgui.text(f"  |u| = {d.magnitude:.5e}")

    imgui.end()
