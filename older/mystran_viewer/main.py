"""
MYSTRAN Viewer
Controls: Left drag=rotate, Right/Mid drag=pan, Scroll=zoom
          F=fit, R=reset, W=wireframe, S=solid, C=contour, Esc=quit
"""
import sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui

from parser.dat_parser import load_dat
from parser.f06_parser import load_f06
from renderer.camera   import Camera
from renderer.mesh_renderer import MeshRenderer
from gui.panels import (ViewerState, draw_menu_bar, draw_left_panel,
                        draw_right_panel, draw_legend, draw_node_info)

def _win_addr(w): return ctypes.cast(w, ctypes.c_void_p).value


class MystranViewerApp:
    def __init__(self):
        self.state      = ViewerState()
        self.camera     = Camera()
        self._window    = None
        self._ctx       = None
        self._mesh_rnd  = None
        self._prev_time = 0.0

    def run(self, dat_file='', f06_file=''):
        if dat_file: self.state.dat_path = dat_file; self.state.request_load_dat = True
        if f06_file: self.state.f06_path = f06_file; self.state.request_load_f06 = True
        self._init_window()
        self._init_gl()
        self._main_loop()
        self._cleanup()

    def _init_window(self):
        if not glfw.init(): raise RuntimeError("GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        self._window = glfw.create_window(1280, 800, "MYSTRAN Viewer", None, None)
        if not self._window: glfw.terminate(); raise RuntimeError("Window failed")
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self._window,   self._on_mouse_move)
        glfw.set_scroll_callback(self._window,       self._on_scroll)
        glfw.set_key_callback(self._window,          self._on_key)

    def _init_gl(self):
        self._ctx = moderngl.create_context()
        # ImGui
        imgui.create_context()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        io.set_ini_filename("")
        self._apply_dark_theme()
        imgui.backends.glfw_init_for_opengl(_win_addr(self._window), True)
        imgui.backends.opengl3_init("#version 330")
        # Renderer
        self._mesh_rnd = MeshRenderer(self._ctx)
        self.camera.reset()

    def _main_loop(self):
        self._prev_time = glfw.get_time()
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            t = glfw.get_time()
            dt = max(t - self._prev_time, 1e-6)
            self._prev_time       = t
            self.state.fps        = 1.0 / dt
            self.state.frame_time = dt

            self._handle_requests()

            w, h = glfw.get_framebuffer_size(self._window)

            # ── 3D scene (same as test_meshrender.py) ─────────────
            self._ctx.viewport = (0, 0, w, h)
            self._ctx.scissor  = None
            self._ctx.clear(0.15, 0.16, 0.20, 1.0)
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.disable(moderngl.BLEND)

            if self.state.model:
                mvp = self.camera.mvp(w / h if h > 0 else 1.0)
                self._mesh_rnd.draw(
                    mvp,
                    display_mode = self.state.display_mode,
                    show_edges   = self.state.show_edges,
                    show_spc     = self.state.show_spc,
                    visible_dims = self.state.visible_dim_tuple(),
                )

            # ── ImGui on top ───────────────────────────────────────
            imgui.backends.glfw_new_frame()
            imgui.backends.opengl3_new_frame()
            imgui.new_frame()
            draw_menu_bar(self.state)
            draw_left_panel(self.state)
            draw_right_panel(self.state, w, h)
            draw_legend(self.state, w, h)
            draw_node_info(self.state, w, h)
            imgui.render()
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

            glfw.swap_buffers(self._window)
            if getattr(self.state, 'request_quit', False): break

    def _handle_requests(self):
        if self.state.request_load_dat and self.state.dat_path:
            self._load_dat(self.state.dat_path); self.state.request_load_dat = False
        if self.state.request_load_f06 and self.state.f06_path:
            self._load_f06(self.state.f06_path); self.state.request_load_f06 = False
        if self.state.request_rebuild and self.state.model:
            self._rebuild(); self.state.request_rebuild = False
        if self.state.request_fit and self.state.model:
            mn, mx = self.state.model.bbox()
            self.camera.fit(self.state.model.center(),
                            float(np.linalg.norm(mx - mn)) * 0.5)
            self.state.request_fit = False

    def _load_dat(self, path):
        try:
            self.state.model = load_dat(path)
            print(f"[DAT] {len(self.state.model.nodes)} nodes, "
                  f"{len(self.state.model.elements)} elements")
            self._rebuild()
            self.state.request_fit = True
        except Exception as e:
            print(f"[DAT] Error: {e}"); import traceback; traceback.print_exc()

    def _load_f06(self, path):
        try:
            self.state.results = load_f06(path)
            sc = self.state.results.subcases
            print(f"[F06] Subcases: {sc}")
            if sc: self.state.subcase = sc[0]
            self._rebuild()
        except Exception as e:
            print(f"[F06] Error: {e}"); import traceback; traceback.print_exc()

    def _rebuild(self):
        if not self.state.model: return
        try:
            self._mesh_rnd.upload(
                self.state.model,
                results      = self.state.results,
                subcase      = self.state.subcase,
                display_mode = self.state.display_mode,
                result_type  = self.state.result_type,
                cmap_name    = self.state.cmap_name,
                deform_scale = self.state.deform_scale,
            )
        except Exception as e:
            print(f"[Renderer] {e}"); import traceback; traceback.print_exc()

    def _cleanup(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()

    def _on_mouse_button(self, win, btn, act, mod):
        if not imgui.get_io().want_capture_mouse:
            x, y = glfw.get_cursor_pos(win)
            self.camera.on_mouse_button(btn, 1 if act==glfw.PRESS else 0, x, y)

    def _on_mouse_move(self, win, x, y):
        if not imgui.get_io().want_capture_mouse:
            self.camera.on_mouse_move(x, y)

    def _on_scroll(self, win, xoff, yoff):
        if not imgui.get_io().want_capture_mouse:
            self.camera.on_scroll(yoff)

    def _on_key(self, win, key, sc, act, mod):
        if act != glfw.PRESS or imgui.get_io().want_capture_keyboard: return
        if   key == glfw.KEY_F:      self.state.request_fit = True
        elif key == glfw.KEY_R:      self.camera.reset()
        elif key == glfw.KEY_W:      self.state.display_mode='wireframe'; self.state.request_rebuild=True
        elif key == glfw.KEY_S:      self.state.display_mode='solid';     self.state.request_rebuild=True
        elif key == glfw.KEY_C:      self.state.display_mode='contour';   self.state.request_rebuild=True
        elif key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)

    def _apply_dark_theme(self):
        style = imgui.get_style()
        style.window_rounding    = 2.0
        style.frame_rounding     = 2.0
        style.grab_rounding      = 2.0
        style.window_border_size = 0.0
        def sc(col, r, g, b, a=1.0):
            style.set_color_(col, imgui.ImVec4(r, g, b, a))
        sc(imgui.Col_.window_bg,          0.13, 0.14, 0.15, 0.92)
        sc(imgui.Col_.frame_bg,           0.20, 0.21, 0.23)
        sc(imgui.Col_.frame_bg_hovered,   0.25, 0.26, 0.28)
        sc(imgui.Col_.frame_bg_active,    0.30, 0.31, 0.33)
        sc(imgui.Col_.title_bg,           0.10, 0.11, 0.12)
        sc(imgui.Col_.title_bg_active,    0.15, 0.16, 0.18)
        sc(imgui.Col_.menu_bar_bg,        0.10, 0.11, 0.12)
        sc(imgui.Col_.button,             0.25, 0.40, 0.60)
        sc(imgui.Col_.button_hovered,     0.35, 0.50, 0.75)
        sc(imgui.Col_.button_active,      0.20, 0.35, 0.55)
        sc(imgui.Col_.header,             0.25, 0.40, 0.60, 0.7)
        sc(imgui.Col_.header_hovered,     0.30, 0.45, 0.65, 0.8)
        sc(imgui.Col_.check_mark,         0.40, 0.85, 0.40)
        sc(imgui.Col_.slider_grab,        0.35, 0.55, 0.80)
        sc(imgui.Col_.slider_grab_active, 0.45, 0.65, 0.90)
        sc(imgui.Col_.text,               0.90, 0.90, 0.90)
        sc(imgui.Col_.separator,          0.30, 0.30, 0.35)
        sc(imgui.Col_.popup_bg,           0.10, 0.11, 0.12, 0.95)


if __name__ == "__main__":
    dat = f06 = ""
    for a in sys.argv[1:]:
        al = a.lower()
        if   al.endswith(('.dat','.nas','.bdf')): dat = a
        elif al.endswith('.f06'):                 f06 = a
    if not dat:
        s = os.path.join(os.path.dirname(__file__), 'samples', 'mixed_demo.dat')
        if os.path.exists(s): dat = s; print(f"[Info] Loading sample: {s}")
    MystranViewerApp().run(dat_file=dat, f06_file=f06)
