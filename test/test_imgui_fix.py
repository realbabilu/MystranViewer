"""
Test ImGui + 3D dengan proper state restore.
"""
import glfw, moderngl, numpy as np, sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))
from imgui_bundle import imgui

from parser.dat_parser import load_dat
from renderer.mesh_renderer import MeshRenderer
from renderer.camera import Camera

def win_addr(w): return ctypes.cast(w, ctypes.c_void_p).value

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(1280, 800, "ImGui+3D fix test", None, None)
glfw.make_context_current(win)
glfw.swap_interval(1)

ctx = moderngl.create_context()

# ImGui init
imgui.create_context()
io = imgui.get_io(); io.set_ini_filename("")
imgui.backends.glfw_init_for_opengl(win_addr(win), True)
imgui.backends.opengl3_init("#version 330")

# Model
sample = os.path.join(os.path.dirname(__file__), 'samples', 'mixed_demo.dat')
model  = load_dat(sample)
renderer = MeshRenderer(ctx)
renderer.upload(model, display_mode='wireframe')
camera = Camera()
mn, mx = model.bbox()
camera.fit(model.center(), float(np.linalg.norm(mx-mn))*0.5)

def on_scroll(win,x,y):
    if not imgui.get_io().want_capture_mouse: camera.on_scroll(y)
def on_mouse(win,btn,act,mod):
    if not imgui.get_io().want_capture_mouse:
        cx,cy=glfw.get_cursor_pos(win)
        camera.on_mouse_button(btn,1 if act==glfw.PRESS else 0,cx,cy)
def on_move(win,x,y):
    if not imgui.get_io().want_capture_mouse: camera.on_mouse_move(x,y)

glfw.set_scroll_callback(win, on_scroll)
glfw.set_mouse_button_callback(win, on_mouse)
glfw.set_cursor_pos_callback(win, on_move)

frame = 0
while not glfw.window_should_close(win):
    glfw.poll_events()
    w,h = glfw.get_framebuffer_size(win)

    # === 3D RENDER — full state reset before ===
    ctx.scissor        = None
    ctx.viewport       = (0, 0, w, h)
    ctx.fbo.clear(0.15, 0.16, 0.20, 1.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    ctx.enable_only(moderngl.DEPTH_TEST)   # clear ALL other flags

    aspect = w/h if h>0 else 1.0
    mvp = camera.mvp(aspect)
    renderer.draw(mvp, display_mode='wireframe')

    # === IMGUI on top ===
    imgui.backends.glfw_new_frame()
    imgui.backends.opengl3_new_frame()
    imgui.new_frame()
    imgui.begin("Info")
    imgui.text(f"Frame {frame}")
    imgui.text(f"Nodes: {len(model.nodes)}")
    imgui.text(f"Elems: {len(model.elements)}")
    imgui.end()
    imgui.render()
    imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    glfw.swap_buffers(win)
    if frame==0: print(f"Frame 0 rendered, viewport={w}x{h}")
    frame+=1

imgui.backends.opengl3_shutdown()
imgui.backends.glfw_shutdown()
imgui.destroy_context()
glfw.terminate()
