"""
Test mesh_renderer tanpa ImGui.
"""
import glfw, moderngl, numpy as np, sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

from parser.dat_parser import load_dat
from renderer.mesh_renderer import MeshRenderer
from renderer.camera import Camera

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(1024, 768, "MeshRenderer test - no ImGui", None, None)
glfw.make_context_current(win)
glfw.swap_interval(1)

ctx = moderngl.create_context()

sample = os.path.join(os.path.dirname(__file__), 'samples', 'mixed_demo.dat')
model  = load_dat(sample)
print(f"Loaded: {len(model.nodes)} nodes, {len(model.elements)} elements")

renderer = MeshRenderer(ctx)
renderer.upload(model, display_mode='wireframe')

camera = Camera()
mn, mx = model.bbox()
camera.fit(model.center(), float(np.linalg.norm(mx - mn)) * 0.5)

def on_scroll(win, x, y): camera.on_scroll(y)
def on_mouse(win, btn, act, mod):
    cx,cy = glfw.get_cursor_pos(win)
    camera.on_mouse_button(btn, 1 if act==glfw.PRESS else 0, cx, cy)
def on_move(win, x, y): camera.on_mouse_move(x, y)
def on_key(win, key, sc, act, mod):
    if act == glfw.PRESS:
        if key == glfw.KEY_ESCAPE: glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_F:
            camera.fit(model.center(), float(np.linalg.norm(mx-mn))*0.5)

glfw.set_scroll_callback(win, on_scroll)
glfw.set_mouse_button_callback(win, on_mouse)
glfw.set_cursor_pos_callback(win, on_move)
glfw.set_key_callback(win, on_key)

frame = 0
while not glfw.window_should_close(win):
    glfw.poll_events()
    w, h = glfw.get_framebuffer_size(win)

    ctx.viewport = (0, 0, w, h)
    ctx.scissor  = None
    ctx.clear(0.15, 0.16, 0.20, 1.0)

    aspect = w/h if h > 0 else 1.0
    mvp = camera.mvp(aspect)
    renderer.draw(mvp, display_mode='wireframe')

    glfw.swap_buffers(win)

    if frame == 0:
        pt = np.array([*model.center(), 1.0], dtype=np.float32)
        r  = mvp @ pt
        ndc = r[:2]/r[3] if abs(r[3]) > 1e-6 else [0,0]
        print(f"Frame 0: center NDC={ndc}, viewport={w}x{h}")
    frame += 1

glfw.terminate()
