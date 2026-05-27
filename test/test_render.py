"""
Minimal render test - draws a colored triangle.
If you see a RED triangle on dark background = rendering works.
If screen is black = fundamental GL/window issue.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glfw
import moderngl
import numpy as np
from imgui_bundle import imgui
import ctypes

def _win_addr(win):
    return ctypes.cast(win, ctypes.c_void_p).value

VERT = """
#version 330 core
in vec2 in_pos;
in vec3 in_col;
out vec3 v_col;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_col = in_col;
}
"""
FRAG = """
#version 330 core
in vec3 v_col;
out vec4 frag;
void main() { frag = vec4(v_col, 1.0); }
"""

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(800, 600, "RENDER TEST - should see red triangle", None, None)
glfw.make_context_current(win)
glfw.swap_interval(1)

ctx = moderngl.create_context()

# Init imgui
imgui.create_context()
io = imgui.get_io()
io.set_ini_filename("")
imgui.backends.glfw_init_for_opengl(_win_addr(win), True)
imgui.backends.opengl3_init("#version 330")

# Simple triangle
prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
vertices = np.array([
    # x,    y,    r,   g,   b
    -0.5, -0.5,  1.0, 0.0, 0.0,
     0.5, -0.5,  0.0, 1.0, 0.0,
     0.0,  0.5,  0.0, 0.0, 1.0,
], dtype=np.float32)

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(prog, [
    (vbo, '2f 3f', 'in_pos', 'in_col'),
])

print("Window open. Should see RGB triangle.")
print("Press Ctrl+C or close window to exit.")

frame = 0
while not glfw.window_should_close(win):
    glfw.poll_events()

    # ImGui frame
    imgui.backends.glfw_new_frame()
    imgui.backends.opengl3_new_frame()
    imgui.new_frame()
    imgui.begin("Test")
    imgui.text(f"Frame {frame}")
    imgui.end()
    imgui.end_frame()

    # 3D
    ctx.clear(0.1, 0.1, 0.15, 1.0)
    w, h = glfw.get_framebuffer_size(win)
    ctx.viewport = (0, 0, w, h)
    vao.render(moderngl.TRIANGLES)

    # ImGui
    imgui.render()
    imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    glfw.swap_buffers(win)
    frame += 1
    if frame == 1:
        print(f"Frame 1 rendered. viewport={w}x{h}")

imgui.backends.opengl3_shutdown()
imgui.backends.glfw_shutdown()
imgui.destroy_context()
glfw.terminate()
