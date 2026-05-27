"""
Test: apakah imgui opengl3 backend merusak GL state untuk render berikutnya?
"""
import glfw, moderngl, numpy as np, sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))
from imgui_bundle import imgui

def win_addr(w): return ctypes.cast(w, ctypes.c_void_p).value

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(800,600,"ImGui state test",None,None)
glfw.make_context_current(win)
glfw.swap_interval(1)

ctx = moderngl.create_context()
imgui.create_context()
io = imgui.get_io(); io.set_ini_filename("")
imgui.backends.glfw_init_for_opengl(win_addr(win), True)
imgui.backends.opengl3_init("#version 330")

prog = ctx.program(
    vertex_shader="  #version 330 core\nin vec2 p;in vec3 c;out vec3 vc;\nvoid main(){gl_Position=vec4(p,0,1);vc=c;}",
    fragment_shader="#version 330 core\nin vec3 vc;out vec4 f;\nvoid main(){f=vec4(vc,1);}"
)
data = np.array([
    -0.8,-0.1, 1,0,0,  0.8,-0.1, 1,0,0,   # red horizontal
    -0.8, 0.1, 0,1,0,  0.8, 0.1, 0,1,0,   # green horizontal
    -0.5,-0.8, 0,0,1,  0.5, 0.8, 0,0,1,   # blue diagonal
], dtype=np.float32)
vbo = ctx.buffer(data.tobytes())
vao = ctx.vertex_array(prog, [(vbo,'2f 3f','p','c')])

frame = 0
mode = 0  # 0=3D only, 1=imgui only, 2=imgui then 3D, 3=3D then imgui
labels = ["3D only (baseline)", "ImGui only", "ImGui THEN 3D", "3D THEN ImGui (correct)"]

while not glfw.window_should_close(win):
    glfw.poll_events()
    w,h = glfw.get_framebuffer_size(win)

    # Change mode every 120 frames
    mode = (frame // 120) % 4

    ctx.viewport = (0,0,w,h)
    ctx.scissor  = None

    if mode == 0:
        # Pure 3D, no imgui
        ctx.clear(0.05,0.05,0.15,1.0)
        vao.render(moderngl.LINES)

    elif mode == 1:
        # Pure ImGui
        imgui.backends.glfw_new_frame()
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        imgui.begin("test"); imgui.text(f"Mode {mode}: {labels[mode]}"); imgui.end()
        ctx.clear(0.05,0.10,0.05,1.0)
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    elif mode == 2:
        # ImGui THEN 3D (wrong order)
        imgui.backends.glfw_new_frame()
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        imgui.begin("test"); imgui.text(f"Mode {mode}: {labels[mode]}"); imgui.end()
        ctx.clear(0.15,0.05,0.05,1.0)
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        # Now try to draw 3D AFTER imgui
        ctx.scissor = None
        vao.render(moderngl.LINES)

    elif mode == 3:
        # 3D THEN ImGui (correct order)
        ctx.clear(0.05,0.05,0.20,1.0)
        vao.render(moderngl.LINES)
        imgui.backends.glfw_new_frame()
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        imgui.begin("test"); imgui.text(f"Mode {mode}: {labels[mode]}"); imgui.end()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    glfw.swap_buffers(win)

    if frame % 120 == 0:
        print(f"Mode {mode}: {labels[mode]} - do you see colored lines?")
    frame += 1

imgui.backends.opengl3_shutdown()
imgui.backends.glfw_shutdown()
imgui.destroy_context()
glfw.terminate()
