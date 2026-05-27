"""
Test tanpa ImGui sama sekali - pure OpenGL + GLFW.
Jika ini muncul, masalah ada di ImGui state management.
"""
import glfw, moderngl, numpy as np, sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(800, 600, "BARE GL TEST - no ImGui", None, None)
glfw.make_context_current(win)

ctx = moderngl.create_context()
prog = ctx.program(
    vertex_shader="""
    #version 330 core
    in vec2 p; in vec3 c; out vec3 vc;
    void main(){ gl_Position=vec4(p,0,1); vc=c; }
    """,
    fragment_shader="""
    #version 330 core
    in vec3 vc; out vec4 f;
    void main(){ f=vec4(vc,1); }
    """
)
data = np.array([
    -0.8,-0.8, 1,0,0,  0.8,-0.8, 1,0,0,
    -0.8,-0.8, 0,1,0, -0.8, 0.8, 0,1,0,
     0.0, 0.8, 0,0,1,  0.8, 0.0, 0,0,1,
], dtype=np.float32)
vbo = ctx.buffer(data.tobytes())
vao = ctx.vertex_array(prog, [(vbo,'2f 3f','p','c')])

frame = 0
while not glfw.window_should_close(win) and frame < 300:
    glfw.poll_events()
    w,h = glfw.get_framebuffer_size(win)
    ctx.viewport = (0,0,w,h)
    ctx.clear(0.1,0.1,0.2,1.0)
    vao.render(moderngl.LINES)
    glfw.swap_buffers(win)
    if frame == 0: print(f"Frame 0: viewport={w}x{h} - do you see colored lines?")
    frame += 1

glfw.terminate()
print("Done")
