"""
Isolate exact VAO creation that works vs tidak.
"""
import glfw, moderngl, numpy as np, sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

glfw.init()
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
win = glfw.create_window(800,600,"VAO test",None,None)
glfw.make_context_current(win)
ctx = moderngl.create_context()

# Identical to test_bare
VERT = """
#version 330 core
in vec2 p; in vec3 c; out vec3 vc;
void main(){ gl_Position=vec4(p,0,1); vc=c; }
"""
FRAG = """
#version 330 core
in vec3 vc; out vec4 f;
void main(){ f=vec4(vc,1); }
"""

prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
data = np.array([
    -0.8,-0.1, 1,0,0,  0.8,-0.1, 1,0,0,
    -0.8, 0.1, 0,1,0,  0.8, 0.1, 0,1,0,
], dtype=np.float32)

print("Test A: single VBO interleaved (same as test_bare)")
vbo_a = ctx.buffer(data.tobytes())
vao_a = ctx.vertex_array(prog, [(vbo_a,'2f 3f','p','c')])

print("Test B: separate VBOs")
pos_data = data.reshape(-1,5)[:,0:2].copy()
col_data = data.reshape(-1,5)[:,2:5].copy()
vbo_p = ctx.buffer(pos_data.astype(np.float32).tobytes())
vbo_c = ctx.buffer(col_data.astype(np.float32).tobytes())
vao_b = ctx.vertex_array(prog, [
    (vbo_p, '2f', 'p'),
    (vbo_c, '3f', 'c'),
])

print("Test C: separate VBOs with 3f pos (vec3 instead of vec2)")
VERT3 = """
#version 330 core
in vec3 in_position; in vec3 in_color; out vec3 vc;
void main(){ gl_Position=vec4(in_position,1); vc=in_color; }
"""
prog3 = ctx.program(vertex_shader=VERT3, fragment_shader=FRAG)
data3 = np.array([
    -0.8,-0.1,0, 1,0,0,  0.8,-0.1,0, 1,0,0,
    -0.8, 0.1,0, 0,1,0,  0.8, 0.1,0, 0,1,0,
], dtype=np.float32)
vbo3 = ctx.buffer(data3.tobytes())
vao_c = ctx.vertex_array(prog3, [(vbo3,'3f 3f','in_position','in_color')])

# Which test shows lines?
tests = [('A: interleaved 2f+3f', vao_a),
         ('B: separate VBOs',     vao_b),
         ('C: interleaved 3f+3f', vao_c)]

frame = 0
while not glfw.window_should_close(win):
    glfw.poll_events()
    w,h = glfw.get_framebuffer_size(win)
    ctx.viewport = (0,0,w,h)
    ctx.scissor  = None

    mode = (frame // 180) % len(tests)
    label, vao = tests[mode]

    ctx.clear(0.1, 0.1, 0.15, 1.0)
    vao.render(moderngl.LINES)
    glfw.swap_buffers(win)

    if frame % 180 == 0:
        print(f"Showing {label} - do you see 2 horizontal lines?")
    frame += 1
    if frame > 180 * len(tests) + 30:
        break

glfw.terminate()
print("Done")
