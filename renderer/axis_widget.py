"""
Axis orientation widget — drawn in a small corner viewport.
Shows RGB XYZ triad that rotates with the camera.

Also provides ViewCube face-click presets (+X, -X, +Y, -Y, +Z, -Z, ISO).
"""

import numpy as np
import moderngl
import pyrr
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""
FRAG = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() { frag_color = vec4(v_color, 1.0); }
"""

# ---------------------------------------------------------------------------
_AXES = np.array([
    # X axis  red
    [0,0,0], [1,0,0],
    # Y axis  green
    [0,0,0], [0,1,0],
    # Z axis  blue
    [0,0,0], [0,0,1],
], dtype=np.float32)

_AXIS_COLORS = np.array([
    [0.95,0.25,0.25],[0.95,0.25,0.25],
    [0.25,0.90,0.25],[0.25,0.90,0.25],
    [0.30,0.55,1.00],[0.30,0.55,1.00],
], dtype=np.float32)

# Arrowhead cones — 6 tris per axis (simplified as a small square cap)
def _cone_tris(tip, base_center, color, r=0.08, n=8):
    """Return vertices+colors for a cone cap."""
    verts = []
    cols  = []
    axis  = tip - base_center
    L = np.linalg.norm(axis)
    if L < 1e-9:
        return verts, cols
    ax = axis / L
    perp = np.array([0,1,0],dtype=np.float32) if abs(ax[1]) < 0.9 else np.array([1,0,0],dtype=np.float32)
    ey = np.cross(ax, perp); ey /= np.linalg.norm(ey)
    ez = np.cross(ax, ey);   ez /= np.linalg.norm(ez)
    ring = [base_center + r*(np.cos(2*np.pi*i/n)*ey + np.sin(2*np.pi*i/n)*ez)
            for i in range(n)]
    for i in range(n):
        j = (i+1)%n
        verts += [tip, ring[i], ring[j]]
        cols  += [color, color*0.6, color*0.6]
    return verts, cols


class AxisWidget:
    SIZE = 90   # viewport pixels

    def __init__(self, ctx: moderngl.Context):
        self.ctx  = ctx
        self._prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self._build_buffers()

    def _build_buffers(self):
        # Axis lines
        vbo_v = self.ctx.buffer(_AXES.tobytes())
        vbo_c = self.ctx.buffer(_AXIS_COLORS.tobytes())
        self._vao_lines = self.ctx.vertex_array(self._prog, [
            (vbo_v, '3f', 'in_position'),
            (vbo_c, '3f', 'in_color'),
        ])
        self._n_lines = 6

        # Arrow cones
        colors = [np.array([0.95,0.25,0.25]),
                  np.array([0.25,0.90,0.25]),
                  np.array([0.30,0.55,1.00])]
        tips   = [np.array([1,0,0],dtype=np.float32),
                  np.array([0,1,0],dtype=np.float32),
                  np.array([0,0,1],dtype=np.float32)]
        bases  = [np.array([0.82,0,0],dtype=np.float32),
                  np.array([0,0.82,0],dtype=np.float32),
                  np.array([0,0,0.82],dtype=np.float32)]

        all_v, all_c = [], []
        for tip, base, col in zip(tips, bases, colors):
            v, c = _cone_tris(tip.astype(np.float32), base, col.astype(np.float32))
            all_v += v; all_c += c

        if all_v:
            av = np.array(all_v, dtype=np.float32).flatten()
            ac = np.array(all_c, dtype=np.float32).flatten()
            vbo_av = self.ctx.buffer(av.tobytes())
            vbo_ac = self.ctx.buffer(ac.tobytes())
            self._vao_cones = self.ctx.vertex_array(self._prog, [
                (vbo_av, '3f', 'in_position'),
                (vbo_ac, '3f', 'in_color'),
            ])
            self._n_cones = len(all_v)
        else:
            self._vao_cones = None
            self._n_cones = 0

    def draw(self, win_w: int, win_h: int,
             view_matrix: np.ndarray,
             corner: str = 'bottom_left'):
        """Render axis triad in a corner viewport."""
        s = self.SIZE
        if corner == 'bottom_left':
            vx, vy = 0, 0
        elif corner == 'bottom_right':
            vx, vy = win_w - s, 0
        elif corner == 'top_right':
            vx, vy = win_w - s, win_h - s
        else:
            vx, vy = 0, win_h - s

        # Extract rotation only from view matrix
        rot = np.array(view_matrix, dtype=np.float32)
        rot[3, :3] = 0; rot[:3, 3] = 0; rot[3, 3] = 1  # strip translation

        # Ortho projection for the triad
        proj = pyrr.matrix44.create_orthogonal_projection_matrix(
            -1.4, 1.4, -1.4, 1.4, -10, 10, dtype=np.float32)
        mvp = pyrr.matrix44.multiply(rot, proj)

        # Render in corner viewport
        prev_vp = self.ctx.viewport
        self.ctx.viewport = (vx, vy, s, s)
        self.ctx.clear(depth=1.0)   # clear depth only

        self._prog['u_mvp'].write(mvp.astype(np.float32).flatten(order='F').tobytes())
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.line_width = 2.5
        self._vao_lines.render(moderngl.LINES)
        if self._vao_cones and self._n_cones > 0:
            self._vao_cones.render(moderngl.TRIANGLES)
        self.ctx.line_width = 1.0

        self.ctx.viewport = prev_vp

    def hit_test(self, mx: float, my: float,
                 win_w: int, win_h: int,
                 corner: str = 'bottom_left') -> Optional[str]:
        """
        Check if mouse (mx, my) is inside the axis widget.
        Returns face name if inside, else None.
        Only checks overall bounding box — face selection is handled
        by ViewCube logic in camera.
        """
        s = self.SIZE
        if corner == 'bottom_left':
            rx, ry = 0, win_h - s
        elif corner == 'bottom_right':
            rx, ry = win_w - s, win_h - s
        elif corner == 'top_right':
            rx, ry = win_w - s, 0
        else:
            rx, ry = 0, 0

        if rx <= mx <= rx + s and ry <= my <= ry + s:
            return 'widget'
        return None
