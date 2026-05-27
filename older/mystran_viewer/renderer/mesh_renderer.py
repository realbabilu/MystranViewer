"""
MeshRenderer: builds and draws OpenGL geometry for MYSTRAN models.

Draw modes:
  - wireframe       : line edges only (gray)
  - solid           : flat shaded filled + edges
  - contour_disp    : filled with displacement colormap
  - contour_stress  : filled with von Mises colormap

Element → face decomposition:
  CBAR/CBEAM/CROD : 1 line (2 nodes)
  CTRIA3          : 1 triangle
  CQUAD4          : 2 triangles
  CTETRA          : 4 triangular faces
  CPENTA          : 2 tri + 3 quad faces
  CHEXA           : 6 quad faces (each split to 2 tri)
"""

import numpy as np
import moderngl
from typing import Optional, Dict, Tuple

from parser.dat_parser import MystranModel
from parser.f06_parser  import F06Results
from renderer.contour   import map_values, COLORMAPS


# ---------------------------------------------------------------------------
# GLSL shaders (OpenGL 3.3 core)
# ---------------------------------------------------------------------------

VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_color;
in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_pos;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color  = in_color;
    v_normal = normalize(mat3(u_model) * in_normal);
    v_pos    = (u_model * vec4(in_position, 1.0)).xyz;
}
"""

FRAG_SHADER = """
#version 330 core

in vec3 v_color;
in vec3 v_normal;
in vec3 v_pos;

uniform vec3  u_light_dir;
uniform float u_ambient;
uniform int   u_flat;       // 1=no lighting, 0=phong

out vec4 frag_color;

void main() {
    vec3 col = v_color;
    if (u_flat == 0) {
        float diff = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
        col = col * (u_ambient + (1.0 - u_ambient) * diff);
    }
    frag_color = vec4(col, 1.0);
}
"""

# Line shader (simpler)
VERT_LINE = """
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

FRAG_LINE = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""


# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
C_WIRE      = np.array([0.3, 0.8, 0.3],  dtype=np.float32)   # bright green
C_SOLID     = np.array([0.55,0.65,0.75], dtype=np.float32)   # steel blue-gray
C_EDGE      = np.array([0.15,0.15,0.15], dtype=np.float32)   # dark edge
C_SPC       = np.array([1.0, 0.3, 0.1],  dtype=np.float32)   # orange
C_FORCE     = np.array([1.0, 1.0, 0.0],  dtype=np.float32)   # yellow
C_NODE      = np.array([0.9, 0.9, 0.9],  dtype=np.float32)   # light gray


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _face_normal(v0, v1, v2) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    l = np.linalg.norm(n)
    return (n / l).astype(np.float32) if l > 1e-30 else np.array([0,0,1], dtype=np.float32)


def _build_face_tris(nodes_xyz: Dict[int, np.ndarray],
                     elem_nodes: list,
                     etype: str):
    """
    Decompose element into triangles.
    Returns list of (v0, v1, v2) vertex triples as np.ndarray (3,).
    """
    def g(nid): return nodes_xyz.get(nid, np.zeros(3))

    tris = []
    if etype in ('CBAR','CBEAM','CROD'):
        return []   # handled separately as lines

    elif etype == 'CTRIA3':
        n = elem_nodes
        tris.append((g(n[0]), g(n[1]), g(n[2])))

    elif etype == 'CQUAD4':
        n = elem_nodes
        tris.append((g(n[0]), g(n[1]), g(n[2])))
        tris.append((g(n[0]), g(n[2]), g(n[3])))

    elif etype == 'CTETRA':
        n = elem_nodes
        faces = [(0,1,2),(0,1,3),(1,2,3),(0,2,3)]
        for f in faces:
            tris.append((g(n[f[0]]), g(n[f[1]]), g(n[f[2]])))

    elif etype == 'CPENTA':
        n = elem_nodes  # 6 nodes
        # bottom tri, top tri
        tris.append((g(n[0]), g(n[1]), g(n[2])))
        tris.append((g(n[3]), g(n[5]), g(n[4])))
        # 3 quad sides split to tris
        sides = [(0,3,4,1),(1,4,5,2),(0,2,5,3)]
        for s in sides:
            tris.append((g(n[s[0]]), g(n[s[1]]), g(n[s[2]])))
            tris.append((g(n[s[0]]), g(n[s[2]]), g(n[s[3]])))

    elif etype == 'CHEXA':
        n = elem_nodes  # 8 nodes
        # 6 faces, each quad split to 2 tris
        faces = [
            (0,1,2,3),(4,7,6,5),(0,4,5,1),
            (1,5,6,2),(2,6,7,3),(3,7,4,0)
        ]
        for f in faces:
            tris.append((g(n[f[0]]), g(n[f[1]]), g(n[f[2]])))
            tris.append((g(n[f[0]]), g(n[f[2]]), g(n[f[3]])))

    return tris


def _build_edge_lines(nodes_xyz: Dict[int, np.ndarray],
                      elem_nodes: list, etype: str):
    """Extract unique edges for wireframe rendering."""
    def g(nid): return nodes_xyz.get(nid, np.zeros(3))

    if etype in ('CBAR','CBEAM','CROD'):
        n = elem_nodes
        return [(g(n[0]), g(n[1]))]

    elif etype == 'CTRIA3':
        n = elem_nodes
        return [(g(n[0]),g(n[1])),(g(n[1]),g(n[2])),(g(n[2]),g(n[0]))]

    elif etype == 'CQUAD4':
        n = elem_nodes
        return [(g(n[0]),g(n[1])),(g(n[1]),g(n[2])),(g(n[2]),g(n[3])),(g(n[3]),g(n[0]))]

    elif etype == 'CTETRA':
        n = elem_nodes
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        return [(g(n[a]),g(n[b])) for a,b in pairs]

    elif etype == 'CPENTA':
        n = elem_nodes
        pairs = [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
        return [(g(n[a]),g(n[b])) for a,b in pairs]

    elif etype == 'CHEXA':
        n = elem_nodes
        pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        return [(g(n[a]),g(n[b])) for a,b in pairs]

    return []


# ---------------------------------------------------------------------------
# MeshRenderer
# ---------------------------------------------------------------------------

class MeshRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._prog_solid = ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER
        )
        self._prog_line = ctx.program(
            vertex_shader=VERT_LINE,
            fragment_shader=FRAG_LINE
        )
        # Buffers
        self._vao_solid  = None
        self._vao_edge   = None
        self._vao_spc    = None
        self._vao_node   = None

        self._n_solid = 0
        self._n_edge  = 0
        self._n_spc   = 0
        self._n_node  = 0

        # Per-triangle element ids (for contour lookup)
        self._tri_elem_ids: np.ndarray = np.array([], dtype=np.int32)

        self._model: Optional[MystranModel] = None

    def upload(self, model: MystranModel,
               results: Optional[F06Results] = None,
               subcase: int = 1,
               display_mode: str = 'wireframe',
               result_type: str = 'displacement',
               cmap_name: str = 'rainbow',
               deform_scale: float = 0.0):
        """
        Build GPU buffers from model + optional results.
        display_mode: 'wireframe' | 'solid' | 'contour'
        result_type : 'displacement' | 'von_mises'
        deform_scale: deformed shape multiplier (0=undeformed)
        """
        self._model = model
        nodes  = model.nodes
        elems  = model.elements

        # Build node position dict (possibly deformed)
        pos: Dict[int, np.ndarray] = {}
        for nid, node in nodes.items():
            p = node.xyz.copy()
            if deform_scale > 0 and results and subcase in results.displacements:
                dr = results.displacements[subcase].get(nid)
                if dr:
                    p = p + dr.translation * deform_scale
            pos[nid] = p

        # ---- Build solid / contour triangles ----
        tri_verts  = []   # (x,y,z) per vertex
        tri_colors = []   # (r,g,b) per vertex
        tri_norms  = []   # (nx,ny,nz) per vertex
        tri_eids   = []   # element id per triangle

        for eid, elem in elems.items():
            tris = _build_face_tris(pos, elem.nodes, elem.type)
            for (v0, v1, v2) in tris:
                n = _face_normal(v0, v1, v2)
                for v in (v0, v1, v2):
                    tri_verts.append(v)
                    tri_norms.append(n)
                    tri_colors.append(C_SOLID)
                tri_eids.append(eid)

        # Apply contour colors if needed
        if display_mode == 'contour' and results:
            if result_type == 'displacement' and subcase in results.displacements:
                disp = results.displacements[subcase]
                mags = {nid: np.linalg.norm(d.translation) for nid, d in disp.items()}
                vals_all = list(mags.values())
                vmin = min(vals_all) if vals_all else 0.0
                vmax = max(vals_all) if vals_all else 1.0

                vi = 0
                for eid, elem in elems.items():
                    tris = _build_face_tris(pos, elem.nodes, elem.type)
                    for (v0, v1, v2) in tris:
                        for nid_ref, v in zip(elem.nodes[:3], (v0,v1,v2)):
                            m = mags.get(nid_ref, 0.0)
                            t = (m - vmin)/(vmax-vmin+1e-30)
                            cmap = list(COLORMAPS.get(cmap_name, COLORMAPS['rainbow']))
                            ci = int(np.clip(t * 255, 0, 255))
                            tri_colors[vi] = cmap[ci]
                            vi += 1

            elif result_type == 'von_mises' and subcase in results.stresses:
                stress = results.stresses[subcase]
                vals_all = [s.von_mises for s in stress.values()]
                vmin = min(vals_all) if vals_all else 0.0
                vmax = max(vals_all) if vals_all else 1.0
                cmap = list(COLORMAPS.get(cmap_name, COLORMAPS['rainbow']))

                vi = 0
                for eid, elem in elems.items():
                    tris = _build_face_tris(pos, elem.nodes, elem.type)
                    for _ in tris:
                        s = stress.get(eid)
                        m = s.von_mises if s else 0.0
                        t = (m - vmin)/(vmax-vmin+1e-30)
                        ci = int(np.clip(t * 255, 0, 255))
                        for _ in range(3):
                            tri_colors[vi] = cmap[ci]
                            vi += 1

        self._tri_elem_ids = np.array(tri_eids, dtype=np.int32)

        # Upload solid buffer
        if tri_verts:
            v_arr = np.array(tri_verts,  dtype=np.float32).flatten()
            c_arr = np.array(tri_colors, dtype=np.float32).flatten()
            n_arr = np.array(tri_norms,  dtype=np.float32).flatten()

            vbo_v = self.ctx.buffer(v_arr.tobytes())
            vbo_c = self.ctx.buffer(c_arr.tobytes())
            vbo_n = self.ctx.buffer(n_arr.tobytes())

            self._vao_solid = self.ctx.vertex_array(
                self._prog_solid,
                [(vbo_v, '3f', 'in_position'),
                 (vbo_c, '3f', 'in_color'),
                 (vbo_n, '3f', 'in_normal')],
            )
            self._n_solid = len(tri_verts)
        else:
            self._vao_solid = None
            self._n_solid = 0

        # ---- Build edge lines ----
        edge_verts  = []
        edge_colors = []

        for eid, elem in elems.items():
            edges = _build_edge_lines(pos, elem.nodes, elem.type)
            for (v0, v1) in edges:
                edge_verts.extend([v0, v1])
                edge_colors.extend([C_WIRE, C_WIRE])

        if edge_verts:
            ev = np.array(edge_verts,  dtype=np.float32).flatten()
            ec = np.array(edge_colors, dtype=np.float32).flatten()
            vbo_ev = self.ctx.buffer(ev.tobytes())
            vbo_ec = self.ctx.buffer(ec.tobytes())
            self._vao_edge = self.ctx.vertex_array(
                self._prog_line,
                [(vbo_ev, '3f', 'in_position'),
                 (vbo_ec, '3f', 'in_color')],
            )
            self._n_edge = len(edge_verts)
        else:
            self._vao_edge = None
            self._n_edge = 0

        # ---- SPC markers (cross at constrained nodes) ----
        spc_verts  = []
        spc_colors = []
        s = model.scale() * 0.02
        spc_nodes = {spc.node_id for spc in model.spcs}
        for nid in spc_nodes:
            p = pos.get(nid)
            if p is None: continue
            for axis in [np.array([s,0,0]), np.array([0,s,0]), np.array([0,0,s])]:
                spc_verts.extend([p - axis, p + axis])
                spc_colors.extend([C_SPC, C_SPC])

        if spc_verts:
            sv = np.array(spc_verts,  dtype=np.float32).flatten()
            sc = np.array(spc_colors, dtype=np.float32).flatten()
            vbo_sv = self.ctx.buffer(sv.tobytes())
            vbo_sc = self.ctx.buffer(sc.tobytes())
            self._vao_spc = self.ctx.vertex_array(
                self._prog_line,
                [(vbo_sv, '3f', 'in_position'),
                 (vbo_sc, '3f', 'in_color')],
            )
            self._n_spc = len(spc_verts)
        else:
            self._vao_spc = None
            self._n_spc = 0

    # -----------------------------------------------------------------------
    def draw(self, mvp: np.ndarray, display_mode: str = 'wireframe',
             show_edges: bool = True, show_spc: bool = True):
        """Render the model."""
        mvp_bytes = mvp.astype(np.float32).flatten().tobytes()
        model_mat = np.eye(4, dtype=np.float32).flatten().tobytes()
        light_dir = np.array([0.5, 1.0, 0.8], dtype=np.float32)

        # Solid / contour pass
        if display_mode in ('solid', 'contour') and self._vao_solid and self._n_solid > 0:
            p = self._prog_solid
            p['u_mvp'].write(mvp_bytes)
            p['u_model'].write(model_mat)
            p['u_light_dir'].value = tuple(light_dir)
            p['u_ambient'].value   = 0.35
            p['u_flat'].value      = 0

            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
            self._vao_solid.render(moderngl.TRIANGLES)
            self.ctx.disable(moderngl.CULL_FACE)

        # Edge lines
        if (display_mode == 'wireframe' or show_edges) and self._vao_edge and self._n_edge > 0:
            p = self._prog_line
            p['u_mvp'].write(mvp_bytes)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self._vao_edge.render(moderngl.LINES)

        # SPC markers
        if show_spc and self._vao_spc and self._n_spc > 0:
            p = self._prog_line
            p['u_mvp'].write(mvp_bytes)
            self.ctx.line_width = 2.0
            self._vao_spc.render(moderngl.LINES)
            self.ctx.line_width = 1.0
