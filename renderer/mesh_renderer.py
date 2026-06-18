"""
MeshRenderer — OpenGL 3.3 Core
Clean version: no module-level numpy, no alpha shader complexity.
Wireframe = colored lines only.
Hidden/solid = Phong-shaded solid + dark edges.
"""

import numpy as np
import moderngl
import colorsys
from typing import Optional, Dict
from parser.dat_parser import MystranModel
from parser.beam_section import section_area_inertias
from parser.f06_parser  import F06Results
from renderer.beam_diagram import (
    _transverse_pload_components,
    _timoshenko_transverse_from_loads,
    _concentrated_pload_components,
    _beam_end_force_components,
    _double_integrate_point_loads,
)
from parser.beam_stress import pbeam_cdef_points
from renderer.contour   import COLORMAPS

# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------
VERT_SOLID = """
#version 330 core
in vec3 in_position; in vec3 in_color; in vec3 in_normal;
uniform mat4 u_mvp;
out vec3 v_color; out vec3 v_normal;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color; v_normal = in_normal;
}
"""
FRAG_SOLID = """
#version 330 core
in vec3 v_color; in vec3 v_normal;
uniform vec3 u_light0, u_light1, u_light2;
uniform float u_ambient;
out vec4 frag_color;
void main() {
    vec3 n = normalize(v_normal);
    float d = max(dot(n,u_light0),0.0)
            + max(dot(n,u_light1),0.0)*0.35
            + max(dot(n,u_light2),0.0)*0.15;
    float lit = clamp(u_ambient + (1.0-u_ambient)*d, 0.0, 1.0);
    frag_color = vec4(v_color*lit, 1.0);
}
"""
# Alpha fill shader for wireframe mode
VERT_FILL = """
#version 330 core
in vec3 in_position; in vec3 in_color; in vec3 in_normal;
uniform mat4 u_mvp;
out vec3 v_color; out vec3 v_normal;
void main() {
    gl_Position = u_mvp * vec4(in_position,1.0);
    v_color=in_color; v_normal=in_normal;
}
"""
FRAG_FILL = """
#version 330 core
in vec3 v_color; in vec3 v_normal;
uniform float u_alpha;
out vec4 frag_color;
void main() {
    float d = abs(dot(normalize(v_normal), vec3(0.5,0.8,0.6)));
    float lit = 0.4 + 0.6*d;
    frag_color = vec4(v_color*lit, u_alpha);
}
"""

VERT_LINE = """
#version 330 core
in vec3 in_position; in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() { gl_Position=u_mvp*vec4(in_position,1.0); v_color=in_color; }
"""
FRAG_LINE = """
#version 330 core
in vec3 v_color; out vec4 frag_color;
void main() { frag_color=vec4(v_color,1.0); }
"""

# ---------------------------------------------------------------------------
# Colors as plain Python lists (no module-level numpy — safe on Python 3.13)
# ---------------------------------------------------------------------------
C_1D_FILL = [0.30, 0.75, 1.00]
C_2D_FILL = [0.45, 0.78, 0.92]
C_3D_FILL = [1.00, 0.70, 0.20]
C_1D_WIRE = [0.50, 0.90, 1.00]
C_2D_WIRE = [0.40, 0.85, 1.00]
C_3D_WIRE = [1.00, 0.85, 0.35]
C_EDGE    = [0.12, 0.18, 0.28]
C_SPC     = [1.00, 0.27, 1.00]
_DIM_FILL = {'1d': C_1D_FILL, '2d': C_2D_FILL, '3d': C_3D_FILL}
_DIM_WIRE = {'1d': C_1D_WIRE, '2d': C_2D_WIRE, '3d': C_3D_WIRE}
_BEAM_STRESS_KEYS = {'sxc','sxd','sxe','sxf','smax','smin'}
_BEAM_STRESS_SURFACE_KEYS = _BEAM_STRESS_KEYS | {'stress3d'}


def _property_wire_color(pid: int, dim: str):
    if pid <= 0:
        return list(_DIM_WIRE[dim])
    hue = ((pid * 0.61803398875) % 1.0)
    sat = 0.55 if dim == '1d' else 0.45
    val = 1.0 if dim == '1d' else 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return [float(r), float(g), float(b)]

def _elem_dim(e):
    if e in ('CBAR','CBEAM','CROD'): return '1d'
    if e in ('CQUAD4','CTRIA3'):     return '2d'
    return '3d'  # CHEXA,CPENTA,CTETRA,CPYRAM

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _cpyram_face_tris(pos, nodes):
    """CPYRAM: 4-node base quad + 4 triangular side faces."""
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    if len(nodes) < 5: return []
    b0,b1,b2,b3,ap = [g(nodes[i]) for i in range(5)]
    tris = []
    # Base (quad split into 2 tris)
    tris += [(b0,b1,b2),(b0,b2,b3)]
    # 4 side triangles
    tris += [(b0,b1,ap),(b1,b2,ap),(b2,b3,ap),(b3,b0,ap)]
    return tris

def _cpyram_edges(pos, nodes):
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    if len(nodes) < 5: return []
    b = [g(nodes[i]) for i in range(4)]; ap = g(nodes[4])
    return [(b[0],b[1]),(b[1],b[2]),(b[2],b[3]),(b[3],b[0]),
            (b[0],ap),(b[1],ap),(b[2],ap),(b[3],ap)]

def _face_normal(v0,v1,v2):
    n = np.cross(v1-v0, v2-v0)
    l = np.linalg.norm(n)
    return (n/l).astype(np.float32) if l>1e-30 else np.array([0,1,0],dtype=np.float32)

def _build_face_tris(pos, nodes, etype):
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    t=[]
    if etype=='CTRIA3':
        t.append((g(nodes[0]),g(nodes[1]),g(nodes[2])))
    elif etype=='CQUAD4':
        t+=[(g(nodes[0]),g(nodes[1]),g(nodes[2])),(g(nodes[0]),g(nodes[2]),g(nodes[3]))]
    elif etype=='CTETRA':
        for f in [(0,1,2),(0,1,3),(1,2,3),(0,2,3)]:
            t.append(tuple(g(nodes[i]) for i in f))
    elif etype=='CPENTA':
        t+=[(g(nodes[0]),g(nodes[1]),g(nodes[2])),(g(nodes[3]),g(nodes[5]),g(nodes[4]))]
        for s in [(0,3,4,1),(1,4,5,2),(0,2,5,3)]:
            t+=[(g(nodes[s[0]]),g(nodes[s[1]]),g(nodes[s[2]])),(g(nodes[s[0]]),g(nodes[s[2]]),g(nodes[s[3]]))]
    elif etype=='CHEXA':
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]:
            t+=[(g(nodes[f[0]]),g(nodes[f[1]]),g(nodes[f[2]])),(g(nodes[f[0]]),g(nodes[f[2]]),g(nodes[f[3]]))]
    return t

def _build_face_tris_with_nodes(pos, nodes, etype):
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    t=[]
    if etype=='CTRIA3':
        t.append(((g(nodes[0]), nodes[0]), (g(nodes[1]), nodes[1]), (g(nodes[2]), nodes[2])))
    elif etype=='CQUAD4':
        t += [
            ((g(nodes[0]), nodes[0]), (g(nodes[1]), nodes[1]), (g(nodes[2]), nodes[2])),
            ((g(nodes[0]), nodes[0]), (g(nodes[2]), nodes[2]), (g(nodes[3]), nodes[3])),
        ]
    elif etype=='CTETRA':
        for f in [(0,1,2),(0,1,3),(1,2,3),(0,2,3)]:
            t.append(tuple((g(nodes[i]), nodes[i]) for i in f))
    elif etype=='CPENTA':
        t += [
            ((g(nodes[0]), nodes[0]), (g(nodes[1]), nodes[1]), (g(nodes[2]), nodes[2])),
            ((g(nodes[3]), nodes[3]), (g(nodes[5]), nodes[5]), (g(nodes[4]), nodes[4])),
        ]
        for s in [(0,3,4,1),(1,4,5,2),(0,2,5,3)]:
            t += [
                ((g(nodes[s[0]]), nodes[s[0]]), (g(nodes[s[1]]), nodes[s[1]]), (g(nodes[s[2]]), nodes[s[2]])),
                ((g(nodes[s[0]]), nodes[s[0]]), (g(nodes[s[2]]), nodes[s[2]]), (g(nodes[s[3]]), nodes[s[3]])),
            ]
    elif etype=='CHEXA':
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]:
            t += [
                ((g(nodes[f[0]]), nodes[f[0]]), (g(nodes[f[1]]), nodes[f[1]]), (g(nodes[f[2]]), nodes[f[2]])),
                ((g(nodes[f[0]]), nodes[f[0]]), (g(nodes[f[2]]), nodes[f[2]]), (g(nodes[f[3]]), nodes[f[3]])),
            ]
    elif etype=='CPYRAM':
        idx_tris = [(0,1,2),(0,2,3),(0,1,4),(1,2,4),(2,3,4),(3,0,4)]
        for tri in idx_tris:
            t.append(tuple((g(nodes[i]), nodes[i]) for i in tri))
    return t

def _build_edge_lines(pos, nodes, etype):
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    if etype in ('CBAR','CBEAM','CROD'): return [(g(nodes[0]),g(nodes[1]))]
    elif etype=='CTRIA3': return [(g(nodes[i]),g(nodes[(i+1)%3])) for i in range(3)]
    elif etype=='CQUAD4': return [(g(nodes[i]),g(nodes[(i+1)%4])) for i in range(4)]
    elif etype=='CTETRA': return [(g(nodes[a]),g(nodes[b])) for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]
    elif etype=='CPENTA': return [(g(nodes[a]),g(nodes[b])) for a,b in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]]
    elif etype=='CHEXA':  return [(g(nodes[a]),g(nodes[b])) for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]]
    elif etype=='CPYRAM': return _cpyram_edges(pos, nodes)
    return []

def _shell_thick_edges(pos, nodes, etype, t):
    """Return ALL boundary edges of an extruded shell:
    top perimeter + bottom perimeter + vertical corner edges.
    NO interior/diagonal edges = clean hidden line view."""
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    if t <= 1e-12:
        return _build_edge_lines(pos, nodes, etype)
    pts = [g(n) for n in nodes]
    if len(pts) < 3: return _build_edge_lines(pos, nodes, etype)
    # Compute face normal
    nrm = np.cross(pts[1]-pts[0], pts[2]-pts[0]).astype(np.float32)
    ln = np.linalg.norm(nrm)
    if ln < 1e-9: return _build_edge_lines(pos, nodes, etype)
    nrm /= ln
    n = len(pts)
    top = [p + nrm*(t*0.5) for p in pts]
    bot = [p - nrm*(t*0.5) for p in pts]
    edges = []
    # Top face perimeter
    for i in range(n): edges.append((top[i], top[(i+1)%n]))
    # Bottom face perimeter
    for i in range(n): edges.append((bot[i], bot[(i+1)%n]))
    # Vertical corner edges
    for i in range(n): edges.append((top[i], bot[i]))
    return edges


def _shell_thick_tris(pos, nodes, etype, t):
    def g(n): return pos.get(n, np.zeros(3,dtype=np.float32))
    ns=[g(nodes[i]) for i in range(3 if etype=='CTRIA3' else 4)]
    nrm=_face_normal(ns[0],ns[1],ns[2])
    top=[v+nrm*(t*.5) for v in ns]; bot=[v-nrm*(t*.5) for v in ns]
    n=len(ns); tris=[]
    if n==3: tris+=[(top[0],top[1],top[2]),(bot[0],bot[2],bot[1])]
    else:    tris+=[(top[0],top[1],top[2]),(top[0],top[2],top[3]),(bot[0],bot[2],bot[1]),(bot[0],bot[3],bot[2])]
    for i in range(n):
        j=(i+1)%n
        tris+=[(bot[i],top[i],top[j]),(bot[i],top[j],bot[j])]
    return tris

def _beam_local_frame(p0,p1,v_orient=None):
    ex=p1-p0; L=np.linalg.norm(ex)
    if L<1e-30: return np.array([1,0,0],dtype=np.float32),np.array([0,1,0],dtype=np.float32),np.array([0,0,1],dtype=np.float32)
    ex=(ex/L).astype(np.float32)
    v=((v_orient/np.linalg.norm(v_orient)).astype(np.float32)
       if v_orient is not None and np.linalg.norm(v_orient)>1e-9
       else (np.array([0,0,1],dtype=np.float32) if abs(ex[2])<0.9 else np.array([0,1,0],dtype=np.float32)))
    ez=v-np.dot(v,ex)*ex; nez=np.linalg.norm(ez)
    if nez<1e-12:
        p=np.array([1,0,0],dtype=np.float32) if abs(ex[0])<0.9 else np.array([0,1,0],dtype=np.float32)
        ez=np.cross(ex,p); nez=np.linalg.norm(ez)
    ez=(ez/nez).astype(np.float32)
    ey=np.cross(ez,ex).astype(np.float32); ey/=(np.linalg.norm(ey)+1e-30)
    return ex,ey,ez


def _disp_components_for_result(d, result_type: str):
    t = d.translation.astype(np.float32).copy()
    r = np.array([d.r1, d.r2, d.r3], dtype=np.float32)
    if result_type == 't1':
        t[1:] = 0.0
        r[:] = 0.0
    elif result_type == 't2':
        t[0] = 0.0; t[2] = 0.0
    elif result_type == 't3':
        t[:2] = 0.0
    return t, r


def _timoshenko_interp(s: float, u0: float, r0: float, u1: float, r1: float,
                       L: float, phi: Optional[float]) -> float:
    phi = max(0.0, float(phi or 0.0))
    if phi <= 1e-12:
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2
        return h00 * u0 + h10 * L * r0 + h01 * u1 + h11 * L * r1

    den = 1.0 + phi
    n1 = (1.0 - 3.0 * s * s + 2.0 * s**3 + phi * (1.0 - s)) / den
    n2 = (L * (s - 2.0 * s * s + s**3) + 0.5 * phi * L * (s - s * s)) / den
    n3 = (3.0 * s * s - 2.0 * s**3 + phi * s) / den
    n4 = (L * (-s * s + s**3) + 0.5 * phi * L * (-s + s * s)) / den
    return n1 * u0 + n2 * r0 + n3 * u1 + n4 * r1


def _beam_timoshenko_phi(model: MystranModel, elem, length: float) -> Optional[tuple]:
    if length <= 1e-12:
        return None
    prop = model.properties.get(getattr(elem, 'pid', 0))
    if prop is None or prop.mid <= 0:
        return None
    mat = model.materials.get(prop.mid)
    if mat is None or mat.E <= 1e-12:
        return None
    G = float(mat.G)
    if G <= 1e-12 and abs(float(mat.nu)) < 0.4999:
        G = float(mat.E) / (2.0 * (1.0 + float(mat.nu)))
    if G <= 1e-12:
        return None
    sec = getattr(prop, 'section', None)
    if sec is None:
        return None
    area, iy, iz = section_area_inertias(sec)
    if area <= 1e-12 or iy <= 1e-16 or iz <= 1e-16:
        return None
    kappa = 5.0 / 6.0
    phi_y = 12.0 * mat.E * iz / (kappa * G * area * length * length)
    phi_z = 12.0 * mat.E * iy / (kappa * G * area * length * length)
    return float(phi_y), float(phi_z)


def _hermite_beam_curve_points(p0, p1, d0, d1, deform_scale: float, nsamp: int = 17, v_orient=None,
                               result_type: str = 'displacement', phi_yz: Optional[tuple] = None,
                               model: Optional[MystranModel] = None, elem=None, load_sid: int = 0,
                               beam_end_data=None):
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)
    ex, ey, ez = _beam_local_frame(p0, p1, v_orient)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-12:
        pts = np.array([p0, p1], dtype=np.float32)
        return pts

    t0, r0 = _disp_components_for_result(d0, result_type)
    t1, r1 = _disp_components_for_result(d1, result_type)
    t0 = t0 * deform_scale
    t1 = t1 * deform_scale

    u0x = float(np.dot(t0, ex)); u1x = float(np.dot(t1, ex))
    u0y = float(np.dot(t0, ey)); u1y = float(np.dot(t1, ey))
    u0z = float(np.dot(t0, ez)); u1z = float(np.dot(t1, ez))

    slope0 = np.cross(r0, ex).astype(np.float32)
    slope1 = np.cross(r1, ex).astype(np.float32)
    s0y = float(np.dot(slope0, ey)); s1y = float(np.dot(slope1, ey))
    s0z = float(np.dot(slope0, ez)); s1z = float(np.dot(slope1, ez))
    phi_y = float(phi_yz[0]) if phi_yz is not None else 0.0
    phi_z = float(phi_yz[1]) if phi_yz is not None else 0.0

    svals = np.linspace(0.0, 1.0, max(3, int(nsamp)), dtype=np.float32)
    xs = (svals.astype(np.float64) * L).astype(np.float64)
    if model is not None and elem is not None:
        prop = model.properties.get(getattr(elem, 'pid', 0))
        mat = model.materials.get(prop.mid) if prop is not None else None
        sec = getattr(prop, 'section', None) if prop is not None else None
        G = float(mat.G) if mat is not None else 0.0
        if G <= 1e-12 and mat is not None and abs(float(mat.nu)) < 0.4999:
            G = float(mat.E) / (2.0 * (1.0 + float(mat.nu)))
        if mat is not None and sec is not None and G > 1e-12:
            A, Iy, Iz = section_area_inertias(sec)
            qy, qz = _transverse_pload_components(model, elem, load_sid, xs, ex, ey, ez)
            py, pz = _concentrated_pload_components(model, elem, load_sid, L, ex, ey, ez)
            ef = _beam_end_force_components(beam_end_data, getattr(elem, 'id', 0))
            kappa = 5.0 / 6.0
            vy_exact = _timoshenko_transverse_from_loads(xs, qy, u0y, s0y, u1y, s1y, float(mat.E), Iz, G, A, kappa) if (A > 1e-12 and Iz > 1e-16 and np.max(np.abs(qy)) > 1e-12) else None
            vz_exact = _timoshenko_transverse_from_loads(xs, qz, u0z, s0z, u1z, s1z, float(mat.E), Iy, G, A, kappa) if (A > 1e-12 and Iy > 1e-16 and np.max(np.abs(qz)) > 1e-12) else None
            if vy_exact is None and ef is not None and py:
                vy_exact = _double_integrate_point_loads(xs, u0y, u1y, ef.get('bm2_A', 0.0), ef.get('ts2_A', 0.0), float(mat.E), Iz, py, G, A, kappa)
            if vz_exact is None and ef is not None and pz:
                vz_exact = _double_integrate_point_loads(xs, u0z, u1z, ef.get('bm1_A', 0.0), ef.get('ts1_A', 0.0), float(mat.E), Iy, pz, G, A, kappa)
        else:
            vy_exact = None
            vz_exact = None
    else:
        vy_exact = None
        vz_exact = None
    out = []
    for i, s in enumerate(svals):
        axial = (1.0 - s) * u0x + s * u1x
        vy = float(vy_exact[i]) if vy_exact is not None else _timoshenko_interp(float(s), u0y, s0y, u1y, s1y, L, phi_y)
        vz = float(vz_exact[i]) if vz_exact is not None else _timoshenko_interp(float(s), u0z, s0z, u1z, s1z, L, phi_z)
        base = p0 + ex * (L * float(s))
        disp = ex * axial + ey * vy + ez * vz
        out.append(base + disp)
    return np.array(out, dtype=np.float32)


def _beam_curve_profile_rings(curve_pts, profile, v_orient=None):
    if curve_pts is None or len(curve_pts) < 2 or not profile or len(profile) < 2:
        return []
    rings = []
    nseg = len(curve_pts)
    for i in range(nseg):
        if i == 0:
            pa = curve_pts[i]
            pb = curve_pts[i + 1]
        elif i == nseg - 1:
            pa = curve_pts[i - 1]
            pb = curve_pts[i]
        else:
            pa = curve_pts[i - 1]
            pb = curve_pts[i + 1]
        ex, ey, ez = _beam_local_frame(pa, pb, v_orient)
        c = curve_pts[i]
        ring = [c + float(y) * ey + float(z) * ez for y, z in profile]
        rings.append(ring)
    return rings

def _section_loops(profile_or_loops):
    if not profile_or_loops:
        return []
    first = profile_or_loops[0]
    if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple, np.ndarray)):
        return [list(loop) for loop in profile_or_loops]
    return [list(profile_or_loops)]


def _beam_curve_section_tris(curve_pts, profile, v_orient=None, cap_ends=True):
    tris = []
    loops = _section_loops(profile)
    for loop in loops:
        rings = _beam_curve_profile_rings(curve_pts, loop, v_orient)
        if len(rings) < 2:
            continue
        n = len(loop)
        for k in range(len(rings) - 1):
            r0 = rings[k]
            r1 = rings[k + 1]
            for i in range(n):
                j = (i + 1) % n
                tris += [(r0[i], r0[j], r1[j]), (r0[i], r1[j], r1[i])]
        if cap_ends and len(loops) == 1 and len(loop) >= 3:
            for i0, i1, i2 in _triangulate_profile(loop):
                tris += [(rings[0][i0], rings[0][i1], rings[0][i2]),
                         (rings[-1][i0], rings[-1][i2], rings[-1][i1])]
    return tris


def _beam_curve_section_edges(curve_pts, profile, v_orient=None):
    loops = _section_loops(profile)
    if not loops:
        if curve_pts is not None and len(curve_pts) >= 2:
            return [(curve_pts[i], curve_pts[i+1]) for i in range(len(curve_pts)-1)]
        return []
    edges = []
    for loop in loops:
        rings = _beam_curve_profile_rings(curve_pts, loop, v_orient)
        if len(rings) < 2:
            continue
        n = len(loop)
        for k in range(len(rings) - 1):
            r0 = rings[k]
            r1 = rings[k + 1]
            for i in range(n):
                edges.append((r0[i], r1[i]))
            if k == 0:
                for i in range(n):
                    edges.append((r0[i], r0[(i + 1) % n]))
        for i in range(n):
            edges.append((rings[-1][i], rings[-1][(i + 1) % n]))
    return edges


def _poly_area2d(profile):
    area = 0.0
    n = len(profile)
    for i in range(n):
        x1, y1 = profile[i]
        x2, y2 = profile[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def _pt_in_tri_2d(p, a, b, c):
    def sgn(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sgn(p, a, b)
    d2 = sgn(p, b, c)
    d3 = sgn(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _triangulate_profile(profile):
    if not profile or len(profile) < 3:
        return []
    pts = [(float(y), float(z)) for y, z in profile]
    ccw = _poly_area2d(pts) > 0.0
    idx = list(range(len(pts)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 1000:
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            i0 = idx[(k - 1) % m]
            i1 = idx[k]
            i2 = idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if ccw:
                if cross <= 1e-12:
                    continue
            else:
                if cross >= -1e-12:
                    continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _pt_in_tri_2d(pts[j], a, b, c):
                    ok = False
                    break
            if not ok:
                continue
            tris.append((i0, i1, i2) if ccw else (i0, i2, i1))
            del idx[k]
            ear_found = True
            break
        if not ear_found:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]) if ccw else (idx[0], idx[2], idx[1]))
    return tris

def _beam_section_tris(p0,p1,profile,v_orient=None,cap_ends=True):
    tris=[]
    loops = _section_loops(profile)
    if not loops: return []
    ex,ey,ez=_beam_local_frame(p0,p1,v_orient)
    def lw(pt,y,z): return pt+float(y)*ey+float(z)*ez
    for loop in loops:
        if len(loop) < 3:
            continue
        r0=[lw(p0,y,z) for y,z in loop]; r1=[lw(p1,y,z) for y,z in loop]; n=len(loop)
        for i in range(n):
            j=(i+1)%n; tris+=[(r0[i],r0[j],r1[j]),(r0[i],r1[j],r1[i])]
        if cap_ends and len(loops) == 1:
            for i0, i1, i2 in _triangulate_profile(loop):
                tris += [(r0[i0], r0[i1], r0[i2]), (r1[i0], r1[i2], r1[i1])]
    return tris

def _beam_section_edges(p0,p1,profile,v_orient=None):
    loops = _section_loops(profile)
    if not loops: return [(p0,p1)]
    ex,ey,ez=_beam_local_frame(p0,p1,v_orient)
    def lw(pt,y,z): return pt+float(y)*ey+float(z)*ez
    edges = []
    for loop in loops:
        if len(loop) < 2:
            continue
        r0=[lw(p0,y,z) for y,z in loop]; r1=[lw(p1,y,z) for y,z in loop]; n=len(loop)
        edges += [(r0[i],r1[i]) for i in range(n)]
        edges += [(r0[i],r0[(i+1)%n]) for i in range(n)]
        edges += [(r1[i],r1[(i+1)%n]) for i in range(n)]
    return edges

def _get_beam_profile(elem,model,fallback):
    prop=model.properties.get(elem.pid); v_orient=getattr(elem,'v_orient',None); profile=None; cap_ends=True
    explicit_section = bool(prop and prop.type in ('PBARL', 'PBEAML') and prop.section and (prop.section.profile or getattr(prop.section, 'loops', None)))
    if explicit_section:
        s=prop.section; n0=model.nodes.get(elem.nodes[0]); n1=model.nodes.get(elem.nodes[1])
        if n0 and n1:
            L=float(np.linalg.norm(n1.xyz-n0.xyz)); bmax=max(s.b,s.h,1e-30)
            sc=min(1.0,(L*.45)/bmax) if bmax>L*.7 else 1.0
            src_loops = getattr(s, 'loops', None) or [s.profile]
            profile=[[(y*sc,z*sc) for y,z in loop] for loop in src_loops if loop]
            cap_ends = bool(getattr(s, 'cap_ends', True))
    if profile is None:
        mn, mx = model.bbox()
        total_len = float(np.max(mx - mn)) if model.nodes else 1.0
        max_area = 0.0
        for e in model.elements.values():
            if e.type not in ('CBAR', 'CBEAM', 'CROD') or len(e.nodes) < 2:
                continue
            p = model.properties.get(e.pid)
            sec = getattr(p, 'section', None) if p is not None else None
            if sec is not None and getattr(sec, 'area', 0.0) > 0.0:
                max_area = max(max_area, float(sec.area))
            elif p is not None:
                try:
                    max_area = max(max_area, float(p.params.get('f3', 0.0) or 0.0))
                except Exception:
                    pass
        this_area = 0.0
        if prop is not None and getattr(prop, 'section', None) is not None:
            this_area = float(getattr(prop.section, 'area', 0.0) or 0.0)
        elif prop is not None:
            try:
                this_area = float(prop.params.get('f3', 0.0) or 0.0)
            except Exception:
                this_area = 0.0
        if max_area <= 1e-30:
            max_area = max(this_area, 1.0)
        area_ratio = np.sqrt(max(this_area, 1e-30) / max(max_area, 1e-30))
        h = (total_len / 12.0) * float(np.clip(area_ratio, 0.2, 1.0))
        w = h * 0.5
        profile=[[(-w*0.5,-h*0.5),(w*0.5,-h*0.5),(w*0.5,h*0.5),(-w*0.5,h*0.5)]]
    return profile,v_orient,cap_ends

def _beam_section_cdef_points(prop):
    if prop is None:
        return None
    ptype = str(getattr(prop, 'type', '')).upper()
    params = getattr(prop, 'params', {}) or {}
    if ptype == 'PBEAM':
        pts = pbeam_cdef_points(params)
        if any(abs(v) > 1e-12 for pair in pts.values() for v in pair):
            return {k: (float(v[0]), float(v[1])) for k, v in pts.items()}
    sec = getattr(prop, 'section', None)
    loops = getattr(sec, 'loops', None) or ([getattr(sec, 'profile', [])] if sec is not None else [])
    flat = [pt for loop in loops for pt in loop]
    if not flat:
        return None
    ys = [float(pt[0]) for pt in flat]
    zs = [float(pt[1]) for pt in flat]
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    return {
        'C': (ymax, zmax),
        'D': (ymax, zmin),
        'E': (ymin, zmin),
        'F': (ymin, zmax),
    }

def _beam_corner_field_value(y, z, cdef_pts, vals):
    if not cdef_pts or not vals:
        return 0.0
    corners = {
        'ur': max(cdef_pts, key=lambda k: cdef_pts[k][0] + cdef_pts[k][1]),
        'ul': max(cdef_pts, key=lambda k: cdef_pts[k][1] - cdef_pts[k][0]),
        'lr': max(cdef_pts, key=lambda k: cdef_pts[k][0] - cdef_pts[k][1]),
        'll': min(cdef_pts, key=lambda k: cdef_pts[k][0] + cdef_pts[k][1]),
    }
    ys = [float(cdef_pts[k][0]) for k in corners.values()]
    zs = [float(cdef_pts[k][1]) for k in corners.values()]
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    sy = float(np.clip((float(y) - ymin) / max(ymax - ymin, 1e-30), 0.0, 1.0))
    sz = float(np.clip((float(z) - zmin) / max(zmax - zmin, 1e-30), 0.0, 1.0))
    vll = float(vals[corners['ll']])
    vlr = float(vals[corners['lr']])
    vur = float(vals[corners['ur']])
    vul = float(vals[corners['ul']])
    return ((1.0 - sy) * (1.0 - sz) * vll +
            sy * (1.0 - sz) * vlr +
            sy * sz * vur +
            (1.0 - sy) * sz * vul)

# ---------------------------------------------------------------------------
# Buffer builders
# ---------------------------------------------------------------------------
def _make_solid_buf(verts, colors, norms):
    if not verts: return None
    n=len(verts)
    buf=np.empty((n,9),dtype=np.float32)
    buf[:,0:3]=np.array(verts, dtype=np.float32)
    buf[:,3:6]=np.array(colors,dtype=np.float32)
    buf[:,6:9]=np.array(norms, dtype=np.float32)
    return buf.flatten()

def _make_line_buf(verts, colors):
    if not verts: return None
    n=len(verts)
    buf=np.empty((n,6),dtype=np.float32)
    buf[:,0:3]=np.array(verts, dtype=np.float32)
    buf[:,3:6]=np.array(colors,dtype=np.float32)
    return buf.flatten()

# ---------------------------------------------------------------------------
# MeshRenderer
# ---------------------------------------------------------------------------
class MeshRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx=ctx
        self._prog_solid=ctx.program(vertex_shader=VERT_SOLID,fragment_shader=FRAG_SOLID)
        self._prog_line =ctx.program(vertex_shader=VERT_LINE, fragment_shader=FRAG_LINE)
        self._prog_fill =ctx.program(vertex_shader=VERT_FILL, fragment_shader=FRAG_FILL)
        self._reset()

    def _reset(self):
        self._solid={'1d':(None,None,0),'2d':(None,None,0),'3d':(None,None,0)}
        self._wire ={'1d':(None,None,0),'2d':(None,None,0),'3d':(None,None,0)}
        self._fill ={'2d':(None,None,0),'3d':(None,None,0)}
        self._undeformed = (None,None,0)  # ghost wireframe for deformed comparison
        self._spc  =(None,None,0)

    def _up_fill(self,verts,colors,norms):
        """Upload fill geometry for wireframe alpha pass."""
        d=_make_solid_buf(verts,colors,norms)
        if d is None: return None,None,0
        vbo=self.ctx.buffer(d.tobytes())
        vao=self.ctx.vertex_array(self._prog_fill,[(vbo,'3f 3f 3f','in_position','in_color','in_normal')])
        return vao,vbo,len(verts)

    def _up_solid(self,verts,colors,norms):
        d=_make_solid_buf(verts,colors,norms)
        if d is None: return None,None,0
        vbo=self.ctx.buffer(d.tobytes())
        vao=self.ctx.vertex_array(self._prog_solid,[(vbo,'3f 3f 3f','in_position','in_color','in_normal')])
        return vao,vbo,len(verts)

    def upload_undeformed(self, model):
        """Upload thin white wireframe of undeformed shape for comparison.
        Uses short dash segments to create dashed-line appearance."""
        pos = {nid: n.xyz.astype(np.float32) for nid,n in model.nodes.items()}
        C_GHOST = [0.85, 0.85, 0.85]
        wv=[]; wc=[]
        dash_frac = 0.45  # fraction of segment that is drawn
        for eid, elem in model.elements.items():
            for va, vb in _build_edge_lines(pos, elem.nodes, elem.type):
                # Split each edge into dash segments
                diff = vb - va
                L = np.linalg.norm(diff)
                if L < 1e-9: continue
                n_dashes = max(2, int(L / (model.scale()*0.04)))
                for i in range(n_dashes):
                    t0 = i / n_dashes
                    t1 = t0 + dash_frac / n_dashes
                    p0 = va + diff * t0
                    p1 = va + diff * t1
                    wv += [p0, p1]; wc += [C_GHOST, C_GHOST]
        self._undeformed = self._up_lines(wv, wc)

    def _up_lines(self,verts,colors):
        d=_make_line_buf(verts,colors)
        if d is None: return None,None,0
        vbo=self.ctx.buffer(d.tobytes())
        vao=self.ctx.vertex_array(self._prog_line,[(vbo,'3f 3f','in_position','in_color')])
        return vao,vbo,len(verts)

    def upload(self,model,results=None,subcase=1,display_mode='wireframe',
               result_type='displacement',cmap_name='rainbow',deform_scale=0.0,
               nodal_result_source='solver_first', active_spc_sid=0, beam_end_data=None):
        self._reset()
        pos={}
        def _result_deform_vec(dr):
            if dr is None:
                return None
            if result_type == 't1':
                return np.array([dr.t1, 0.0, 0.0], dtype=np.float32)
            if result_type == 't2':
                return np.array([0.0, dr.t2, 0.0], dtype=np.float32)
            if result_type == 't3':
                return np.array([0.0, 0.0, dr.t3], dtype=np.float32)
            return dr.translation.astype(np.float32)

        for nid,node in model.nodes.items():
            p=node.xyz.astype(np.float32).copy()
            if deform_scale>0 and results and subcase in results.displacements:
                dr=results.displacements[subcase].get(nid)
                dv = _result_deform_vec(dr)
                if dv is not None: p=p+dv*deform_scale
            pos[nid]=p

        fallback=model.scale()*.025

        # Contour colors
        c_node={}; c_elem={}
        disp_lo = 0.0
        disp_hi = 1.0
        disp_cmap = None
        beam_lo = 0.0
        beam_hi = 1.0
        beam_cmap = None
        beam_station_vals = {}
        if display_mode=='contour' and results:
            cmap=COLORMAPS[cmap_name]
            if result_type=='displacement' and subcase in results.displacements:
                mags={n:float(np.linalg.norm(d.translation)) for n,d in results.displacements[subcase].items()}
                lo,hi=min(mags.values(),default=0),max(mags.values(),default=1)
                disp_lo, disp_hi = lo, hi
                disp_cmap = cmap
                for n,m in mags.items(): c_node[n]=cmap[int(np.clip((m-lo)/(hi-lo+1e-30),0,1)*255)].tolist()
            elif subcase in results.stresses and result_type in (
                    'von_mises','oxx','oyy','txy','omax','omin',
                    'von_mises_top','von_mises_bottom',
                    'nodal_vm','noxx','noyy','ntxy','nomax','nomin'):
                st = results.stresses[subcase]
                # Handle nodal avg dict vs element stress dict
                elem_stress = {k:v for k,v in st.items()
                               if k not in ('_nodal_avg', '_nodal_avg_components', '_nodal_acc',
                                            '_solver_nodal_avg', '_solver_nodal_avg_components',
                                            '_derived_nodal_avg', '_derived_nodal_avg_components',
                                            '_shell_corner_contribs')
                               and isinstance(v, object)
                               and hasattr(v,'values')
                               and hasattr(v, 'von_mises')
                               and getattr(v, 'elem_type', '').upper() not in ('CBAR', 'CBEAM')}
                if nodal_result_source == 'derived':
                    nodal_avg   = st.get('_derived_nodal_avg', {})
                    nodal_comp  = st.get('_derived_nodal_avg_components', {})
                else:
                    nodal_avg   = st.get('_solver_nodal_avg', {}) or st.get('_derived_nodal_avg', {})
                    nodal_comp  = st.get('_solver_nodal_avg_components', {}) or st.get('_derived_nodal_avg_components', {})
                if result_type == 'displacement':
                    pass  # handled above
                elif result_type in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin'):
                    # Nodal averaged von mises -> node colors
                    base = {
                        'nodal_vm': 'von_mises',
                        'noxx': 'oxx',
                        'noyy': 'oyy',
                        'ntxy': 'txy',
                        'nomax': 'omax',
                        'nomin': 'omin',
                    }[result_type]
                    if nodal_comp:
                        nav = [vals.get(base, 0.0) for vals in nodal_comp.values()]
                        lo,hi = min(nav), max(nav)
                        for nid, vals in nodal_comp.items():
                            v = vals.get(base, 0.0)
                            c_node[nid] = cmap[int(np.clip((v-lo)/(hi-lo+1e-30),0,1)*255)].tolist()
                else:
                    def _get_val(es):
                        if result_type == 'von_mises': return es.von_mises
                        return es.values.get(result_type, es.von_mises)
                    vms = [_get_val(s) for s in elem_stress.values()]
                    if not vms: vms = [0, 1]
                    lo,hi = min(vms), max(vms)
                    for e,s in elem_stress.items():
                        v = _get_val(s)
                        c_elem[e] = cmap[int(np.clip((v-lo)/(hi-lo+1e-30),0,1)*255)].tolist()
            # Force/moment contours (element)
            elif result_type in ('fx','fy','fxy','mx','my','mxy','qx','qy'):
                if hasattr(results,'forces') and subcase in results.forces:
                    fc = results.forces[subcase]
                    real = {k:v for k,v in fc.items()
                            if k not in ('_nodal_avg', '_solver_nodal_avg', '_derived_nodal_avg')
                            and hasattr(v,'values')}
                    fvals = [v.values.get(result_type,0) for v in real.values()]
                    if fvals:
                        lo,hi = min(fvals),max(fvals)
                        for eid,ef in real.items():
                            v = ef.values.get(result_type,0)
                            c_elem[eid] = cmap[int(np.clip((v-lo)/(hi-lo+1e-30),0,1)*255)].tolist()
            # Force/moment contours (nodal averaged)
            elif result_type in ('nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy'):
                _nmap = {'nfx':'fx','nfy':'fy','nfxy':'fxy','nmx':'mx','nmy':'my',
                         'nmxy':'mxy','nqx':'qx','nqy':'qy'}
                base = _nmap[result_type]
                if hasattr(results,'forces') and subcase in results.forces:
                    if nodal_result_source == 'derived':
                        nav = results.forces[subcase].get('_derived_nodal_avg', {})
                    else:
                        nav = results.forces[subcase].get('_solver_nodal_avg', {}) or results.forces[subcase].get('_derived_nodal_avg', {})
                    if nav:
                        nvals = [v.get(base,0) for v in nav.values()]
                        lo,hi = min(nvals),max(nvals)
                        for nid, vd in nav.items():
                            v = vd.get(base,0)
                            c_node[nid] = cmap[int(np.clip((v-lo)/(hi-lo+1e-30),0,1)*255)].tolist()

            # Displacement component contours
            elif result_type in ('t1','t2','t3') and subcase in results.displacements:
                comp = {'t1':0,'t2':1,'t3':2}[result_type]
                vals = {n: float(d.translation[comp])
                        for n,d in results.displacements[subcase].items()}
                lo,hi = min(vals.values(),default=0), max(vals.values(),default=1)
                disp_lo, disp_hi = lo, hi
                disp_cmap = cmap
                for n,v in vals.items():
                    c_node[n] = cmap[int(np.clip((v-lo)/(hi-lo+1e-30),0,1)*255)].tolist()
            elif result_type in _BEAM_STRESS_SURFACE_KEYS and beam_end_data:
                vals = []
                for eid, bd in beam_end_data.items():
                    if not getattr(bd, 'stations', None):
                        continue
                    sds = np.array([float(st.sd) for st in bd.stations], dtype=np.float32)
                    if len(sds) == 0:
                        continue
                    if result_type == 'stress3d':
                        cvals = np.array([float(getattr(st, 'sxc', 0.0)) for st in bd.stations], dtype=np.float32)
                        dvals = np.array([float(getattr(st, 'sxd', 0.0)) for st in bd.stations], dtype=np.float32)
                        evals = np.array([float(getattr(st, 'sxe', 0.0)) for st in bd.stations], dtype=np.float32)
                        fvals = np.array([float(getattr(st, 'sxf', 0.0)) for st in bd.stations], dtype=np.float32)
                        bvals = 0.25 * (cvals + dvals + evals + fvals)
                        vals.extend(cvals.tolist())
                        vals.extend(dvals.tolist())
                        vals.extend(evals.tolist())
                        vals.extend(fvals.tolist())
                    else:
                        bvals = np.array([float(getattr(st, result_type, 0.0)) for st in bd.stations], dtype=np.float32)
                        vals.extend(bvals.tolist())
                    beam_station_vals[eid] = (sds, bvals)
                if vals:
                    beam_lo, beam_hi = min(vals), max(vals)
                    beam_cmap = cmap

        sv={d:[] for d in('1d','2d','3d')}; sc={d:[] for d in('1d','2d','3d')}; sn={d:[] for d in('1d','2d','3d')}
        wv={d:[] for d in('1d','2d','3d')}; wc={d:[] for d in('1d','2d','3d')}

        for eid,elem in model.elements.items():
            dim=_elem_dim(elem.type)
            fc=list(_DIM_FILL[dim]); wc2=_property_wire_color(getattr(elem, 'pid', 0), dim)
            use_nodal_contour = (display_mode == 'contour' and eid not in c_elem and
                                 elem.nodes and any(n in c_node for n in elem.nodes))
            if display_mode=='contour':
                if eid in c_elem: fc=list(c_elem[eid])
                elif elem.nodes and any(n in c_node for n in elem.nodes):
                    nc=[c_node[n] for n in elem.nodes if n in c_node]
                    if nc: fc=list(np.mean(nc,axis=0))

            if elem.type in ('CBAR','CBEAM','CROD'):
                if len(elem.nodes)<2: continue
                n0 = elem.nodes[0]; n1 = elem.nodes[1]
                p0=pos.get(n0); p1=pos.get(n1)
                if p0 is None or p1 is None: continue
                beam_surface_contour = (display_mode == 'contour' and beam_cmap is not None and
                                        result_type in _BEAM_STRESS_SURFACE_KEYS and eid in beam_station_vals)
                curve_pts = None
                if (elem.type in ('CBAR', 'CBEAM') and deform_scale > 0 and results and
                        subcase in results.displacements):
                    d0 = results.displacements[subcase].get(n0)
                    d1 = results.displacements[subcase].get(n1)
                    if d0 is not None and d1 is not None:
                        p0u = model.nodes[n0].xyz.astype(np.float32)
                        p1u = model.nodes[n1].xyz.astype(np.float32)
                        load_sid = int(getattr(model, 'subcase_loads', {}).get(subcase, 0) or 0)
                        curve_phi = _beam_timoshenko_phi(model, elem, float(np.linalg.norm(p1u - p0u)))
                        curve_pts = _hermite_beam_curve_points(
                            p0u, p1u, d0, d1, deform_scale, nsamp=17,
                            v_orient=getattr(elem, 'v_orient', None),
                            result_type=result_type, phi_yz=curve_phi,
                            model=model, elem=elem, load_sid=load_sid,
                            beam_end_data=beam_end_data)
                if display_mode in ('wireframe','contour') and not beam_surface_contour:
                    # 1D: always line style; contour = colored line
                    col = fc if display_mode=='contour' else wc2
                    if curve_pts is not None and len(curve_pts) >= 2:
                        if display_mode == 'contour' and beam_cmap is not None and result_type in _BEAM_STRESS_SURFACE_KEYS and eid in beam_station_vals:
                            sds, bvals = beam_station_vals[eid]
                            for i in range(len(curve_pts) - 1):
                                s0 = float(i) / float(len(curve_pts) - 1)
                                s1 = float(i + 1) / float(len(curve_pts) - 1)
                                v0 = float(np.interp(s0, sds, bvals))
                                v1 = float(np.interp(s1, sds, bvals))
                                c0 = beam_cmap[int(np.clip((v0 - beam_lo) / (beam_hi - beam_lo + 1e-30), 0, 1) * 255)].tolist()
                                c1 = beam_cmap[int(np.clip((v1 - beam_lo) / (beam_hi - beam_lo + 1e-30), 0, 1) * 255)].tolist()
                                wv['1d'] += [curve_pts[i], curve_pts[i+1]]
                                wc['1d'] += [c0, c1]
                        elif display_mode == 'contour' and disp_cmap is not None and result_type in ('displacement', 't1', 't2', 't3'):
                            p0u = model.nodes[n0].xyz.astype(np.float32)
                            p1u = model.nodes[n1].xyz.astype(np.float32)
                            for i in range(len(curve_pts) - 1):
                                s0 = float(i) / float(len(curve_pts) - 1)
                                s1 = float(i + 1) / float(len(curve_pts) - 1)
                                b0 = p0u + (p1u - p0u) * s0
                                b1 = p0u + (p1u - p0u) * s1
                                dvec0 = curve_pts[i] - b0
                                dvec1 = curve_pts[i + 1] - b1
                                if result_type == 'displacement':
                                    v0 = float(np.linalg.norm(dvec0) / max(deform_scale, 1e-30))
                                    v1 = float(np.linalg.norm(dvec1) / max(deform_scale, 1e-30))
                                else:
                                    comp_idx = {'t1': 0, 't2': 1, 't3': 2}[result_type]
                                    v0 = float(dvec0[comp_idx] / max(deform_scale, 1e-30))
                                    v1 = float(dvec1[comp_idx] / max(deform_scale, 1e-30))
                                c0 = disp_cmap[int(np.clip((v0 - disp_lo) / (disp_hi - disp_lo + 1e-30), 0, 1) * 255)].tolist()
                                c1 = disp_cmap[int(np.clip((v1 - disp_lo) / (disp_hi - disp_lo + 1e-30), 0, 1) * 255)].tolist()
                                wv['1d'] += [curve_pts[i], curve_pts[i+1]]
                                wc['1d'] += [c0, c1]
                        else:
                            for i in range(len(curve_pts) - 1):
                                wv['1d'] += [curve_pts[i], curve_pts[i+1]]
                                wc['1d'] += [col, col]
                    else:
                        if display_mode == 'contour' and beam_cmap is not None and result_type in _BEAM_STRESS_SURFACE_KEYS and eid in beam_station_vals:
                            sds, bvals = beam_station_vals[eid]
                            nsamp = 11
                            pts_line = [p0 + (p1 - p0) * (float(i) / float(nsamp - 1)) for i in range(nsamp)]
                            for i in range(nsamp - 1):
                                s0 = float(i) / float(nsamp - 1)
                                s1 = float(i + 1) / float(nsamp - 1)
                                v0 = float(np.interp(s0, sds, bvals))
                                v1 = float(np.interp(s1, sds, bvals))
                                c0 = beam_cmap[int(np.clip((v0 - beam_lo) / (beam_hi - beam_lo + 1e-30), 0, 1) * 255)].tolist()
                                c1 = beam_cmap[int(np.clip((v1 - beam_lo) / (beam_hi - beam_lo + 1e-30), 0, 1) * 255)].tolist()
                                wv['1d'] += [pts_line[i], pts_line[i+1]]
                                wc['1d'] += [c0, c1]
                        else:
                            wv['1d']+=[p0,p1]; wc['1d']+=[col,col]
                else:  # solid/hidden or beam contour: extruded section
                    prof,vori,cap_ends=_get_beam_profile(elem,model,fallback)
                    tris = (_beam_curve_section_tris(curve_pts, prof, vori, cap_ends)
                            if curve_pts is not None and len(curve_pts) >= 2
                            else _beam_section_tris(p0,p1,prof,vori,cap_ends))
                    beam_axis = p1 - p0
                    beam_len = float(np.linalg.norm(beam_axis))
                    beam_dir = (beam_axis / beam_len).astype(np.float32) if beam_len > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    _, beam_ey, beam_ez = _beam_local_frame(p0, p1, vori)
                    prop = model.properties.get(getattr(elem, 'pid', 0))
                    cdef_pts = _beam_section_cdef_points(prop)
                    sds = bvals = None
                    corner_series = None
                    if beam_surface_contour:
                        sds, bvals = beam_station_vals[eid]
                        bd = beam_end_data.get(eid) if beam_end_data else None
                        if bd is not None and getattr(bd, 'stations', None):
                            corner_series = {
                                'C': np.array([float(getattr(st, 'sxc', 0.0)) for st in bd.stations], dtype=np.float32),
                                'D': np.array([float(getattr(st, 'sxd', 0.0)) for st in bd.stations], dtype=np.float32),
                                'E': np.array([float(getattr(st, 'sxe', 0.0)) for st in bd.stations], dtype=np.float32),
                                'F': np.array([float(getattr(st, 'sxf', 0.0)) for st in bd.stations], dtype=np.float32),
                            }
                    for v0,v1,v2 in tris:
                        nm=_face_normal(v0,v1,v2)
                        for v in(v0,v1,v2):
                            sv['1d'].append(v)
                            if beam_surface_contour and sds is not None and bvals is not None:
                                sval = float(np.clip(np.dot(v - p0, beam_dir) / max(beam_len, 1e-30), 0.0, 1.0))
                                if result_type == 'stress3d' and cdef_pts is not None and corner_series is not None:
                                    center = p0 + beam_dir * (sval * beam_len)
                                    yloc = float(np.dot(v - center, beam_ey))
                                    zloc = float(np.dot(v - center, beam_ez))
                                    corner_vals = {k: float(np.interp(sval, sds, arr)) for k, arr in corner_series.items()}
                                    bval = _beam_corner_field_value(yloc, zloc, cdef_pts, corner_vals)
                                else:
                                    bval = float(np.interp(sval, sds, bvals))
                                cval = beam_cmap[int(np.clip((bval - beam_lo) / (beam_hi - beam_lo + 1e-30), 0, 1) * 255)].tolist()
                                sc['1d'].append(cval)
                            else:
                                sc['1d'].append(fc)
                            sn['1d'].append(nm)
                    edges = (_beam_curve_section_edges(curve_pts, prof, vori)
                             if curve_pts is not None and len(curve_pts) >= 2
                             else _beam_section_edges(p0,p1,prof,vori))
                    for va,vb in edges:
                        wv['1d']+=[va,vb]; wc['1d']+=[list(C_EDGE),list(C_EDGE)]
                continue

            if elem.type in ('CTRIA3','CQUAD4'):
                if display_mode in ('wireframe','contour'):
                    # wireframe/contour: flat edges + alpha fill pass handles color
                    edge_col = [0.08,0.10,0.15] if display_mode=='contour' else wc2
                    for va,vb in _build_edge_lines(pos,elem.nodes,elem.type):
                        wv['2d']+=[va,vb]; wc['2d']+=[edge_col, edge_col]
                else:
                    # solid/hidden: full thick extrusion, proper edges
                    prop=model.properties.get(elem.pid); thick=0.0
                    if prop and prop.type=='PSHELL':
                        try: thick=float(str(prop.params.get('f3',0)).strip() or 0)
                        except: pass
                    tris=_shell_thick_tris(pos,elem.nodes,elem.type,thick) if thick>1e-12                          else _build_face_tris(pos,elem.nodes,elem.type)
                    for v0,v1,v2 in tris:
                        nm=_face_normal(v0,v1,v2)
                        for v in(v0,v1,v2): sv['2d'].append(v); sc['2d'].append(fc); sn['2d'].append(nm)
                    if thick > 1e-12:
                        for va,vb in _shell_thick_edges(pos,elem.nodes,elem.type,thick):
                            wv['2d']+=[va,vb]; wc['2d']+=[list(C_EDGE),list(C_EDGE)]
                    else:
                        for va,vb in _build_edge_lines(pos,elem.nodes,elem.type):
                            wv['2d']+=[va,vb]; wc['2d']+=[list(C_EDGE),list(C_EDGE)]
                continue

            if display_mode=='wireframe':
                edges = _cpyram_edges(pos,elem.nodes) if elem.type=='CPYRAM'                         else _build_edge_lines(pos,elem.nodes,elem.type)
                for va,vb in edges: wv['3d']+=[va,vb]; wc['3d']+=[wc2,wc2]
            else:
                if use_nodal_contour:
                    tris = _build_face_tris_with_nodes(pos, elem.nodes, elem.type)
                    for tri in tris:
                        (v0, n0), (v1, n1), (v2, n2) = tri
                        nm=_face_normal(v0,v1,v2)
                        for v, nid in ((v0, n0), (v1, n1), (v2, n2)):
                            sv['3d'].append(v)
                            sc['3d'].append(list(c_node.get(nid, fc)))
                            sn['3d'].append(nm)
                else:
                    tris = _cpyram_face_tris(pos,elem.nodes) if elem.type=='CPYRAM'                        else _build_face_tris(pos,elem.nodes,elem.type)
                    for v0,v1,v2 in tris:
                        nm=_face_normal(v0,v1,v2)
                        for v in(v0,v1,v2): sv['3d'].append(v); sc['3d'].append(fc); sn['3d'].append(nm)
                edge_c = list(C_EDGE) if display_mode=='solid' else [c*0.6 for c in fc]
                edges = _cpyram_edges(pos,elem.nodes) if elem.type=='CPYRAM'                         else _build_edge_lines(pos,elem.nodes,elem.type)
                for va,vb in edges: wv['3d']+=[va,vb]; wc['3d']+=[edge_c, edge_c]

        # Build fill geometry (wireframe=uniform, contour=per-element color)
        fv2=[]; fc2=[]; fn2=[]
        fv3=[]; fc3=[]; fn3=[]
        for eid,elem in model.elements.items():
            if elem.type in ('CTRIA3','CQUAD4'):
                if display_mode=='contour':
                    if eid in c_elem:
                        fill_c = list(c_elem[eid])
                    elif elem.nodes and elem.nodes[0] in c_node:
                        ns=[c_node[n] for n in elem.nodes if n in c_node]
                        fill_c=list(np.mean(ns,axis=0)) if ns else list(_DIM_FILL['2d'])
                    else:
                        fill_c=list(_DIM_FILL['2d'])
                else:
                    fill_c=list(_DIM_FILL['2d'])
                if display_mode=='contour' and eid not in c_elem and any(n in c_node for n in elem.nodes):
                    for tri in _build_face_tris_with_nodes(pos, elem.nodes, elem.type):
                        (v0, n0), (v1, n1), (v2, n2) = tri
                        nm=_face_normal(v0,v1,v2)
                        for v, nid in ((v0, n0), (v1, n1), (v2, n2)):
                            fv2.append(v); fc2.append(list(c_node.get(nid, fill_c))); fn2.append(nm)
                else:
                    for v0,v1,v2 in _build_face_tris(pos,elem.nodes,elem.type):
                        nm=_face_normal(v0,v1,v2)
                        for v in(v0,v1,v2): fv2.append(v); fc2.append(fill_c); fn2.append(nm)
            elif elem.type in ('CHEXA','CPENTA','CTETRA','CPYRAM'):
                if display_mode=='contour' and eid in c_elem:
                    fill_c=list(c_elem[eid])
                else:
                    fill_c=list(_DIM_FILL['3d'])
                if display_mode=='contour' and eid not in c_elem and any(n in c_node for n in elem.nodes):
                    face_tris = _build_face_tris_with_nodes(pos, elem.nodes, elem.type)
                    for tri in face_tris:
                        (v0, n0), (v1, n1), (v2, n2) = tri
                        nm=_face_normal(v0,v1,v2)
                        for v, nid in ((v0, n0), (v1, n1), (v2, n2)):
                            fv3.append(v); fc3.append(list(c_node.get(nid, fill_c))); fn3.append(nm)
                else:
                    face_tris = _cpyram_face_tris(pos,elem.nodes) if elem.type=='CPYRAM'                             else _build_face_tris(pos,elem.nodes,elem.type)
                    for v0,v1,v2 in face_tris:
                        nm=_face_normal(v0,v1,v2)
                        for v in(v0,v1,v2): fv3.append(v); fc3.append(fill_c); fn3.append(nm)

        s=model.scale()*.025; spv=[]; spc=[]
        seen=set()
        for spc_obj in model.spcs:
            if active_spc_sid and int(spc_obj.id) != int(active_spc_sid):
                continue
            nid=spc_obj.node_id
            if nid in seen: continue
            seen.add(nid); p=pos.get(nid)
            if p is None: continue
            for ax in[np.array([s,0,0],dtype=np.float32),np.array([0,s,0],dtype=np.float32),np.array([0,0,s],dtype=np.float32)]:
                spv+=[p-ax,p+ax]; spc+=[list(C_SPC),list(C_SPC)]

        for dim in('1d','2d','3d'):
            self._solid[dim]=self._up_solid(sv[dim],sc[dim],sn[dim])
            self._wire[dim] =self._up_lines(wv[dim],wc[dim])
        self._fill['2d']=self._up_fill(fv2,fc2,fn2)
        self._fill['3d']=self._up_fill(fv3,fc3,fn3)
        self._spc=self._up_lines(spv,spc)

    def draw(self,mvp,display_mode='wireframe',show_edges=True,show_spc=True,
             visible_dims=('1d','2d','3d'),show_undeformed=False):
        mvp_b=mvp.astype(np.float32).flatten().tobytes()
        L0=np.array([0.5,1.0,0.7],dtype=np.float32);  L0/=np.linalg.norm(L0)
        L1=np.array([-0.8,0.2,0.3],dtype=np.float32); L1/=np.linalg.norm(L1)
        L2=np.array([0.0,-0.6,-0.5],dtype=np.float32);L2/=np.linalg.norm(L2)

        # Alpha fill pass:
        # wireframe = uniform color 18% (hint of surface)
        # contour   = contour color 65% (colored semi-transparent)
        if display_mode in ('wireframe', 'contour'):
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            p = self._prog_fill
            p['u_mvp'].write(mvp_b)
            self.ctx.polygon_offset = (2.0, 2.0)
            if '2d' in visible_dims:
                vao,_,cnt = self._fill['2d']
                if vao and cnt > 0:
                    p['u_alpha'].value = 0.65 if display_mode=='contour' else 0.18
                    vao.render(moderngl.TRIANGLES)
            if '3d' in visible_dims:
                vao,_,cnt = self._fill['3d']
                if vao and cnt > 0:
                    if   display_mode == 'wireframe': p['u_alpha'].value = 0.15
                    elif display_mode == 'contour':   p['u_alpha'].value = 0.65
                    else:                             p['u_alpha'].value = 0.0
                    if p['u_alpha'].value > 0:
                        vao.render(moderngl.TRIANGLES)
            self.ctx.polygon_offset = (0.0, 0.0)
            self.ctx.disable(moderngl.BLEND)

        if display_mode in ('solid', 'contour'):
            p=self._prog_solid
            p['u_mvp'].write(mvp_b)
            p['u_light0'].value=tuple(L0); p['u_light1'].value=tuple(L1); p['u_light2'].value=tuple(L2)
            p['u_ambient'].value=0.38
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.polygon_offset=(1.0,1.0)
            dims_to_draw = visible_dims if display_mode == 'solid' else tuple(d for d in visible_dims if d == '1d')
            for dim in dims_to_draw:
                vao,_,cnt=self._solid[dim]
                if vao and cnt>0: vao.render(moderngl.TRIANGLES)
            self.ctx.polygon_offset=(0.0,0.0)

        p=self._prog_line; p['u_mvp'].write(mvp_b)
        self.ctx.enable(moderngl.DEPTH_TEST)
        lw=1.5 if display_mode in('solid','contour') else 1.0
        self.ctx.line_width=lw
        for dim in visible_dims:
            vao,_,cnt=self._wire[dim]
            if vao and cnt>0: vao.render(moderngl.LINES)
        self.ctx.line_width=1.0

        # Undeformed ghost (thin white wireframe)
        if show_undeformed:
            vao,_,cnt = self._undeformed
            if vao and cnt > 0:
                p['u_mvp'].write(mvp_b)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.line_width = 0.8
                vao.render(moderngl.LINES)
                self.ctx.line_width = 1.0
                self.ctx.enable(moderngl.DEPTH_TEST)

        vao,_,cnt=self._spc
        if show_spc and vao and cnt>0:
            self.ctx.line_width=2.0; vao.render(moderngl.LINES); self.ctx.line_width=1.0
