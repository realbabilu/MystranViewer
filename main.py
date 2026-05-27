"""
MYSTRAN Viewer - Clean stable version
Controls: Left drag=rotate, Right/Mid drag=pan, Scroll=zoom
          F=fit, R=reset, W=wireframe, S=solid/hidden, C=contour, O=ortho, Esc=quit
"""
# Suppress numpy/Python 3.13 MINGW warnings
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys as _sys
if _sys.version_info >= (3, 13):
    print(f"[Warning] Python 3.13 detected - recommended: Python 3.11/3.12")
    print()

import sys, os, ctypes
sys.path.insert(0, os.path.dirname(__file__))

import glfw
import moderngl
import numpy as np
import pyrr
from imgui_bundle import imgui

from parser.dat_parser import load_dat
from parser.f06_parser import load_f06
from parser.beam_section import section_area_inertias
from renderer.camera   import Camera
from renderer.mesh_renderer import MeshRenderer
from renderer.beam_diagram  import BeamDiagramRenderer, auto_beam_scale, beam_diagram_bounds, _hermite_beam_points
from gui.panels import (ViewerState, draw_menu_bar, draw_left_panel,
                        draw_right_panel, draw_legend, draw_node_info,
                        draw_model_browser_windows, draw_status_strip,
                        draw_contour_toolbar, draw_beam_diagram_panel)

def _win_addr(w): return ctypes.cast(w, ctypes.c_void_p).value


def _active_deform_scale(state):
    return float(state.deform_scale) if (getattr(state, 'show_deformed', True) and state.deform_scale > 0) else 0.0


def _result_deform_vec(result_type, d):
    if d is None:
        return None
    if result_type == 't1':
        return np.array([d.t1, 0.0, 0.0], dtype=np.float32)
    if result_type == 't2':
        return np.array([0.0, d.t2, 0.0], dtype=np.float32)
    if result_type == 't3':
        return np.array([0.0, 0.0, d.t3], dtype=np.float32)
    return d.translation.astype(np.float32)


def _beam_timoshenko_phi(model, elem):
    if model is None or elem is None or len(getattr(elem, 'nodes', [])) < 2:
        return None
    n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
    if n0 is None or n1 is None:
        return None
    L = float(np.linalg.norm(n1.xyz - n0.xyz))
    if L <= 1e-12:
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
    phi_y = 12.0 * mat.E * iz / (kappa * G * area * L * L)
    phi_z = 12.0 * mat.E * iy / (kappa * G * area * L * L)
    return float(phi_y), float(phi_z)


# ---------------------------------------------------------------------------
# Axis triad via ImGui draw list (no GL viewport change)
# ---------------------------------------------------------------------------
def _draw_axis_imgui(view_matrix, win_w, win_h, size=70, margin=14, right_panel_w=255, top_offset=58):
    cx = win_w - right_panel_w - margin - size
    cy = top_offset + margin + size
    rot = np.array(view_matrix, dtype=np.float32)
    rot[3,:3] = 0; rot[:3,3] = 0; rot[3,3] = 1
    proj = pyrr.matrix44.create_orthogonal_projection_matrix(
        -1.5, 1.5, -1.5, 1.5, -10, 10, dtype=np.float32)
    mvp = pyrr.matrix44.multiply(rot, proj)

    def project(v):
        # Use mvp.T to match GPU convention
        p = mvp.T @ np.array([*v, 1.0], dtype=np.float32)
        w = p[3] if abs(p[3]) > 1e-6 else 1.0
        return (cx + p[0]/w * size, cy - p[1]/w * size)

    axes = [([1,0,0], 0xFF5566FF, "X"), ([0,1,0], 0xFF44CC44, "Y"), ([0,0,1], 0xFFFF8833, "Z")]
    origin = project([0,0,0])
    dl = imgui.get_foreground_draw_list()
    dl.add_circle_filled(origin, size - 6, 0x44000000)
    for dir_vec, color, label in axes:
        tip = project(dir_vec)
        neg = project([-v*0.4 for v in dir_vec])
        dl.add_line(neg, tip, (color & 0x00FFFFFF) | 0x66000000, 1.5)
        dl.add_line(origin, tip, color, 2.5)
        dl.add_circle_filled(tip, 4.5, color)
        dl.add_text((tip[0]+5, tip[1]-7), color, label)


def _beam_section_cdef_dims(prop):
    sec = getattr(prop, 'section', None) if prop is not None else None
    if sec is None:
        return None
    shp = str(getattr(sec, 'shape', '')).upper()
    dims = list(getattr(sec, 'dims', []) or [])
    if shp == 'I' and len(dims) >= 6:
        H, bf1, bf2, tf1, tf2, tw = dims[:6]
        return float(max(bf1, bf2)), float(H), shp
    if shp in ('RECT', 'BAR') and len(dims) >= 2:
        return float(dims[0]), float(dims[1]), shp
    if shp == 'BOX' and len(dims) >= 2:
        return float(dims[0]), float(dims[1]), shp
    if shp in ('T', 'L') and len(dims) >= 2:
        return float(dims[0]), float(dims[1]), shp
    b = float(getattr(sec, 'b', 0.0) or 0.0)
    h = float(getattr(sec, 'h', 0.0) or 0.0)
    if b > 1e-12 and h > 1e-12:
        return b, h, shp or 'RECT'
    return None


def _draw_beam_cdef_panel(prop, win_w, right_panel_w=255, top_offset=58):
    dims = _beam_section_cdef_dims(prop)
    if dims is None:
        return
    b, h, shp = dims
    dl = imgui.get_foreground_draw_list()
    panel_w = 150.0
    panel_h = 122.0
    x0 = win_w - right_panel_w - 14.0 - panel_w
    y0 = top_offset + 14.0 + 70.0 * 2.0 + 14.0
    x1 = x0 + panel_w
    y1 = y0 + panel_h
    dl.add_rect_filled((x0, y0), (x1, y1), 0xCC202327, 4.0)
    dl.add_rect((x0, y0), (x1, y1), 0xAA9098A0, 4.0, 1.0, 0)
    dl.add_text((x0 + 8, y0 + 6), 0xFFDDE6EE, f"C/D/E/F ({shp})")

    sx0 = x0 + 22.0
    sy0 = y0 + 30.0
    sx1 = x1 - 24.0
    sy1 = y1 - 26.0
    aspect = abs(b) / max(abs(h), 1e-12)
    box_w = min((sx1 - sx0) * 0.82, (sy1 - sy0) * 0.82 * max(aspect, 0.35))
    box_h = min((sy1 - sy0) * 0.70, box_w / max(aspect, 0.35))
    cx = 0.5 * (sx0 + sx1)
    cy = 0.5 * (sy0 + sy1) + 4.0
    rx0 = cx - box_w * 0.5
    rx1 = cx + box_w * 0.5
    ry0 = cy - box_h * 0.5
    ry1 = cy + box_h * 0.5

    dl.add_rect((rx0, ry0), (rx1, ry1), 0xFFF0F0F0, 0.0, 1.5, 0)
    pts = {
        'F': (rx0, ry0),
        'C': (rx1, ry0),
        'E': (rx0, ry1),
        'D': (rx1, ry1),
    }
    for lab, pt in pts.items():
        dl.add_circle_filled(pt, 2.8, 0xFFF0F0F0)
        offs = {
            'F': (-10, -12), 'C': (6, -12),
            'E': (-10, 2), 'D': (6, 2),
        }[lab]
        dl.add_text((pt[0] + offs[0], pt[1] + offs[1]), 0xFFF8F8B0, lab)

    ax_x = cx
    ax_y = cy
    dl.add_line((ax_x, ax_y), (ax_x, ry0 - 12), 0xFF44CC44, 1.6)
    dl.add_triangle_filled((ax_x, ry0 - 16), (ax_x - 3, ry0 - 10), (ax_x + 3, ry0 - 10), 0xFF44CC44)
    dl.add_text((ax_x - 8, ry0 - 28), 0xFF44CC44, "y")
    dl.add_line((ax_x, ax_y), (rx1 + 16, ax_y), 0xFFFF8833, 1.6)
    dl.add_triangle_filled((rx1 + 20, ax_y), (rx1 + 14, ax_y - 3), (rx1 + 14, ax_y + 3), 0xFFFF8833)
    dl.add_text((rx1 + 24, ax_y - 7), 0xFFFF8833, "z")
    dl.add_text((rx0 + 8, ry1 + 8), 0xFFBFD0E0, f"b={_format_overlay_value(b)}")
    dl.add_text((rx1 + 8, cy - 6), 0xFFBFD0E0, f"h={_format_overlay_value(h)}")


# ---------------------------------------------------------------------------
# Notation overlays
# ---------------------------------------------------------------------------
def _project(xyz, mvp, win_w, win_h, ortho=False):
    # Use mvp.T because GPU receives row-major bytes read as column-major
    # (same convention as mesh_renderer.draw)
    m = mvp.T
    p = m @ np.array([*xyz, 1.0], dtype=np.float32)
    pw = p[3]
    if abs(pw) < 1e-9: return None
    nx, ny = p[0]/pw, p[1]/pw
    if abs(nx) > 1.4 or abs(ny) > 1.4: return None
    return ((nx + 1.0)*0.5*win_w, (1.0 - ny)*0.5*win_h)


def _elem_local_frame(elem, model):
    """Compute (origin, x1, x2, x3, scale) for element local axes.
    origin = midpoint for beam, centroid for shell/solid.
    scale  = 1/4 beam length, or avg edge length / 2 for shell/solid.
    """
    pos = {nid: model.nodes[nid].xyz for nid in elem.nodes if nid in model.nodes}
    if len(pos) < 2: return None
    pts = [pos[n].astype(np.float32) for n in elem.nodes if n in pos]

    if elem.type in ('CQUAD4','CTRIA3'):
        p0,p1 = pts[0], pts[1]
        p3 = pts[3] if elem.type=='CQUAD4' and len(pts)>=4 else pts[2]
        x1 = p1-p0; n=np.linalg.norm(x1)
        if n<1e-12: return None
        x1/=n
        x3 = np.cross(p1-p0, p3-p0).astype(np.float32)
        n3=np.linalg.norm(x3)
        if n3<1e-12: return None
        x3/=n3; x2=np.cross(x3,x1).astype(np.float32); x2/=(np.linalg.norm(x2)+1e-12)
        cen = np.mean(pts, axis=0)
        # Scale = avg edge / 2
        edges = [np.linalg.norm(pts[(i+1)%len(pts)]-pts[i]) for i in range(len(pts))]
        scale = float(np.mean(edges)) * 0.5
        return cen, x1, x2, x3, scale

    elif elem.type in ('CBAR','CBEAM','CROD'):
        p0,p1=pts[0],pts[1]
        L = np.linalg.norm(p1-p0)
        if L<1e-12: return None
        x1=(p1-p0)/L
        vo=getattr(elem,'v_orient',None)
        v=(vo/np.linalg.norm(vo)).astype(np.float32) if vo is not None and np.linalg.norm(vo)>1e-9           else (np.array([0,0,1],dtype=np.float32) if abs(x1[2])<0.9 else np.array([0,1,0],dtype=np.float32))
        x3=v-np.dot(v,x1)*x1; n3=np.linalg.norm(x3)
        if n3<1e-12:
            perp=np.array([1,0,0],dtype=np.float32) if abs(x1[0])<0.9 else np.array([0,1,0],dtype=np.float32)
            x3=np.cross(x1,perp); n3=np.linalg.norm(x3)
        x3=(x3/n3).astype(np.float32); x2=np.cross(x3,x1).astype(np.float32); x2/=(np.linalg.norm(x2)+1e-12)
        midpoint = (p0+p1)*0.5      # axes at beam midpoint
        scale = L * 0.25            # 1/4 beam length
        return midpoint, x1, x2, x3, scale

    elif elem.type in ('CHEXA','CPENTA','CTETRA'):
        cen = np.mean(pts, axis=0)
        edges = [np.linalg.norm(pts[(i+1)%len(pts)]-pts[i]) for i in range(min(4,len(pts)))]
        scale = float(np.mean(edges)) * 0.4
        return cen, np.array([1,0,0],dtype=np.float32), np.array([0,1,0],dtype=np.float32), np.array([0,0,1],dtype=np.float32), scale

    return None


def _draw_arrow_2d(dl, sc0, sc1, color, thickness=1.5, head=6.0):
    """Draw 2D arrow from sc0 to sc1 on ImGui draw list."""
    if sc0 is None or sc1 is None: return
    dl.add_line(sc0, sc1, color, thickness)
    dx=sc1[0]-sc0[0]; dy=sc1[1]-sc0[1]
    ln=np.sqrt(dx**2+dy**2)
    if ln<2: return
    dx/=ln; dy/=ln
    aw=head*0.5
    ax1=(sc1[0]-dx*head+(-dy)*aw, sc1[1]-dy*head+dx*aw)
    ax2=(sc1[0]-dx*head-(-dy)*aw, sc1[1]-dy*head-dx*aw)
    dl.add_triangle_filled(sc1, ax1, ax2, color)


def _format_reaction_value(val, engineering=False, zero_tol=1e-9):
    v = float(val)
    if abs(v) < zero_tol:
        return "0"
    v = abs(v)
    if engineering:
        return f"{v:.3e}"
    if abs(v) >= 1e6:
        return f"{v:.3e}"
    text = f"{v:.3f}".rstrip('0').rstrip('.')
    if text == "-0":
        text = "0"
    return text


def _format_overlay_value(val):
    v = float(val)
    if abs(v) < 1e-12:
        return "0.000"
    if abs(v) < 1e-2:
        return f"{v:.3e}"
    text = f"{v:.3f}"
    if len(text) <= 10:
        return text
    return f"{v:.3e}"


def _format_load_value(val):
    v = float(val)
    if abs(v) < 1e-12:
        return "0.0"
    text = f"{v:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
    if len(text) <= 8:
        return text
    return f"{v:.3e}"


def _pload1_type_label(ltype: str) -> str:
    t = str(ltype or '').strip().upper()
    if t.endswith('E'):
        base = t[:-1]
        return f"Local {base}"
    return f"Global {t}"


def _beam_local_axes(elem, p0w, p1w):
    axis = p1w - p0w
    L = float(np.linalg.norm(axis))
    if L < 1e-9:
        return None
    ex = (axis / L).astype(np.float32)
    vo = getattr(elem, 'v_orient', None)
    if vo is not None and np.linalg.norm(vo) > 1e-9:
        v = (vo / np.linalg.norm(vo)).astype(np.float32)
    else:
        v = np.array([0, 0, 1], dtype=np.float32) if abs(ex[2]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
    x3 = v - np.dot(v, ex) * ex
    n3 = float(np.linalg.norm(x3))
    if n3 < 1e-12:
        alt = np.array([1, 0, 0], dtype=np.float32) if abs(ex[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        x3 = np.cross(ex, alt).astype(np.float32)
        n3 = float(np.linalg.norm(x3))
        if n3 < 1e-12:
            return None
    x3 = (x3 / n3).astype(np.float32)
    x2 = np.cross(x3, ex).astype(np.float32)
    x2 = x2 / (np.linalg.norm(x2) + 1e-12)
    return ex, x2, x3, L


def _draw_moment_glyph_2d(dl, sc0, sc_axis, color, sign=1.0, size=10.0):
    if sc0 is None or sc_axis is None:
        return
    dx = float(sc_axis[0] - sc0[0])
    dy = float(sc_axis[1] - sc0[1])
    dn = float(np.hypot(dx, dy))
    if dn < 1e-6:
        return
    ux, uy = dx / dn, dy / dn
    px, py = -uy, ux
    half = size * 0.75
    a = (sc0[0] - ux * half, sc0[1] - uy * half)
    b = (sc0[0] + ux * half, sc0[1] + uy * half)
    dl.add_line(a, b, color, 1.4)
    r = size * 0.52
    c = (sc0[0] + px * r * 0.35, sc0[1] + py * r * 0.35)
    dl.add_circle(c, r, color, 0, 1.2)
    tang_sign = 1.0 if sign >= 0 else -1.0
    tip = (c[0] + px * r * tang_sign, c[1] + py * r * tang_sign)
    tail = (tip[0] - ux * 6.0 * tang_sign, tip[1] - uy * 6.0 * tang_sign)
    _draw_arrow_2d(dl, tail, tip, color, 1.3, 4.5)


def _with_alpha_u32(color: int, alpha: int) -> int:
    alpha = max(0, min(255, int(alpha)))
    return (alpha << 24) | (color & 0x00FFFFFF)


def _dist_to_seg_2d(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    den = abx*abx + aby*aby
    if den <= 1e-12:
        dx = px - ax
        dy = py - ay
        return float(np.hypot(dx, dy))
    t = ((px - ax)*abx + (py - ay)*aby) / den
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return float(np.hypot(px - qx, py - qy))


def _beam_hover_lines(model, elem, prop, length):
    lines = [f"Beam {elem.id}  PID {elem.pid}", f"L = {_format_overlay_value(length)}"]
    if getattr(elem, 'pa', '') or getattr(elem, 'pb', ''):
        rel = []
        if getattr(elem, 'pa', ''):
            rel.append(f"PA={elem.pa}")
        if getattr(elem, 'pb', ''):
            rel.append(f"PB={elem.pb}")
        lines.append("  ".join(rel))
    g0_ref = int(getattr(elem, 'g0_ref', 0) or 0)
    vo = getattr(elem, 'v_orient', None)
    if g0_ref > 0:
        lines.append(f"Orient: G0={g0_ref}")
    elif vo is not None and np.linalg.norm(vo) > 1e-9:
        lines.append(f"Orient: v=({_format_overlay_value(vo[0])}, {_format_overlay_value(vo[1])}, {_format_overlay_value(vo[2])})")
    if model is not None and len(getattr(elem, 'nodes', [])) >= 2:
        n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
        if n0 is not None and n1 is not None:
            p0 = n0.xyz.astype(np.float32); p1 = n1.xyz.astype(np.float32)
            ex = p1 - p0
            L = float(np.linalg.norm(ex))
            if L > 1e-12:
                ex = (ex / L).astype(np.float32)
                if vo is not None and np.linalg.norm(vo) > 1e-9:
                    v = (vo / np.linalg.norm(vo)).astype(np.float32)
                else:
                    v = np.array([0, 0, 1], dtype=np.float32) if abs(ex[2]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
                ez = v - np.dot(v, ex) * ex
                nez = float(np.linalg.norm(ez))
                if nez < 1e-12:
                    perp = np.array([1, 0, 0], dtype=np.float32) if abs(ex[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
                    ez = np.cross(ex, perp)
                    nez = float(np.linalg.norm(ez))
                if nez > 1e-12:
                    ez = (ez / nez).astype(np.float32)
                    ey = np.cross(ez, ex).astype(np.float32)
                    ey /= (np.linalg.norm(ey) + 1e-12)
                    lines.append(f"ex=({_format_overlay_value(ex[0])}, {_format_overlay_value(ex[1])}, {_format_overlay_value(ex[2])})")
                    lines.append(f"ey=({_format_overlay_value(ey[0])}, {_format_overlay_value(ey[1])}, {_format_overlay_value(ey[2])})")
                    lines.append(f"ez=({_format_overlay_value(ez[0])}, {_format_overlay_value(ez[1])}, {_format_overlay_value(ez[2])})")
    if prop is None:
        return lines
    if prop.type in ('PBAR', 'PBEAM', 'PROD'):
        lines.append(f"Type: {prop.type}")
        if prop.type == 'PROD':
            if 'f3' in prop.params:
                lines.append(f"A={_format_overlay_value(float(prop.params.get('f3', 0) or 0))}")
            if 'f4' in prop.params:
                lines.append(f"J={_format_overlay_value(float(prop.params.get('f4', 0) or 0))}")
        elif prop.type == 'PBAR':
            vals = []
            if 'f3' in prop.params: vals.append(f"A={_format_overlay_value(float(prop.params.get('f3', 0) or 0))}")
            if 'f4' in prop.params: vals.append(f"I1={_format_overlay_value(float(prop.params.get('f4', 0) or 0))}")
            if 'f5' in prop.params: vals.append(f"I2={_format_overlay_value(float(prop.params.get('f5', 0) or 0))}")
            if 'f6' in prop.params: vals.append(f"J={_format_overlay_value(float(prop.params.get('f6', 0) or 0))}")
            if vals:
                lines.extend(vals[:2])
                if len(vals) > 2:
                    lines.append("  ".join(vals[2:]))
        elif prop.type == 'PBEAM':
            vals = []
            if 'f3' in prop.params: vals.append(f"A={_format_overlay_value(float(prop.params.get('f3', 0) or 0))}")
            if 'f4' in prop.params: vals.append(f"I1={_format_overlay_value(float(prop.params.get('f4', 0) or 0))}")
            if 'f5' in prop.params: vals.append(f"I2={_format_overlay_value(float(prop.params.get('f5', 0) or 0))}")
            if 'f7' in prop.params: vals.append(f"J={_format_overlay_value(float(prop.params.get('f7', 0) or 0))}")
            if vals:
                lines.extend(vals[:2])
                if len(vals) > 2:
                    lines.append("  ".join(vals[2:]))
        return lines
    sec = getattr(prop, 'section', None)
    if sec is None:
        return lines
    lines.append(f"{sec.label}  A = {_format_overlay_value(getattr(sec, 'area', 0.0))}")
    dims = list(getattr(sec, 'dims', []) or [])
    shp = str(getattr(sec, 'shape', '')).upper()
    if shp == 'I' and len(dims) >= 6:
        H, bf1, bf2, tf1, tf2, tw = dims[:6]
        lines.append(f"H={_format_overlay_value(H)}  bf1={_format_overlay_value(bf1)}  bf2={_format_overlay_value(bf2)}")
        lines.append(f"tf1={_format_overlay_value(tf1)}  tf2={_format_overlay_value(tf2)}  tw={_format_overlay_value(tw)}")
    elif shp == 'RECT' and len(dims) >= 2:
        lines.append(f"b={_format_overlay_value(dims[0])}  h={_format_overlay_value(dims[1])}")
    elif shp == 'CIRCLE' and len(dims) >= 1:
        lines.append(f"d={_format_overlay_value(dims[0])}")
    elif shp == 'PIPE' and len(dims) >= 2:
        lines.append(f"do={_format_overlay_value(dims[0])}  di={_format_overlay_value(dims[1])}")
    elif shp == 'BOX' and len(dims) >= 4:
        lines.append(f"b={_format_overlay_value(dims[0])}  h={_format_overlay_value(dims[1])}")
        lines.append(f"t1={_format_overlay_value(dims[2])}  t2={_format_overlay_value(dims[3])}")
    elif shp == 'T' and len(dims) >= 4:
        lines.append(f"bf={_format_overlay_value(dims[0])}  H={_format_overlay_value(dims[1])}")
        lines.append(f"tf={_format_overlay_value(dims[2])}  tw={_format_overlay_value(dims[3])}")
    elif shp == 'L' and len(dims) >= 4:
        lines.append(f"b={_format_overlay_value(dims[0])}  h={_format_overlay_value(dims[1])}")
        lines.append(f"tb={_format_overlay_value(dims[2])}  th={_format_overlay_value(dims[3])}")
    else:
        lines.append(f"b={_format_overlay_value(getattr(sec, 'b', 0.0))}  h={_format_overlay_value(getattr(sec, 'h', 0.0))}")
    return lines


def _draw_beam_hover_info(model, mvp, iw, ih, ortho, state):
    if hasattr(state, 'hover_status'):
        state.hover_status = ""
    if model is None or state.display_mode not in ('solid', 'beam', 'beam_v2', 'contour'):
        return
    mx = float(getattr(state, 'mouse_x', -1.0))
    my = float(getattr(state, 'mouse_y', -1.0))
    if mx < 270 or my < 20 or mx > iw - 255 or my > ih:
        return

    if state.display_mode in ('beam', 'beam_v2'):
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'diagram_data', lambda: getattr(diag, 'beam_data', {}))() if diag is not None else {}
        if not beam_data:
            return
        result_key = getattr(state, 'beam_result_key', 'bm1')
        best = None
        best_d = 18.0
        for eid, bd in beam_data.items():
            elem = model.elements.get(eid)
            if elem is None or len(elem.nodes) < 2 or not bd.stations:
                continue
            n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
            if n0 is None or n1 is None:
                continue
            p0 = n0.xyz.astype(np.float32)
            p1 = n1.xyz.astype(np.float32)
            sc0 = _project(p0, mvp, iw, ih, ortho)
            sc1 = _project(p1, mvp, iw, ih, ortho)
            if sc0 is None or sc1 is None:
                continue
            d = _dist_to_seg_2d(mx, my, sc0[0], sc0[1], sc1[0], sc1[1])
            if d < best_d:
                best = (eid, elem, bd)
                best_d = d
        if best is None:
            return
        eid, elem, bd = best
        prop = model.properties.get(elem.pid)
        vals = [float(getattr(st, result_key, 0.0)) for st in bd.stations]
        if not vals:
            return
        i_max = max(range(len(vals)), key=lambda i: vals[i])
        i_min = min(range(len(vals)), key=lambda i: vals[i])
        v0 = vals[0]
        v1 = vals[-1]
        s0 = float(bd.stations[0].sd)
        s1 = float(bd.stations[-1].sd)
        s_max = float(bd.stations[i_max].sd)
        s_min = float(bd.stations[i_min].sd)
        label_map = dict(getattr(diag, 'available_results', lambda: [])()) if diag is not None else {}
        title = label_map.get(result_key, result_key)
        if state.display_mode == 'beam_v2':
            title += " [v2]"
        lines = [f"Beam {eid} - {title}"]
        lines.append(f"End A  s={s0:.3f}  v={_format_overlay_value(v0)}")
        lines.append(f"End B  s={s1:.3f}  v={_format_overlay_value(v1)}")
        lines.append(f"Local MAX  s={s_max:.3f}  v={_format_overlay_value(vals[i_max])}")
        lines.append(f"Local MIN  s={s_min:.3f}  v={_format_overlay_value(vals[i_min])}")
        state.hover_status = lines[0]
        _draw_hover_box(lines, mx, my, iw, ih)
        if prop is not None:
            _draw_beam_cdef_panel(prop, iw)
        return

    best = None
    best_d = 14.0
    for elem in model.elements.values():
        if elem.type not in ('CBAR', 'CBEAM', 'CROD') or len(elem.nodes) < 2:
            continue
        n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
        if n0 is None or n1 is None:
            continue
        sc0 = _project(n0.xyz.astype(np.float32), mvp, iw, ih, ortho)
        sc1 = _project(n1.xyz.astype(np.float32), mvp, iw, ih, ortho)
        if sc0 is None or sc1 is None:
            continue
        d = _dist_to_seg_2d(mx, my, sc0[0], sc0[1], sc1[0], sc1[1])
        if d < best_d:
            L = float(np.linalg.norm(n1.xyz - n0.xyz))
            prop = model.properties.get(elem.pid)
            best = (elem, prop, L)
            best_d = d
    if best is None:
        return
    elem, prop, L = best
    lines = _beam_hover_lines(model, elem, prop, L)
    if not lines:
        return
    if state.display_mode == 'contour':
        _draw_beam_cdef_panel(prop, iw)
        return
    state.hover_status = lines[0]
    _draw_hover_box(lines, mx, my, iw, ih)
    _draw_beam_cdef_panel(prop, iw)


def _nearest_node_hover(model, mvp, iw, ih, ortho, mx, my, best_d=14.0, xyz_fn=None):
    best = None
    for nid, node in model.nodes.items():
        xyz = xyz_fn(nid, node) if xyz_fn is not None else node.xyz.astype(np.float32)
        if xyz is None:
            continue
        sc = _project(np.asarray(xyz, dtype=np.float32), mvp, iw, ih, ortho)
        if sc is None or not _in_safe_zone(sc, iw):
            continue
        d = float(np.hypot(sc[0] - mx, sc[1] - my))
        if d < best_d:
            best = (nid, node, sc)
            best_d = d
    return best


def _nearest_beam_displacement_hover(model, results, subcase, deform_scale, mvp, iw, ih, ortho, mx, my,
                                     best_d=14.0, result_type='displacement', beam_end_data=None):
    if model is None or results is None or deform_scale <= 0.0:
        return None
    disp = getattr(results, 'displacements', {}).get(subcase, {})
    if not disp:
        return None
    best = None
    for eid, elem in model.elements.items():
        if elem.type not in ('CBAR', 'CBEAM', 'CROD') or len(elem.nodes) < 2:
            continue
        n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
        d0 = disp.get(elem.nodes[0]); d1 = disp.get(elem.nodes[1])
        if n0 is None or n1 is None or d0 is None or d1 is None:
            continue
        p0 = n0.xyz.astype(np.float32)
        p1 = n1.xyz.astype(np.float32)
        if elem.type in ('CBAR', 'CBEAM'):
            phi_yz = _beam_timoshenko_phi(model, elem)
            load_sid = int(getattr(model, 'subcase_loads', {}).get(subcase, 0) or 0)
            base_pts, def_pts = _hermite_beam_points(
                p0, p1, d0, d1, deform_scale, nsamp=25,
                v_orient=getattr(elem, 'v_orient', None),
                result_type=result_type, phi_yz=phi_yz,
                model=model, elem=elem, load_sid=load_sid,
                beam_end_data=beam_end_data)
        else:
            v0 = _result_deform_vec(result_type, d0)
            v1 = _result_deform_vec(result_type, d1)
            base_pts = np.array([p0, p1], dtype=np.float32)
            def_pts = np.array([
                p0 + (v0 if v0 is not None else 0.0) * deform_scale,
                p1 + (v1 if v1 is not None else 0.0) * deform_scale
            ], dtype=np.float32)
        scr = []
        for pt in def_pts:
            sc = _project(pt, mvp, iw, ih, ortho)
            if sc is None:
                scr = []
                break
            scr.append(sc)
        if len(scr) < 2:
            continue
        local_best_d = best_d
        local_best_idx = -1
        for i in range(len(scr) - 1):
            dseg = _dist_to_seg_2d(mx, my, scr[i][0], scr[i][1], scr[i+1][0], scr[i+1][1])
            if dseg < local_best_d:
                local_best_d = dseg
                local_best_idx = i
        if local_best_idx >= 0:
            i = local_best_idx
            q_base = 0.5 * (base_pts[i] + base_pts[i+1])
            q_def = 0.5 * (def_pts[i] + def_pts[i+1])
            q_disp = (q_def - q_base) / deform_scale
            mag = float(np.linalg.norm(q_disp))
            s = (i + 0.5) / max(1, (len(def_pts) - 1))
            best = (eid, float(s), q_disp.astype(np.float32), mag)
            best_d = local_best_d
    return best


def _nearest_elem_hover(model, mvp, iw, ih, ortho, mx, my, best_d=20.0, xyz_fn=None):
    best = None
    for eid, elem in model.elements.items():
        pts = []
        for nid in elem.nodes:
            node = model.nodes.get(nid)
            if node is None:
                continue
            if xyz_fn is not None:
                xyz = xyz_fn(nid, node)
                if xyz is None:
                    continue
                pts.append(np.asarray(xyz, dtype=np.float32))
            else:
                pts.append(node.xyz.astype(np.float32))
        if not pts:
            continue
        cen = np.mean(np.array(pts, dtype=np.float32), axis=0)
        sc = _project(cen, mvp, iw, ih, ortho)
        if sc is None or not _in_safe_zone(sc, iw):
            continue
        d = float(np.hypot(sc[0] - mx, sc[1] - my))
        if d < best_d:
            best = (eid, elem, sc)
            best_d = d
    return best


def _draw_hover_box(lines, mx, my, iw, ih):
    if not lines:
        return
    dl = imgui.get_foreground_draw_list()
    char_w = 7.2
    w = max(160.0, max(len(s) for s in lines) * char_w + 14.0)
    h = len(lines) * 18.0 + 10.0
    x = min(mx + 14.0, iw - 255 - w - 8.0)
    y = max(24.0, min(my + 12.0, ih - h - 8.0))
    dl.add_rect_filled((x, y), (x + w, y + h), 0xDD1C1F24, 4.0)
    dl.add_rect((x, y), (x + w, y + h), 0xFF5A6470, 4.0, 1.2)
    for i, line in enumerate(lines):
        col = 0xFFBDE8FF if i == 0 else 0xFFE8E8E8
        dl.add_text((x + 7.0, y + 5.0 + i * 18.0), col, line)


def _spc_dofs_for_node(state, model, nid: int) -> str:
    spc_sid = _resolved_spc_sid(state)
    dofs = set()
    for spc in getattr(model, 'spcs', []):
        if int(spc.node_id) != int(nid):
            continue
        if spc_sid and int(spc.id) != spc_sid:
            continue
        for ch in str(spc.dofs):
            if ch.isdigit():
                dofs.add(ch)
    return ''.join(sorted(dofs))


def _draw_result_hover_info(model, mvp, iw, ih, ortho, state):
    state.hover_status = ""
    if model is None or state.display_mode not in ('wireframe', 'contour'):
        return
    mx = float(getattr(state, 'mouse_x', -1.0))
    my = float(getattr(state, 'mouse_y', -1.0))
    if mx < 270 or my < 20 or mx > iw - 255 or my > ih:
        return
    sc = state.subcase
    rt = state.result_type
    r = state.results

    nodal_types = {'displacement', 't1', 't2', 't3',
                   'nodal_vm', 'noxx', 'noyy', 'ntxy', 'nomax', 'nomin',
                   'nfx', 'nfy', 'nfxy', 'nmx', 'nmy', 'nmxy', 'nqx', 'nqy'}
    elem_types = {'von_mises', 'oxx', 'oyy', 'txy', 'omax', 'omin',
                  'von_mises_top', 'von_mises_bottom',
                  'sxc', 'sxd', 'sxe', 'sxf', 'smax', 'smin',
                  'fx', 'fy', 'fxy', 'mx', 'my', 'mxy', 'qx', 'qy'}

    if state.display_mode == 'wireframe':
        def _hover_xyz(nid, node):
            if _active_deform_scale(state) <= 0.0 or r is None or sc not in getattr(r, 'displacements', {}):
                return node.xyz.astype(np.float32)
            d = r.displacements[sc].get(nid)
            if d is None:
                return node.xyz.astype(np.float32)
            dv = _result_deform_vec(rt, d)
            return node.xyz.astype(np.float32) + (dv if dv is not None else 0.0) * _active_deform_scale(state)

        hit = _nearest_node_hover(model, mvp, iw, ih, ortho, mx, my, 14.0, xyz_fn=_hover_xyz)
        if hit is not None:
            nid, node, _ = hit
            lines = [f"Node {nid}"]
            active_spc_nodes = _active_spc_nodes(state, model)
            bc_sid = _resolved_spc_sid(state)
            dof_str = _spc_dofs_for_node(state, model, nid) if nid in active_spc_nodes else ""
            if state.show_constraints and nid in active_spc_nodes:
                lines[0] = f"BC set {bc_sid} - SPC {dof_str or '?'}"
            if r is not None and sc in r.displacements:
                d = r.displacements[sc].get(nid)
                if d is not None and _active_deform_scale(state) > 0.0:
                    lines.append(f"|u|={_format_overlay_value(d.magnitude)}")
                    lines.append(f"T1={_format_overlay_value(d.t1)}  T2={_format_overlay_value(d.t2)}  T3={_format_overlay_value(d.t3)}")
            if r is not None:
                rx = getattr(r, 'reactions', {}).get(sc, {}).get(nid, {})
                if nid in active_spc_nodes and (state.show_reaction_forces or state.show_reaction_moments) and rx:
                    if state.show_reaction_forces and state.show_reaction_moments:
                        lines[0] = f"BC set {bc_sid} - Total Reaction"
                        lines.append(f"TX={_format_reaction_value(rx.get('tx', 0.0), state.reaction_engineering)}  TY={_format_reaction_value(rx.get('ty', 0.0), state.reaction_engineering)}  TZ={_format_reaction_value(rx.get('tz', 0.0), state.reaction_engineering)}")
                        lines.append(f"RX={_format_reaction_value(rx.get('rx', 0.0), state.reaction_engineering)}  RY={_format_reaction_value(rx.get('ry', 0.0), state.reaction_engineering)}  RZ={_format_reaction_value(rx.get('rz', 0.0), state.reaction_engineering)}")
                    elif state.show_reaction_forces:
                        lines[0] = f"BC set {bc_sid} - Reaction TX/TY/TZ"
                        lines.append(f"TX={_format_reaction_value(rx.get('tx', 0.0), state.reaction_engineering)}  TY={_format_reaction_value(rx.get('ty', 0.0), state.reaction_engineering)}  TZ={_format_reaction_value(rx.get('tz', 0.0), state.reaction_engineering)}")
                    elif state.show_reaction_moments:
                        lines[0] = f"BC set {bc_sid} - Reaction RX/RY/RZ"
                        lines.append(f"RX={_format_reaction_value(rx.get('rx', 0.0), state.reaction_engineering)}  RY={_format_reaction_value(rx.get('ry', 0.0), state.reaction_engineering)}  RZ={_format_reaction_value(rx.get('rz', 0.0), state.reaction_engineering)}")
            load_sid = _resolved_load_sid(state)
            if state.show_forces:
                for force in getattr(model, 'forces', []):
                    if force.node_id != nid:
                        continue
                    if load_sid and int(force.sid) != load_sid:
                        continue
                    vec = force.direction.astype(np.float32) * float(force.magnitude)
                    lines[0] = f"Load set {load_sid} - Point Force"
                    lines.append(f"F=({_format_load_value(vec[0])}, {_format_load_value(vec[1])}, {_format_load_value(vec[2])})")
            if state.show_moments:
                for moment in getattr(model, 'moments', []):
                    if moment.node_id != nid:
                        continue
                    if load_sid and int(moment.sid) != load_sid:
                        continue
                    vec = moment.direction.astype(np.float32) * float(moment.magnitude)
                    lines[0] = f"Load set {load_sid} - Moment Load"
                    lines.append(f"M=({_format_load_value(vec[0])}, {_format_load_value(vec[1])}, {_format_load_value(vec[2])})")
            if len(lines) == 1 and state.show_spc and nid in active_spc_nodes and not (state.show_reaction_forces or state.show_reaction_moments):
                lines[0] = f"BC set {bc_sid} - SPC {dof_str or '?'}"
            if len(lines) > 1:
                state.hover_status = lines[0]
                _draw_hover_box(lines, mx, my, iw, ih)
                return

        if r is not None and sc in getattr(r, 'displacements', {}) and _active_deform_scale(state) > 0.0:
            bhit = _nearest_beam_displacement_hover(
                model, r, sc, _active_deform_scale(state), mvp, iw, ih, ortho, mx, my, 14.0, rt,
                getattr(getattr(state, 'beam_diagram', None), 'beam_data', {}))
            if bhit is not None:
                eid, s_beam, q_disp, mag = bhit
                lines = [f"Displacement beam {eid}  s={s_beam:.3f}"]
                lines.append(f"|u|={_format_overlay_value(mag)}")
                lines.append(f"T1={_format_overlay_value(q_disp[0])}  T2={_format_overlay_value(q_disp[1])}  T3={_format_overlay_value(q_disp[2])}")
                state.hover_status = lines[0]
                _draw_hover_box(lines, mx, my, iw, ih)
                return

        if state.show_pressure:
            hit_e = _nearest_elem_hover(model, mvp, iw, ih, ortho, mx, my, 20.0)
            if hit_e is None:
                return
            eid, elem, _ = hit_e
            load_sid = _resolved_load_sid(state)
            lines = [f"Elem {eid}"]
            for pl in getattr(model, 'pload1s', []):
                if int(pl.get('eid', 0)) != eid:
                    continue
                if load_sid and int(pl.get('sid', 0)) != load_sid:
                    continue
                lines[0] = f"Load set {load_sid} - Element Load"
                tlabel = _pload1_type_label(pl.get('type', 'PLOAD1'))
                p1 = float(pl.get('p1', 0.0))
                p2 = float(pl.get('p2', p1))
                x1 = float(pl.get('x1', 0.0))
                x2 = float(pl.get('x2', x1))
                if abs(x2 - x1) <= 1e-9:
                    lines.append(f"{tlabel} point  x={_format_load_value(x1)}  p={_format_load_value(p1)}")
                elif abs(p1 - p2) <= 1e-9:
                    lines.append(f"{tlabel} dist  x={_format_load_value(x1)}..{_format_load_value(x2)}  p={_format_load_value(p1)}")
                else:
                    lines.append(f"{tlabel} taper  x={_format_load_value(x1)}..{_format_load_value(x2)}  p={_format_load_value(p1)}..{_format_load_value(p2)}")
            if len(lines) > 1:
                state.hover_status = lines[0]
                _draw_hover_box(lines, mx, my, iw, ih)
            return

    def _hover_xyz(nid, node):
        if _active_deform_scale(state) <= 0.0 or r is None or sc not in getattr(r, 'displacements', {}):
            return node.xyz.astype(np.float32)
        d = r.displacements[sc].get(nid)
        if d is None:
            return node.xyz.astype(np.float32)
        dv = _result_deform_vec(rt, d)
        return node.xyz.astype(np.float32) + (dv if dv is not None else 0.0) * _active_deform_scale(state)

    if rt in nodal_types:
        if r is None:
            return
        hit = _nearest_node_hover(model, mvp, iw, ih, ortho, mx, my, 14.0, xyz_fn=_hover_xyz)
        if hit is None:
            if rt in ('displacement', 't1', 't2', 't3') and sc in getattr(r, 'displacements', {}) and _active_deform_scale(state) > 0.0:
                bhit = _nearest_beam_displacement_hover(
                    model, r, sc, _active_deform_scale(state), mvp, iw, ih, ortho, mx, my, 14.0, rt,
                    getattr(getattr(state, 'beam_diagram', None), 'beam_data', {}))
                if bhit is not None:
                    eid, s_beam, q_disp, mag = bhit
                    lines = [f"Displacement beam {eid}  s={s_beam:.3f}"]
                    if rt == 'displacement':
                        lines.append(f"|u|={_format_overlay_value(mag)}")
                        lines.append(f"TX={_format_overlay_value(q_disp[0])}  TY={_format_overlay_value(q_disp[1])}  TZ={_format_overlay_value(q_disp[2])}")
                    else:
                        comp_idx = {'t1': 0, 't2': 1, 't3': 2}[rt]
                        lines.append(f"{rt.upper()}={_format_overlay_value(float(q_disp[comp_idx]))}")
                    state.hover_status = lines[0]
                    _draw_hover_box(lines, mx, my, iw, ih)
                return
            return
        nid, node, _ = hit
        lines = [f"Node {nid}"]
        if rt == 'displacement' and sc in r.displacements:
            d = r.displacements[sc].get(nid)
            if d is not None:
                lines[0] = f"Displacement node {nid}"
                lines.append(f"|u|={_format_overlay_value(d.magnitude)}")
                lines.append(f"TX={_format_overlay_value(d.t1)}  TY={_format_overlay_value(d.t2)}  TZ={_format_overlay_value(d.t3)}")
        elif rt in ('t1', 't2', 't3') and sc in r.displacements:
            d = r.displacements[sc].get(nid)
            if d is not None:
                comp = {'t1': d.t1, 't2': d.t2, 't3': d.t3}[rt]
                lines[0] = f"Displacement node {nid}"
                lines.append(f"{rt.upper()}={_format_overlay_value(comp)}")
        elif rt in ('nodal_vm', 'noxx', 'noyy', 'ntxy', 'nomax', 'nomin') and sc in r.stresses:
            nav = state.nodal_stress_components()
            key = {'nodal_vm': 'von_mises', 'noxx': 'oxx', 'noyy': 'oyy', 'ntxy': 'txy', 'nomax': 'omax', 'nomin': 'omin'}[rt]
            if nid in nav:
                lines[0] = f"Stress {key.upper()} node {nid}"
                lines.append(f"{key.upper()}={_format_overlay_value(nav[nid].get(key, 0.0))}")
        elif rt in ('nfx', 'nfy', 'nfxy', 'nmx', 'nmy', 'nmxy', 'nqx', 'nqy') and sc in getattr(r, 'forces', {}):
            nav = state.nodal_force_components()
            key = {'nfx':'fx','nfy':'fy','nfxy':'fxy','nmx':'mx','nmy':'my','nmxy':'mxy','nqx':'qx','nqy':'qy'}[rt]
            if nid in nav:
                lines[0] = f"Force {key.upper()} node {nid}"
                lines.append(f"{key.upper()}={_format_overlay_value(nav[nid].get(key, 0.0))}")
        if len(lines) <= 1:
            return
        state.hover_status = lines[0]
        _draw_hover_box(lines, mx, my, iw, ih)
        return

    if rt in elem_types:
        if r is None:
            return
        hit = _nearest_elem_hover(model, mvp, iw, ih, ortho, mx, my, 20.0, xyz_fn=_hover_xyz)
        if hit is None:
            return
        eid, elem, _ = hit
        lines = [f"Elem {eid}"]
        if rt in ('von_mises', 'oxx', 'oyy', 'txy', 'omax', 'omin', 'von_mises_top', 'von_mises_bottom') and sc in r.stresses:
            es = r.stresses[sc].get(eid)
            if es is not None and hasattr(es, 'values') and not _is_beam_stress_obj(es):
                val = es.von_mises if rt == 'von_mises' else es.values.get(rt, es.von_mises)
                lines[0] = f"Stress {rt.upper()} element {eid}"
                lines.append(f"{rt.upper()}={_format_overlay_value(val)}")
        elif rt in ('sxc', 'sxd', 'sxe', 'sxf', 'smax', 'smin'):
            diag = getattr(state, 'beam_diagram', None)
            bd = getattr(diag, 'beam_data', {}).get(eid) if diag is not None else None
            if bd is not None and getattr(bd, 'stations', None):
                lines[0] = f"Beam stress {rt.upper()} element {eid}"
                for st in bd.stations:
                    lines.append(f"s={st.sd:.3f}  {rt.upper()}={_format_overlay_value(float(getattr(st, rt, 0.0)))}")
        elif rt in ('fx', 'fy', 'fxy', 'mx', 'my', 'mxy', 'qx', 'qy') and sc in getattr(r, 'forces', {}):
            ef = r.forces[sc].get(eid)
            if ef is not None and hasattr(ef, 'values'):
                lines[0] = f"Force {rt.upper()} element {eid}"
                lines.append(f"{rt.upper()}={_format_overlay_value(ef.values.get(rt, 0.0))}")
        if len(lines) <= 1:
            return
        state.hover_status = lines[0]
        _draw_hover_box(lines, mx, my, iw, ih)


def _eigen_result_summary(ed):
    if not ed:
        return ""
    if ed.get('is_buckling'):
        eig = ed.get('eigenvalue', ed.get('load_factor', 0.0))
        return f"Buckling - Eigenvalue {eig:.4g}"
    hz = ed.get('freq_hz', 0.0)
    return f"Mode {ed.get('mode', 0)} - {hz:.3g} Hz"


def _is_beam_stress_obj(es) -> bool:
    return getattr(es, 'elem_type', '').upper() in ('CBAR', 'CBEAM')


def _beam_release_labels(dofs: str):
    out = []
    for ch in str(dofs or ''):
        if ch == '1':
            out.append('A1')
        elif ch == '2':
            out.append('V2')
        elif ch == '3':
            out.append('V3')
        elif ch == '4':
            out.append('T1')
        elif ch == '5':
            out.append('M2')
        elif ch == '6':
            out.append('M3')
    return out


def _mesh_display_mode(display_mode: str) -> str:
    """Map UI-only display modes to mesh renderer modes."""
    if display_mode in ('beam', 'beam_v2'):
        return 'wireframe'
    return display_mode


def _model_has_beams(model) -> bool:
    if model is None:
        return False
    return any(elem.type in ('CBAR', 'CBEAM', 'CROD') for elem in model.elements.values())


def _stress_nodal_components(state):
    return state.nodal_stress_components() if hasattr(state, 'nodal_stress_components') else {}


def _force_nodal_components(state):
    return state.nodal_force_components() if hasattr(state, 'nodal_force_components') else {}


def _resolved_load_sid(state):
    mode = getattr(state, 'load_filter_mode', 'all')
    if mode == 'manual':
        return int(getattr(state, 'active_load_case', 0) or 0)
    if mode == 'subcase':
        model = getattr(state, 'model', None)
        if model is None:
            return 0
        return int(getattr(model, 'subcase_loads', {}).get(getattr(state, 'subcase', 0), 0) or 0)
    return 0


def _resolved_spc_sid(state):
    mode = getattr(state, 'spc_filter_mode', 'all')
    if mode == 'manual':
        return int(getattr(state, 'active_spc_case', 0) or 0)
    if mode == 'subcase':
        model = getattr(state, 'model', None)
        if model is None:
            return 0
        return int(getattr(model, 'subcase_spcs', {}).get(getattr(state, 'subcase', 0), 0) or 0)
    return 0


def _active_spc_nodes(state, model):
    spc_sid = _resolved_spc_sid(state)
    nodes = set()
    for spc in getattr(model, 'spcs', []):
        if spc_sid and int(spc.id) != spc_sid:
            continue
        nodes.add(int(spc.node_id))
    return nodes


def _pload1_station_fraction(pl, beam_length: float) -> tuple[float, float]:
    x1 = float(pl.get('x1', 0.0))
    x2 = float(pl.get('x2', x1))
    use_fraction = str(pl.get('scale', '')).strip().upper().startswith('FR')

    def _station_t(x):
        if use_fraction:
            return float(np.clip(x, 0.0, 1.0))
        if beam_length <= 1e-12:
            return 0.0
        return float(np.clip(x / beam_length, 0.0, 1.0))

    t1 = _station_t(x1)
    t2 = _station_t(x2)
    return (t1, t2) if t1 <= t2 else (t2, t1)


def _active_load_force_scale(state, model) -> float:
    load_sid = _resolved_load_sid(state)
    mags = []
    for force in getattr(model, 'forces', []):
        if load_sid and force.sid != load_sid:
            continue
        mags.append(abs(float(force.magnitude)))
    for pl in getattr(model, 'pload1s', []):
        if load_sid and pl.get('sid', 0) != load_sid:
            continue
        elem = model.elements.get(pl.get('eid', 0))
        if elem is None or len(elem.nodes) < 2:
            continue
        n0 = model.nodes.get(elem.nodes[0])
        n1 = model.nodes.get(elem.nodes[1])
        if n0 is None or n1 is None:
            continue
        L = float(np.linalg.norm(n1.xyz - n0.xyz))
        ta, tb = _pload1_station_fraction(pl, L)
        p1 = abs(float(pl.get('p1', 0.0)))
        p2 = abs(float(pl.get('p2', pl.get('p1', 0.0))))
        if abs(tb - ta) <= 1e-6:
            mags.append(max(p1, p2))
        else:
            mags.append(0.5 * (p1 + p2) * max((tb - ta) * L, 0.0))
    return max(mags) if mags else 0.0


def _parse_f06_eigen_data(path, results):
    import re
    eigen_data = {}
    if results is None:
        return eigen_data
    sub_re = re.compile(r'(?:SUBCASE|OUTPUT FOR SUBCASE)\s+(\d+)', re.I)
    mode_row_re = re.compile(
        r'^\s*(\d+)\s+(\d+)\s+([+-]?\d+\.\d+E[+-]?\d+)\s+([+-]?\d+\.\d+E[+-]?\d+)\s+([+-]?\d+\.\d+E[+-]?\d+)')
    current_sc = 1
    in_table = False
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                m = sub_re.search(line)
                if m:
                    current_sc = int(m.group(1))
                if 'R E A L   E I G E N V A L U E S' in line:
                    in_table = True
                    continue
                if in_table:
                    mr = mode_row_re.match(line)
                    if mr:
                        mode_num = int(mr.group(1))
                        ev_nm = float(mr.group(3))
                        cycles = float(mr.group(5))
                        is_buckling = 'BUCKLING' in path.upper()
                        mode_key = current_sc * 1000 + mode_num
                        target_key = mode_key if mode_key in results.displacements else current_sc
                        eigen_data[target_key] = {
                            'mode': mode_num,
                            'freq_hz': abs(ev_nm) if is_buckling else cycles,
                            'eigenvalue': abs(ev_nm),
                            'load_factor': ev_nm,
                            'is_buckling': is_buckling,
                            'name': 'F06'
                        }
                        continue
                    if line.strip().startswith('***') or 'DATA BLOCK' in line or 'PAGE' in line:
                        in_table = False
    except Exception:
        return {}
    return eigen_data


def _in_safe_zone(sc, iw):
    """True if screen point is in the safe viewport zone (not over left/right panels)."""
    if sc is None: return False
    x = sc[0]
    return 275 < x < (iw - 258)


def _draw_notation(model, mvp, iw, ih, ortho, state):
    import numpy as np  # must be at top to avoid UnboundLocalError
    from gui.panels import _ELEMENT_FORCE_MAP, _ELEMENT_STRESS_MAP
    any_on = any([state.show_nodes, state.show_node_nums, state.show_elem_nums,
                    state.show_node_values, state.show_elem_values,
                    state.show_station_nodes,
                    state.show_constraints, state.show_reaction_forces,
                    state.show_reaction_moments, state.show_local_axes,
                    state.show_forces, state.show_moments, state.show_pressure])
    if not any_on: return

    dl = imgui.get_foreground_draw_list()
    C_N  = 0xFFAAFFFF   # node dot: yellow
    C_NN = 0xFF88FFFF   # node num: light cyan
    C_EN = 0xFFFFFF66   # elem num: yellow
    C_S  = 0xFFFF44FF   # spc: magenta
    C_SD = 0xFF5566FF   # constraint dof text: pink/red
    C_F  = 0xFF00EEFF   # force: bright yellow
    C_M  = 0xFF00AAFF   # moment: orange
    C_P  = 0xFF44FFAA   # pressure: green

    ms = model.scale()
    arrow_len = ms * 0.12   # screen arrow length in world units
    load_sid = _resolved_load_sid(state)
    load_ref = _active_load_force_scale(state, model)

    dl.push_clip_rect((0.0, 20.0), (float(iw), float(ih)), True)

    # ── Node dots & numbers ─────────────────────────────────────────
    # Build deformed positions for notation
    def _def_xyz(nid):
        node = model.nodes.get(nid)
        if node is None: return None
        xyz = node.xyz.astype(np.float32).copy()
        if (_active_deform_scale(state) > 0 and state.results is not None and
                state.subcase in state.results.displacements):
            d = state.results.displacements[state.subcase].get(nid)
            dv = _result_deform_vec(state.result_type, d) if d is not None else None
            if dv is not None:
                xyz = xyz + dv * _active_deform_scale(state)
        return xyz

    if state.show_nodes or state.show_node_nums:
        for nid, node in model.nodes.items():
            sc = _project(_def_xyz(nid), mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw): continue
            if state.show_nodes:
                dl.add_circle_filled(sc, 3.0, C_N)
                dl.add_circle(sc, 3.0, 0xFF000000, 0, 1.0)
            if state.show_node_nums:
                dl.add_text((sc[0]+4, sc[1]-6), C_NN, str(nid))

    if state.show_station_nodes and state.display_mode in ('beam', 'beam_v2'):
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'diagram_data', lambda: getattr(diag, 'beam_data', {}))() if diag is not None else {}
        shown = set()
        for eid, bd in beam_data.items():
            elem = model.elements.get(eid)
            if elem is None or len(elem.nodes) < 2:
                continue
            n0 = model.nodes.get(elem.nodes[0])
            n1 = model.nodes.get(elem.nodes[1])
            if n0 is None or n1 is None:
                continue
            p0 = _def_xyz(elem.nodes[0])
            p1 = _def_xyz(elem.nodes[1])
            if p0 is None or p1 is None:
                continue
            axis = p1 - p0
            for st in bd.stations:
                key = (eid, round(float(st.sd), 6))
                if key in shown:
                    continue
                xyz = p0 + axis * float(st.sd)
                sc = _project(xyz, mvp, iw, ih, ortho)
                if not _in_safe_zone(sc, iw):
                    continue
                dl.add_circle_filled(sc, 2.5, 0xFFAAFFFF)
                shown.add(key)

    # ── Node result values ──────────────────────────────────────────────
    if state.show_node_values and state.results and state.display_mode == 'contour':
        sc_k = state.subcase
        rt = state.result_type
        node_vals = {}
        if rt in ('displacement','t1','t2','t3') and sc_k in state.results.displacements:
            disp = state.results.displacements[sc_k]
            if rt == 'displacement':
                node_vals = {nid: float(np.linalg.norm(d.translation)) for nid, d in disp.items()}
            else:
                comp = {'t1':0,'t2':1,'t3':2}[rt]
                node_vals = {nid: float(d.translation[comp]) for nid, d in disp.items()}
        elif rt in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin') and sc_k in state.results.stresses:
            nav = _stress_nodal_components(state)
            base = _ELEMENT_STRESS_MAP.get(rt, 'von_mises')
            node_vals = {nid: float(vals.get(base, 0.0)) for nid, vals in nav.items()}
        elif rt in ('nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy') and sc_k in state.results.forces:
            nav = _force_nodal_components(state)
            base = _ELEMENT_FORCE_MAP.get(rt, rt)
            node_vals = {nid: float(vals.get(base, 0.0)) for nid, vals in nav.items()}
        for nid, val in node_vals.items():
            sc = _project(_def_xyz(nid), mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw):
                continue
            dl.add_text((sc[0]+4, sc[1]+6), 0xFFAAFFAA, _format_overlay_value(val))

    # ── Element numbers ─────────────────────────────────────────────
    if state.show_elem_nums:
        pos = {nid: n.xyz for nid,n in model.nodes.items()}
        for eid, elem in model.elements.items():
            pts = [_def_xyz(n) for n in elem.nodes]
            pts = [p for p in pts if p is not None]
            if not pts: continue
            c = np.mean(pts[:4] if elem.type=='CHEXA' else pts[:3] if elem.type in('CPENTA','CTETRA') else pts, axis=0)
            sc = _project(c, mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw): continue
            label = str(eid)
            if elem.type in ('CBAR','CBEAM','CROD') and len(pts) >= 2:
                sc0 = _project(pts[0], mvp, iw, ih, ortho)
                sc1 = _project(pts[1], mvp, iw, ih, ortho)
                if sc0 is not None and sc1 is not None:
                    dx = float(sc1[0] - sc0[0])
                    dy = float(sc1[1] - sc0[1])
                    dn = float(np.hypot(dx, dy))
                    if dn > 1e-6:
                        text_w = len(label) * 7.5
                        px = -dy / dn
                        py = dx / dn
                        off = 18.0
                        tx1 = sc[0] + px * off - text_w * 0.5
                        ty1 = sc[1] + py * off - 6.0
                        tx2 = sc[0] - px * off - text_w * 0.5
                        ty2 = sc[1] - py * off - 6.0
                        def _score(tx, ty):
                            left = tx - 278.0
                            right = (iw - 258.0) - (tx + text_w)
                            top = ty - 24.0
                            bot = (ih - 18.0) - ty
                            return min(left, right, top, bot)
                        tx, ty = (tx1, ty1) if _score(tx1, ty1) >= _score(tx2, ty2) else (tx2, ty2)
                        tx = max(278.0, min(float(iw - 258.0 - text_w), tx))
                        ty = max(24.0, min(float(ih - 18.0), ty))
                        dl.add_text((float(round(tx)), float(round(ty))), C_EN, label)
                        continue
            dl.add_text((sc[0]-len(label)*3.5, sc[1]-5), C_EN, label)

    # ── Element result values ───────────────────────────────────────────
    if state.show_elem_values and state.results and state.display_mode == 'contour':
        sc_k = state.subcase
        rt = state.result_type
        elem_vals = {}
        if rt in ('von_mises','oxx','oyy','txy','omax','omin',
                  'von_mises_top','von_mises_bottom') and sc_k in state.results.stresses:
            st = state.results.stresses[sc_k]
            for eid, es in st.items():
                if (hasattr(es, 'elem_id') and hasattr(es, 'values') and hasattr(es, 'von_mises')
                        and not _is_beam_stress_obj(es)):
                    elem_vals[eid] = es.von_mises if rt == 'von_mises' else float(es.values.get(rt, es.von_mises))
        elif rt in ('fx','fy','fxy','mx','my','mxy','qx','qy') and sc_k in state.results.forces:
            for eid, ef in state.results.forces[sc_k].items():
                if hasattr(ef, 'elem_id') and hasattr(ef, 'values'):
                    elem_vals[eid] = float(ef.values.get(rt, 0.0))
        for eid, val in elem_vals.items():
            elem = model.elements.get(eid)
            if not elem:
                continue
            pts = [_def_xyz(n) for n in elem.nodes]
            pts = [p for p in pts if p is not None]
            if not pts:
                continue
            c = np.mean(pts[:4] if elem.type=='CHEXA' else pts[:3] if elem.type in('CPENTA','CTETRA') else pts, axis=0)
            sc = _project(c, mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw):
                continue
            dl.add_text((sc[0]+4, sc[1]+6), 0xFFFFCC88, _format_overlay_value(val))

    # Beam station values
    if state.show_elem_values and state.display_mode in ('beam', 'beam_v2'):
        diag = getattr(state, 'beam_diagram', None)
        beam_data = getattr(diag, 'diagram_data', lambda: getattr(diag, 'beam_data', {}))() if diag is not None else {}
        result_key = getattr(state, 'beam_result_key', 'bm1')
        scale = float(getattr(state, 'beam_scale', 0.0))
        if beam_data:
            if scale <= 0.0:
                try:
                    scale = auto_beam_scale(beam_data, model, result_key)
                except Exception:
                    scale = 1.0
            for eid, bd in beam_data.items():
                if len(bd.stations) < 1:
                    continue
                elem = model.elements.get(eid)
                if elem is None or len(elem.nodes) < 2:
                    continue
                n0 = model.nodes.get(elem.nodes[0])
                n1 = model.nodes.get(elem.nodes[1])
                if n0 is None or n1 is None:
                    continue
                p0 = n0.xyz.astype(np.float32)
                p1 = n1.xyz.astype(np.float32)
                axis = p1 - p0
                L = float(np.linalg.norm(axis))
                if L < 1e-9:
                    continue
                ex = (axis / L).astype(np.float32)
                vo = getattr(elem, 'v_orient', None)
                if vo is not None and np.linalg.norm(vo) > 1e-9:
                    v = (vo / np.linalg.norm(vo)).astype(np.float32)
                else:
                    v = np.array([0,0,1],dtype=np.float32) if abs(ex[2])<0.9 else np.array([0,1,0],dtype=np.float32)
                ez = v - np.dot(v, ex) * ex
                nez = float(np.linalg.norm(ez))
                if nez < 1e-12:
                    continue
                ey = (ez / nez).astype(np.float32)
                label_indices = []
                nst = len(bd.stations)
                if nst <= 4:
                    label_indices = list(range(nst))
                else:
                    vals = [float(getattr(st, result_key, 0.0)) for st in bd.stations]
                    label_indices = [0, nst - 1, int(np.argmax(np.abs(vals)))]
                    for i in range(1, nst - 1):
                        if ((vals[i] >= vals[i-1] and vals[i] >= vals[i+1]) or
                            (vals[i] <= vals[i-1] and vals[i] <= vals[i+1])):
                            label_indices.append(i)
                    label_indices = sorted(set(label_indices))
                prev_text = None
                for idx in label_indices:
                    st = bd.stations[idx]
                    sval = float(getattr(st, result_key, 0.0))
                    text = _format_overlay_value(sval)
                    if prev_text == text:
                        continue
                    base = p0 + ex * L * float(st.sd)
                    tip = base + ey * sval * scale
                    scb = _project(base, mvp, iw, ih, ortho)
                    sct = _project(tip, mvp, iw, ih, ortho)
                    if scb is None or sct is None or not _in_safe_zone(sct, iw):
                        continue
                    dx = float(sct[0] - scb[0])
                    dy = float(sct[1] - scb[1])
                    dn = float(np.hypot(dx, dy))
                    if dn > 1e-6:
                        ux, uy = dx / dn, dy / dn
                        px, py = -uy, ux
                    else:
                        ux, uy = 0.0, -1.0
                        px, py = 1.0, 0.0
                    tx = sct[0] + ux * 10.0 + px * 3.0
                    ty = sct[1] + uy * 10.0 + py * 3.0 - 6.0
                    dl.add_text((tx, ty), 0xFFFFCC88, text)
                    prev_text = text

    # ── Constraints with DOF direction arrows ───────────────────────
    if state.show_constraints or state.show_reaction_forces or state.show_reaction_moments:
        seen = {}
        reactions = {}
        active_spc_sid = _resolved_spc_sid(state)
        if state.results is not None:
            reactions = getattr(state.results, 'reactions', {}).get(state.subcase, {})
        max_force_reaction = 0.0
        max_moment_reaction = 0.0
        for vals in reactions.values():
            max_force_reaction = max(max_force_reaction,
                                     abs(float(vals.get('tx', 0.0))),
                                     abs(float(vals.get('ty', 0.0))),
                                     abs(float(vals.get('tz', 0.0))))
            max_moment_reaction = max(max_moment_reaction,
                                      abs(float(vals.get('rx', 0.0))),
                                      abs(float(vals.get('ry', 0.0))),
                                      abs(float(vals.get('rz', 0.0))))
        for spc in model.spcs:
            if active_spc_sid and int(spc.id) != active_spc_sid:
                continue
            nid = spc.node_id
            if nid not in seen: seen[nid] = set()
            for ch in str(spc.dofs):
                if ch.isdigit() and 1<=int(ch)<=6:
                    seen[nid].add(int(ch))

        for nid, dofs in seen.items():
            xyz = _def_xyz(nid)
            if xyz is None: continue
            sc = _project(xyz, mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw): continue

            # Triangle pin symbol
            s=8.0
            if state.show_constraints:
                dl.add_triangle_filled((sc[0],sc[1]),(sc[0]-s,sc[1]+s*1.5),(sc[0]+s,sc[1]+s*1.5), C_S)
                dl.add_triangle((sc[0],sc[1]),(sc[0]-s,sc[1]+s*1.5),(sc[0]+s,sc[1]+s*1.5), 0xFF000000, 1.0)

            # Constraint dirs are fixed-size local-axis style arrows.
            spc_scale = ms * 0.04
            dof_vecs = {1:np.array([1,0,0],dtype=np.float32),
                        2:np.array([0,1,0],dtype=np.float32),
                        3:np.array([0,0,1],dtype=np.float32)}
            dof_cols = {
                1:0xFF66FFFF,  # yellow
                2:0xFF44CC44,  # green
                3:0xFFFFFFFF,  # white
            }
            constraint_cols = {
                1:0xFF5566FF,
                2:0xFF44CC44,
                3:0xFFFF8833,
            }
            rx = reactions.get(nid, {})
            reaction_keys = {1:'tx', 2:'ty', 3:'tz', 4:'rx', 5:'ry', 6:'rz'}
            reaction_labels = {1:'TX', 2:'TY', 3:'TZ', 4:'RX', 5:'RY', 6:'RZ'}

            if state.show_constraints:
                for axis_id in (1, 2, 3):
                    cvec = dof_vecs[axis_id]
                    ccol = constraint_cols[axis_id]
                    c_tip_w = xyz + cvec * spc_scale
                    sc_tip_constraint = _project(c_tip_w, mvp, iw, ih, ortho)
                    if sc_tip_constraint is None:
                        continue
                    _draw_arrow_2d(dl, sc, sc_tip_constraint, ccol, 1.6, 4.5)
                    dl.add_text((sc_tip_constraint[0]+2, sc_tip_constraint[1]-5), ccol, str(axis_id))

            for dof in sorted(dofs):
                base_dof = dof if dof<=3 else dof-3
                if base_dof not in dof_vecs: continue
                vec = dof_vecs[base_dof]
                col = dof_cols[base_dof]
                ccol = constraint_cols[base_dof]
                key = reaction_keys[dof]
                val = rx.get(key, None)
                scale_len = spc_scale
                sign = 1.0
                has_reaction_arrow = False
                if val is not None:
                    aval = abs(float(val))
                    sign = -1.0 if float(val) < 0.0 else 1.0
                    vmax = max_force_reaction if dof <= 3 else max_moment_reaction
                    if aval > 1e-9 and vmax > 1e-12:
                        scale_len = spc_scale * max(0.15, min(1.0, aval / vmax))
                        has_reaction_arrow = True
                c_tip_w = xyz + vec * spc_scale
                sc_tip_constraint = _project(c_tip_w, mvp, iw, ih, ortho)
                r_tip_w = xyz + vec * (scale_len * sign)
                sc_tip = _project(r_tip_w, mvp, iw, ih, ortho)
                if sc_tip_constraint is None and sc_tip is None:
                    continue
                ref_tip = sc_tip if sc_tip is not None else sc_tip_constraint
                dx = ref_tip[0] - sc[0]
                dy = ref_tip[1] - sc[1]
                dn = float(np.hypot(dx, dy))
                if dn > 1e-6:
                    ux, uy = dx / dn, dy / dn
                else:
                    ux, uy = 1.0, 0.0
                px, py = -uy, ux
                if dof <= 3:
                    if state.show_reaction_forces and has_reaction_arrow and sc_tip is not None:
                        _draw_arrow_2d(dl, sc, sc_tip, col, 2.0, 5.0)
                    if state.show_reaction_forces:
                        txt = None if val is None else (
                            f"{reaction_labels[dof]}={_format_reaction_value(val, state.reaction_engineering)}")
                    else:
                        txt = None
                    if txt:
                        offset = 9.0 if base_dof == 1 else (-11.0 if base_dof == 3 else -8.0)
                        if not has_reaction_arrow or sc_tip is None:
                            tpos = (sc[0] + px * 6.0,
                                    sc[1] + offset)
                        else:
                            tpos = (sc_tip[0] + ux * 8.0 + px * 4.0,
                                    sc_tip[1] + uy * 8.0 + offset)
                        dl.add_text(tpos, col, txt)
                else:
                    if state.show_reaction_moments and has_reaction_arrow and sc_tip is not None:
                        dl.add_line(sc, sc_tip, col, 1.5)
                        dl.add_circle(sc_tip, 3.5, col, 0, 1.5)
                    if state.show_reaction_moments:
                        txt = None if val is None else (
                            f"{reaction_labels[dof]}={_format_reaction_value(val, state.reaction_engineering)}")
                    else:
                        txt = None
                    if txt:
                        offset = 9.0 if base_dof == 1 else (-11.0 if base_dof == 3 else -8.0)
                        if not has_reaction_arrow or sc_tip is None:
                            tpos = (sc[0] + px * 6.0,
                                    sc[1] + offset)
                        else:
                            tpos = (sc_tip[0] + ux * 8.0 + px * 4.0,
                                    sc_tip[1] + uy * 8.0 + offset)
                        dl.add_text(tpos, col, txt)

            # DOF label (only if partial constraint)
            if state.show_constraints and len(dofs) < 6:
                dof_str = ''.join(str(d) for d in sorted(dofs))
                dl.add_text((sc[0]+s+2, sc[1]+1), C_SD, dof_str)

    # Beam end releases / partial fixity markers
    if state.show_constraints:
        for eid, elem in model.elements.items():
            if elem.type not in ('CBAR', 'CBEAM') or len(elem.nodes) < 2:
                continue
            for end_idx, dof_str in ((0, getattr(elem, 'pa', '')), (1, getattr(elem, 'pb', ''))):
                labels = _beam_release_labels(dof_str)
                if not labels:
                    continue
                nid = elem.nodes[end_idx]
                xyz = _def_xyz(nid)
                if xyz is None:
                    continue
                sc = _project(xyz, mvp, iw, ih, ortho)
                if not _in_safe_zone(sc, iw):
                    continue
                other_nid = elem.nodes[1 - end_idx]
                other_xyz = _def_xyz(other_nid)
                if other_xyz is None:
                    continue
                axis = other_xyz - xyz
                na = float(np.linalg.norm(axis))
                if na > 1e-9:
                    ux, uy = 0.0, 0.0
                    sc_other = _project(other_xyz, mvp, iw, ih, ortho)
                    if sc_other is not None:
                        dx = float(sc_other[0] - sc[0])
                        dy = float(sc_other[1] - sc[1])
                        dn = float(np.hypot(dx, dy))
                        if dn > 1e-9:
                            ux, uy = dx / dn, dy / dn
                    px, py = -uy, ux
                else:
                    px, py = 1.0, 0.0
                side = 16.0 if end_idx == 0 else -16.0
                cx = sc[0] + px * side
                cy = sc[1] + py * side
                dl.add_circle_filled((cx, cy), 9.0, 0xCC2A2030)
                dl.add_circle((cx, cy), 9.0, 0xFFFF77FF, 0, 2.0)
                txt = '/'.join(labels)
                dl.add_text((cx + 12.0, cy - 6.0), 0xFFFFB0FF, txt)

        # Rigid spiders (RBE2 / RBE3)
        for rigid in getattr(model, 'rigids', []):
            ref_xyz = _def_xyz(getattr(rigid, 'ref_grid', 0))
            if ref_xyz is None:
                continue
            sc_ref = _project(ref_xyz, mvp, iw, ih, ortho)
            if not _in_safe_zone(sc_ref, iw):
                continue
            col = 0xFF66CCFF if rigid.type == 'RBE3' else 0xFFFFAA66
            dep_pts = []
            for nid in getattr(rigid, 'dep_grids', []):
                xyz = _def_xyz(nid)
                if xyz is None:
                    continue
                sc_dep = _project(xyz, mvp, iw, ih, ortho)
                if sc_dep is None or not _in_safe_zone(sc_dep, iw):
                    continue
                dep_pts.append((nid, sc_dep))
            if not dep_pts:
                continue
            for _, sc_dep in dep_pts:
                dl.add_line(sc_ref, sc_dep, col, 1.4)
                dl.add_circle_filled(sc_dep, 2.5, col)
            dl.add_circle_filled(sc_ref, 4.0, col)
            dl.add_circle(sc_ref, 7.0, col, 0, 1.8)
            dl.add_text((sc_ref[0] + 8.0, sc_ref[1] - 6.0), col, rigid.type)

    # ── Local element axes ───────────────────────────────────────────
    if state.show_local_axes:
        for eid, elem in model.elements.items():
            dim = 'frame' if elem.type in('CBAR','CBEAM','CROD') else                   'shell' if elem.type in('CQUAD4','CTRIA3') else 'solid'
            if dim=='frame' and not state.local_axis_frame: continue
            if dim=='shell' and not state.local_axis_shell: continue
            if dim=='solid' and not state.local_axis_solid: continue

            axes = _elem_local_frame(elem, model)
            if axes is None: continue
            orig_raw, x1, x2, x3, ax_scale = axes
            # Use deformed centroid
            def_pts = [_def_xyz(n) for n in elem.nodes]
            def_pts = [p for p in def_pts if p is not None]
            orig = np.mean(def_pts, axis=0).astype(np.float32) if def_pts else orig_raw
            sc0 = _project(orig, mvp, iw, ih, ortho)
            if sc0 is None: continue

            for vec, col, lbl in [(x1,0xFF4444FF,'1'),(x2,0xFF44CC44,'2'),(x3,0xFFFF8833,'3')]:
                tip_w = orig + vec * ax_scale
                sc1 = _project(tip_w, mvp, iw, ih, ortho)
                if sc1 is None: continue
                _draw_arrow_2d(dl, sc0, sc1, col, 2.0, 6.0)
                dl.add_text((sc1[0]+2, sc1[1]-5), col, lbl)

    # ── Point forces ─────────────────────────────────────────────────
    if state.show_forces:
        # Group by load case
        for force in model.forces:
            if load_sid and force.sid != load_sid: continue
            f_xyz = _def_xyz(force.node_id)
            if f_xyz is None: continue
            mag = float(force.magnitude)
            d = (force.direction.astype(np.float32) * mag)
            nd = np.linalg.norm(d)
            if nd < 1e-9: continue
            d /= nd
            scale_fac = max(0.25, abs(mag) / load_ref) if load_ref > 1e-12 else 1.0
            tip_w = f_xyz + d * arrow_len * scale_fac
            sc0 = _project(f_xyz, mvp, iw, ih, ortho)
            sc1 = _project(tip_w, mvp, iw, ih, ortho)
            if sc0 and sc1:
                _draw_arrow_2d(dl, sc0, sc1, C_F, 2.0, 7.0)
                dl.add_text((sc1[0]+3, sc1[1]-5), C_F,
                             f'F={_format_load_value(force.magnitude)}')

    # ── Moments ───────────────────────────────────────────────────────
    if state.show_moments:
        for moment in model.moments:
            if load_sid and moment.sid != load_sid: continue
            node = model.nodes.get(moment.node_id)
            if node is None: continue
            sc = _project(node.xyz, mvp, iw, ih, ortho)
            if not _in_safe_zone(sc, iw): continue
            # Draw moment as circle with arrow
            dl.add_circle(sc, 10.0, C_M, 0, 2.0)
            dl.add_text((sc[0]+12, sc[1]-5), C_M, f'M={_format_load_value(moment.magnitude)}')

    # ── Pressure loads (PLOAD4) ──────────────────────────────────────
    if state.show_pressure:
        pos = {nid: n.xyz for nid,n in model.nodes.items()}
        for pl in model.pload2s:
            if load_sid and pl['sid'] != load_sid: continue
            for eid in pl.get('eids', []):
                elem = model.elements.get(eid)
                if elem is None: continue
                pts = [pos[n] for n in elem.nodes if n in pos]
                if not pts: continue
                cen = np.mean(pts, axis=0).astype(np.float32)
                if len(pts) >= 3:
                    nrm = np.cross(pts[1]-pts[0], pts[2]-pts[0])
                    nn = np.linalg.norm(nrm)
                    nrm = (nrm/nn).astype(np.float32) if nn>1e-9 else np.array([0,0,1],dtype=np.float32)
                else:
                    nrm = np.array([0,0,1],dtype=np.float32)
                tip_w = cen + nrm * arrow_len * np.sign(pl['pressure'] or 1)
                sc0 = _project(cen, mvp, iw, ih, ortho)
                sc1 = _project(tip_w, mvp, iw, ih, ortho)
                if sc0 and sc1:
                    _draw_arrow_2d(dl, sc0, sc1, C_P, 1.5, 5.0)
                    dl.add_text((sc1[0]+2, sc1[1]-4), C_P, f'p={_format_load_value(pl["pressure"])}')
        for pl in model.pload4s:
            if load_sid and pl['sid'] != load_sid: continue
            elem = model.elements.get(pl['eid'])
            if elem is None: continue
            pts = [pos[n] for n in elem.nodes if n in pos]
            if not pts: continue
            cen = np.mean(pts, axis=0).astype(np.float32)
            if len(pts) >= 3:
                nrm = np.cross(pts[1]-pts[0], pts[2]-pts[0])
                nn = np.linalg.norm(nrm)
                nrm = (nrm/nn).astype(np.float32) if nn>1e-9 else np.array([0,0,1],dtype=np.float32)
            else:
                nrm = np.array([0,0,1],dtype=np.float32)
            tip_w = cen + nrm * arrow_len * np.sign(pl['pressure'] or 1)
            sc0 = _project(cen, mvp, iw, ih, ortho)
            sc1 = _project(tip_w, mvp, iw, ih, ortho)
            if sc0 and sc1:
                _draw_arrow_2d(dl, sc0, sc1, C_P, 1.5, 5.0)
                dl.add_text((sc1[0]+2, sc1[1]-4), C_P, f'p={_format_load_value(pl["pressure"])}')

        # ── PLOAD1: distributed load on beam elements ──────────────
        for pl in model.pload1s:
            if load_sid and pl['sid'] != load_sid: continue
            elem = model.elements.get(pl['eid'])
            if elem is None or len(elem.nodes) < 2: continue
            n0 = model.nodes.get(elem.nodes[0])
            n1 = model.nodes.get(elem.nodes[1])
            if n0 is None or n1 is None: continue
            p0w = n0.xyz.astype(np.float32)
            p1w = n1.xyz.astype(np.float32)
            axes = _beam_local_axes(elem, p0w, p1w)
            if axes is None:
                continue
            ex, ey, ez, L = axes
            x1 = float(pl.get('x1', 0.0))
            x2 = float(pl.get('x2', x1))
            p1 = float(pl.get('p1', 0.0))
            p2 = float(pl.get('p2', p1))
            ltype = str(pl.get('type', '')).strip().upper()
            scale_mode = str(pl.get('scale', '')).strip().upper()
            use_fraction = scale_mode.startswith('FR')

            def _station_t(x):
                if use_fraction:
                    return float(np.clip(x, 0.0, 1.0))
                return float(np.clip(x / L, 0.0, 1.0))

            t1 = _station_t(x1)
            t2 = _station_t(x2)
            ta, tb = (t1, t2) if t1 <= t2 else (t2, t1)
            pa, pb = (p1, p2) if t1 <= t2 else (p2, p1)
            if abs(tb - ta) <= 1e-6:
                eff_mag = max(abs(pa), abs(pb))
            else:
                eff_mag = 0.5 * (abs(pa) + abs(pb)) * max((tb - ta) * L, 0.0)
            load_scale_fac = max(0.25, eff_mag / load_ref) if load_ref > 1e-12 else 1.0

            gx = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            gy = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            gz = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            force_dirs = {
                'FX': gx, 'FY': gy, 'FZ': gz,
                'FXE': ex, 'FYE': ey, 'FZE': ez,
            }
            moment_axes = {
                'MX': gx, 'MY': gy, 'MZ': gz,
                'MXE': ex, 'MYE': ey, 'MZE': ez,
            }

            def _draw_force_arrow(base_t, mag, direction, scale_fac=1.0):
                pt = p0w + ex * (base_t * L)
                tip = pt + direction * arrow_len * np.sign(mag or 1.0) * scale_fac * load_scale_fac
                sc0 = _project(pt, mvp, iw, ih, ortho)
                sc1 = _project(tip, mvp, iw, ih, ortho)
                if sc0 and sc1:
                    _draw_arrow_2d(dl, sc0, sc1, C_P, 1.5, 6.0)
                return sc0, sc1

            def _draw_moment_marker(base_t, mag, axis_vec, scale_fac=1.0):
                pt = p0w + ex * (base_t * L)
                axis_tip = pt + axis_vec * arrow_len * 0.6 * scale_fac * load_scale_fac
                sc0 = _project(pt, mvp, iw, ih, ortho)
                sca = _project(axis_tip, mvp, iw, ih, ortho)
                if sc0 and sca:
                    _draw_moment_glyph_2d(dl, sc0, sca, C_P, np.sign(mag or 1.0), size=8.0 + 4.0 * scale_fac * load_scale_fac)
                return sc0, sca

            is_point = abs(tb - ta) <= 1e-6
            is_force = ltype in force_dirs
            is_moment = ltype in moment_axes
            type_label = _pload1_type_label(ltype)
            if is_point:
                use_mag = pa if abs(pa) >= abs(pb) else pb
                if is_force:
                    sc0, sc1 = _draw_force_arrow(ta, use_mag, force_dirs[ltype], 1.5)
                    if sc1:
                        txt = f'{type_label}={_format_load_value(use_mag)}'
                        off = 10.0 if use_mag >= 0 else -16.0
                        dl.add_text((sc1[0]+2, sc1[1]+off), C_P, txt)
                elif is_moment:
                    sc0, sc1 = _draw_moment_marker(ta, use_mag, moment_axes[ltype], 1.4)
                    if sc0:
                        dl.add_text((sc0[0]+12.0, sc0[1]-6.0), C_P, f'{type_label}={_format_load_value(use_mag)}')
            else:
                if is_moment:
                    ndiv = 3 if abs(pa - pb) <= 1e-9 else 4
                else:
                    ndiv = max(5, int(round((tb - ta) * 10.0)) + 1)
                prev_top = None
                for i in range(ndiv):
                    u = i / max(ndiv - 1, 1)
                    tt = ta + (tb - ta) * u
                    pm = pa + (pb - pa) * u
                    if is_force:
                        pt = p0w + ex * (tt * L)
                        top = pt + force_dirs[ltype] * arrow_len * np.sign(pm or 1.0) * load_scale_fac
                        scb = _project(pt, mvp, iw, ih, ortho)
                        sct = _project(top, mvp, iw, ih, ortho)
                        if scb and sct:
                            _draw_arrow_2d(dl, scb, sct, C_P, 1.3, 5.0)
                            if prev_top and prev_base:
                                dl.add_line(prev_top, sct, C_P, 1.2)
                                dl.add_line(prev_base, scb, C_P, 1.0)
                            prev_top = sct
                            prev_base = scb
                    elif is_moment:
                        scb, sct = _draw_moment_marker(tt, pm, moment_axes[ltype], 0.9)
                        if scb and sct:
                            if prev_top:
                                dl.add_line(prev_top, sct, _with_alpha_u32(C_P, 110), 1.0)
                            prev_top = sct
                mid_t = 0.5 * (ta + tb)
                mid_p = 0.5 * (pa + pb)
                if is_force:
                    _, sc_mid = _draw_force_arrow(mid_t, mid_p, force_dirs[ltype], 1.0)
                    if sc_mid:
                        if abs(pa - pb) <= 1e-9:
                            txt = f'{type_label}={_format_load_value(pa)}'
                        else:
                            txt = f'{type_label}={_format_load_value(pa)}..{_format_load_value(pb)}'
                        off = 10.0 if mid_p >= 0 else -16.0
                        dl.add_text((sc_mid[0]+2, sc_mid[1]+off), C_P, txt)
                elif is_moment:
                    sc_mid, _ = _draw_moment_marker(mid_t, mid_p, moment_axes[ltype], 1.0)
                    if sc_mid:
                        if abs(pa - pb) <= 1e-9:
                            txt = f'{type_label}={_format_load_value(pa)}'
                        else:
                            txt = f'{type_label}={_format_load_value(pa)}..{_format_load_value(pb)}'
                        dl.add_text((sc_mid[0]+12.0, sc_mid[1]-14.0), C_P, txt)

    # ── Max/Min value pointers (when results active) ──────────────────
    if state.results and state.display_mode == 'contour':
        import numpy as np
        sc_k = state.subcase; r = state.results; rt = state.result_type
        max_pos = min_pos = None

        if rt in ('displacement','t1','t2','t3') and sc_k in r.displacements:
            disp = r.displacements[sc_k]
            if rt == 'displacement':
                items = {nid: float(np.linalg.norm(d.translation)) for nid,d in disp.items()}
            else:
                comp = {'t1':0,'t2':1,'t3':2}[rt]
                items = {nid: float(d.translation[comp]) for nid,d in disp.items()}
            if items:
                max_nid = max(items, key=items.get)
                min_nid = min(items, key=items.get)
                n = model.nodes.get(max_nid); n2 = model.nodes.get(min_nid)
                # Use deformed position if available
                if n:
                    d_max = disp.get(max_nid)
                    ads = _active_deform_scale(state)
                    pos_max = n.xyz + d_max.translation*ads if d_max and ads > 0 else n.xyz
                    max_pos = (_project(pos_max, mvp, iw, ih, ortho), f"MAX N{max_nid}", items[max_nid])
                if n2:
                    d_min = disp.get(min_nid)
                    pos_min = n2.xyz + d_min.translation*ads if d_min and ads > 0 else n2.xyz
                    min_pos = (_project(pos_min, mvp, iw, ih, ortho), f"MIN N{min_nid}", items[min_nid])

        elif rt in ('nodal_vm','noxx','noyy','ntxy','nomax','nomin') and sc_k in r.stresses:
            nav = _stress_nodal_components(state)
            base = _ELEMENT_STRESS_MAP.get(rt, 'von_mises')
            if isinstance(nav, dict) and nav:
                max_nid = max(nav, key=lambda k: nav[k].get(base, 0.0))
                min_nid = min(nav, key=lambda k: nav[k].get(base, 0.0))
                for nid, lbl, sign in [(max_nid,'MAX',1),(min_nid,'MIN',-1)]:
                    xyz = _def_xyz(nid)
                    if xyz is None: continue
                    sc_pt = _project(xyz, mvp, iw, ih, ortho)
                    val = nav[nid].get(base, 0.0)
                    if sign > 0: max_pos = (sc_pt, f"{lbl} N{nid}", val)
                    else:        min_pos = (sc_pt, f"{lbl} N{nid}", val)

        elif rt in ('von_mises','oxx','oyy','txy','omax','omin',
                    'von_mises_top','von_mises_bottom') and sc_k in r.stresses:
            st = r.stresses[sc_k]
            def _gv(es):
                if rt=='von_mises': return es.von_mises
                return es.values.get(rt, es.von_mises)
            elem_st = {k:v for k,v in st.items()
                       if k not in ('_nodal_avg', '_nodal_avg_components', '_nodal_acc',
                                    '_solver_nodal_avg', '_solver_nodal_avg_components',
                                    '_derived_nodal_avg', '_derived_nodal_avg_components',
                                    '_shell_corner_contribs')
                       and hasattr(v,'values') and hasattr(v, 'von_mises')
                       and not _is_beam_stress_obj(v)}
            if elem_st:
                max_eid = max(elem_st, key=lambda k: _gv(elem_st[k]))
                min_eid = min(elem_st, key=lambda k: _gv(elem_st[k]))
                for eid, lbl, sign in [(max_eid,'MAX',1),(min_eid,'MIN',-1)]:
                    elem = model.elements.get(eid)
                    if not elem: continue
                    def_pts = [_def_xyz(n) for n in elem.nodes]
                    def_pts = [p for p in def_pts if p is not None]
                    if not def_pts: continue
                    cen = np.mean(def_pts, axis=0)
                    sc_pt = _project(cen, mvp, iw, ih, ortho)
                    val = _gv(elem_st[eid])
                    if sign > 0: max_pos = (sc_pt, f"{lbl} E{eid}", val)
                    else:        min_pos = (sc_pt, f"{lbl} E{eid}", val)
        elif rt in ('sxc', 'sxd', 'sxe', 'sxf', 'smax', 'smin'):
            diag = getattr(state, 'beam_diagram', None)
            beam_data = getattr(diag, 'beam_data', {}) if diag is not None else {}
            best_max = None
            best_min = None
            for eid, bd in beam_data.items():
                if not getattr(bd, 'stations', None):
                    continue
                vals = [float(getattr(st, rt, 0.0)) for st in bd.stations]
                if not vals:
                    continue
                i_max = max(range(len(vals)), key=lambda i: vals[i])
                i_min = min(range(len(vals)), key=lambda i: vals[i])
                if best_max is None or vals[i_max] > best_max[2]:
                    best_max = (eid, bd.stations[i_max].sd, vals[i_max])
                if best_min is None or vals[i_min] < best_min[2]:
                    best_min = (eid, bd.stations[i_min].sd, vals[i_min])
            for info, lbl, sign in ((best_max, 'MAX', 1), (best_min, 'MIN', -1)):
                if info is None:
                    continue
                eid, sd, val = info
                elem = model.elements.get(eid)
                if elem is None or len(elem.nodes) < 2:
                    continue
                p0 = _def_xyz(elem.nodes[0]); p1 = _def_xyz(elem.nodes[1])
                if p0 is None or p1 is None:
                    continue
                xyz = p0 + (p1 - p0) * float(sd)
                sc_pt = _project(xyz, mvp, iw, ih, ortho)
                if sign > 0:
                    max_pos = (sc_pt, f"{lbl} B{eid}", val)
                else:
                    min_pos = (sc_pt, f"{lbl} B{eid}", val)

        # Force element max/min (element centroid pointers, no node moments)
        elif rt in ('fx','fy','fxy','mx','my','mxy','qx','qy'):
            if hasattr(r,'forces') and sc_k in r.forces:
                fc = {k:v for k,v in r.forces[sc_k].items()
                      if k not in ('_nodal_avg', '_solver_nodal_avg', '_derived_nodal_avg')
                      and hasattr(v,'values')}
                if fc:
                    max_eid = max(fc, key=lambda k: fc[k].values.get(rt,0))
                    min_eid = min(fc, key=lambda k: fc[k].values.get(rt,0))
                    for eid, lbl, sign in [(max_eid,'MAX',1),(min_eid,'MIN',-1)]:
                        elem = model.elements.get(eid)
                        if not elem: continue
                        def_pts = [_def_xyz(n) for n in elem.nodes]
                        def_pts = [p for p in def_pts if p is not None]
                        if not def_pts: continue
                        cen = np.mean(def_pts, axis=0)
                        sc_pt = _project(cen, mvp, iw, ih, ortho)
                        val = fc[eid].values.get(rt, 0)
                        if sign > 0: max_pos = (sc_pt, f"{lbl} E{eid}", val)
                        else:        min_pos = (sc_pt, f"{lbl} E{eid}", val)
        elif rt in ('nfx','nfy','nfxy','nmx','nmy','nmxy','nqx','nqy'):
            if hasattr(r,'forces') and sc_k in r.forces:
                nav = _force_nodal_components(state)
                base = _ELEMENT_FORCE_MAP.get(rt, rt)
                if nav:
                    max_nid = max(nav, key=lambda k: nav[k].get(base, 0.0))
                    min_nid = min(nav, key=lambda k: nav[k].get(base, 0.0))
                    for nid, lbl, sign in [(max_nid,'MAX',1),(min_nid,'MIN',-1)]:
                        xyz = _def_xyz(nid)
                        if xyz is None:
                            continue
                        sc_pt = _project(xyz, mvp, iw, ih, ortho)
                        val = nav[nid].get(base, 0.0)
                        if sign > 0: max_pos = (sc_pt, f"{lbl} N{nid}", val)
                        else:        min_pos = (sc_pt, f"{lbl} N{nid}", val)

        # Draw pointers
        for pt_data, col_fill, col_text in [
                (max_pos, 0xFF0000FF, 0xFF88FFFF),
                (min_pos, 0xFF4444FF, 0xFF88FFFF)]:
            if pt_data is None: continue
            sc_pt, lbl, val = pt_data
            if not _in_safe_zone(sc_pt, iw): continue
            x, y = sc_pt
            # Triangle pointer
            dl.add_triangle_filled((x,y-14),(x-7,y-24),(x+7,y-24), col_fill)
            dl.add_circle_filled((x,y), 4.0, col_fill)
            # Label
            dl.add_text((x+8, y-22), col_text, f"{lbl} = {_format_overlay_value(val)}")

    dl.pop_clip_rect()

# ---------------------------------------------------------------------------
# View presets
# ---------------------------------------------------------------------------
_VIEW_PRESETS = [
    ("+X", 90, 0),  ("-X", 270, 0),
    ("+Y",  0,90),  ("-Y",   0,-90),
    ("+Z",  0, 0),  ("-Z", 180, 0),
    ("ISO",45,25),
]


class MystranViewerApp:
    def __init__(self):
        self.state      = ViewerState()
        self.camera     = Camera()
        self._window    = None
        self._ctx       = None
        self._mesh_rnd  = None
        self._beam_rnd  = None
        self._prev_time = 0.0
        self._last_move    = 0.0
        self._last_rebuild = 0.0
        self._navigating   = False

    def run(self, dat_file='', f06_file=''):
        if dat_file: self.state.dat_path=dat_file; self.state.request_load_dat=True
        if f06_file: self.state.f06_path=f06_file; self.state.request_load_f06=True
        self._init_window()
        self._init_gl()
        self._main_loop()
        self._cleanup()

    def _auto_static_deform_scale(self):
        if self.state.model is None or self.state.results is None:
            return
        if self.state.eigen_data:
            return
        try:
            coords = np.array([n.xyz for n in self.state.model.nodes.values()], dtype=np.float64)
            if len(coords) == 0:
                return
            bbox_max = float((coords.max(axis=0) - coords.min(axis=0)).max())
            if bbox_max <= 1e-12:
                return
            sc = self.state.subcase
            max_u = 0.0
            if sc in getattr(self.state.results, 'displacements', {}):
                try:
                    max_u = float(self.state.results.max_displacement(sc))
                except Exception:
                    disp = self.state.results.displacements.get(sc, {})
                    if disp:
                        max_u = max(float(d.magnitude) for d in disp.values())
            if max_u <= 1e-20:
                return
            target = bbox_max * 0.08
            scale = target / max_u
            if not np.isfinite(scale) or scale <= 0.0:
                return
            self.state.deform_scale = round(float(scale), 4)
            print(f"[AutoScale] Static deform scale: {self.state.deform_scale:.4f} ({target:.4g} target / {max_u:.4g} max |u|)")
        except Exception as err:
            print(f"[AutoScale] Static deform scale skipped: {err}")

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
        imgui.create_context()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        io.set_ini_filename("")
        self._apply_dark_theme()
        imgui.backends.glfw_init_for_opengl(_win_addr(self._window), True)
        imgui.backends.opengl3_init("#version 330")
        self._mesh_rnd = MeshRenderer(self._ctx)
        self.camera.reset()

    def _main_loop(self):
        self._prev_time = glfw.get_time()
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            t  = glfw.get_time()
            dt = max(t - self._prev_time, 1e-6)
            self._prev_time       = t
            self.state.fps        = 1.0/dt
            self.state.frame_time = dt

            self._handle_requests()

            fw, fh = glfw.get_framebuffer_size(self._window)
            iw, ih = glfw.get_window_size(self._window)

            # ── 3D ───────────────────────────────────────────────
            self._ctx.viewport = (0,0,fw,fh)
            self._ctx.scissor  = None
            self._ctx.clear(0.15,0.16,0.20,1.0)
            self._ctx.enable(moderngl.DEPTH_TEST)
            self._ctx.disable(moderngl.BLEND)

            mvp = None
            if self.state.model:
                aspect = iw/ih if ih>0 else 1.0
                mvp = self.camera.mvp_ortho(aspect) if self.state.ortho_mode \
                      else self.camera.mvp(aspect)
                mesh_mode = _mesh_display_mode(self.state.display_mode)
                self._mesh_rnd.draw(
                    mvp,
                    display_mode    = mesh_mode,
                    show_edges      = self.state.show_edges,
                    show_spc        = self.state.show_spc,
                    visible_dims    = self.state.visible_dim_tuple(),
                    show_undeformed = (self.state.show_undeformed and
                                       _active_deform_scale(self.state) > 0),
                )

                # ── Beam diagrams ─────────────────────────────────────
                if self.state.display_mode in ('beam', 'beam_v2') and self._beam_rnd and self._beam_rnd.beam_data:
                    if self.state.request_beam_rebuild:
                        if self.state.display_mode == 'beam_v2':
                            self._beam_rnd.upload_exact(
                                self.state.model,
                                result_key = self.state.beam_result_key,
                                scale      = self.state.beam_scale,
                                alpha      = self.state.beam_alpha,
                                load_sid   = _resolved_load_sid(self.state))
                        else:
                            self._beam_rnd.upload(
                                self.state.model,
                                result_key = self.state.beam_result_key,
                                scale      = self.state.beam_scale,
                                alpha      = self.state.beam_alpha)
                        self.state.request_beam_rebuild = False
                    mvp_b = mvp.astype(np.float32).flatten().tobytes()
                    self._beam_rnd.draw(mvp_b)

            # ── ImGui ─────────────────────────────────────────────
            imgui.backends.glfw_new_frame()
            imgui.backends.opengl3_new_frame()
            imgui.new_frame()

            draw_menu_bar(self.state)
            draw_left_panel(self.state, ih)
            draw_right_panel(self.state, self.camera, iw, ih)
            draw_legend(self.state, iw, ih)
            draw_status_strip(self.state, iw, ih)
            draw_contour_toolbar(self.state, iw, ih)
            draw_beam_diagram_panel(self.state, iw, ih)
            draw_node_info(self.state, iw, ih)
            draw_model_browser_windows(self.state, iw, ih)
            self.state.hover_status = ""
            # Both axis and notation: hidden during navigation, shown when still
            idle = (not self._navigating and
                    (glfw.get_time()-self._last_move) > 0.8 and
                    (glfw.get_time()-self._last_rebuild) > 0.3)
            if self.state.model and idle:
                _draw_axis_imgui(self.camera.view_matrix(), iw, ih)
                if mvp is not None:
                    _draw_notation(self.state.model, mvp, fw, fh,
                                    self.state.ortho_mode, self.state)
                    _draw_beam_hover_info(self.state.model, mvp, fw, fh,
                                          self.state.ortho_mode, self.state)
                    _draw_result_hover_info(self.state.model, mvp, fw, fh,
                                            self.state.ortho_mode, self.state)

            imgui.render()
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

            glfw.swap_buffers(self._window)
            if getattr(self.state,'request_quit',False): break

    def _export_viewport_bounds(self, fw, fh, iw, ih):
        sx = fw / max(iw, 1)
        sy = fh / max(ih, 1)
        x0 = int(round(270 * sx))
        x1 = int(round((iw - 255) * sx))
        y0 = int(round(20 * sy))
        return x0, x1, y0

    def _export_viewport_bounds_window(self, iw, ih):
        return 270, max(271, iw - 255), 20

    def _crop_export_frame(self, frame, fw, fh, iw, ih):
        x0, x1, y0 = self._export_viewport_bounds(fw, fh, iw, ih)
        x0 = max(0, min(frame.shape[1], x0))
        x1 = max(x0 + 1, min(frame.shape[1], x1))
        y0 = max(0, min(frame.shape[0] - 1, y0))
        cropped = frame[y0:, x0:x1, :]
        h, w = cropped.shape[:2]
        if w > 1 and (w % 2):
            cropped = cropped[:, :w-1, :]
            w -= 1
        if h > 1 and (h % 2):
            cropped = cropped[:h-1, :, :]
        return cropped

    def _render_export_frame(self, fw, fh, iw, ih):
        import numpy as np
        self._ctx.viewport = (0, 0, fw, fh)
        self._ctx.scissor  = None
        self._ctx.clear(0.15,0.16,0.20,1.0)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.BLEND)

        mvp = None
        if self.state.model:
            aspect = iw/ih if ih>0 else 1.0
            mvp = self.camera.mvp_ortho(aspect) if self.state.ortho_mode else self.camera.mvp(aspect)
            mesh_mode = _mesh_display_mode(self.state.display_mode)
            self._mesh_rnd.draw(
                mvp,
                display_mode    = mesh_mode,
                show_edges      = self.state.show_edges,
                show_spc        = self.state.show_spc,
                visible_dims    = self.state.visible_dim_tuple(),
                show_undeformed = (self.state.show_undeformed and self.state.deform_scale > 0),
            )

            if self.state.display_mode in ('beam', 'beam_v2') and self._beam_rnd and self._beam_rnd.beam_data:
                mvp_b = mvp.astype(np.float32).flatten().tobytes()
                self._beam_rnd.draw(mvp_b)

        imgui.backends.glfw_new_frame()
        imgui.backends.opengl3_new_frame()
        imgui.new_frame()
        draw_menu_bar(self.state)
        draw_left_panel(self.state, ih)
        draw_right_panel(self.state, self.camera, iw, ih)
        draw_legend(self.state, iw, ih)
        draw_status_strip(self.state, iw, ih)
        draw_contour_toolbar(self.state, iw, ih)
        draw_beam_diagram_panel(self.state, iw, ih)
        draw_node_info(self.state, iw, ih)
        draw_model_browser_windows(self.state, iw, ih)
        self.state.hover_status = ""
        if self.state.model:
            _draw_axis_imgui(self.camera.view_matrix(), iw, ih)
            if mvp is not None:
                _draw_notation(self.state.model, mvp, fw, fh,
                               self.state.ortho_mode, self.state)
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        x0, x1, y0 = self._export_viewport_bounds(fw, fh, iw, ih)
        x0 = max(0, min(fw - 1, x0))
        x1 = max(x0 + 1, min(fw, x1))
        width = x1 - x0
        height = max(1, fh - max(0, min(fh - 1, y0)))

        raw = self._ctx.screen.read(viewport=(x0, 0, width, height), components=3)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)[::-1]
        h, w = frame.shape[:2]
        if w > 1 and (w % 2):
            frame = frame[:, :w-1, :]
            w -= 1
        if h > 1 and (h % 2):
            frame = frame[:h-1, :, :]
        return frame

    def _handle_requests(self):
        if getattr(self.state, 'request_export_png', False):
            self.state.request_export_png = False
            self._export_png()
            return
        if getattr(self.state, 'request_export_mp4', False):
            self.state.request_export_mp4 = False
            self._export_mp4()
            return
        if self.state.request_load_dat and self.state.dat_path:
            self._load_dat(self.state.dat_path); self.state.request_load_dat=False
        if self.state.request_load_f06 and self.state.f06_path:
            self._load_f06(self.state.f06_path); self.state.request_load_f06=False
        if self.state.request_rebuild and self.state.model:
            self._rebuild(); self.state.request_rebuild=False
            self._last_rebuild = glfw.get_time()
        if self.state.request_fit and self.state.model:
            used_beam_fit = False
            if self.state.display_mode in ('beam', 'beam_v2') and self._beam_rnd and self._beam_rnd.beam_data:
                bounds = beam_diagram_bounds(
                    self._beam_rnd.diagram_data(),
                    self.state.model,
                    self.state.beam_result_key,
                    self.state.beam_scale)
                if bounds is not None:
                    mn, mx = bounds
                    self.camera.fit(((mn + mx) * 0.5).astype(np.float32),
                                    float(np.linalg.norm(mx - mn)) * 0.6)
                    used_beam_fit = True
            if not used_beam_fit:
                mn,mx=self.state.model.bbox()
                self.camera.fit(self.state.model.center(), float(np.linalg.norm(mx-mn))*.5)
            self.state.request_fit=False

    def _load_dat(self, path):
        try:
            self.state.model=load_dat(path)
            # Clear previous results when new model loaded
            self.state.results = None
            self.state.subcase = 1
            self.state.eigen_data = {}
            self.state.deform_scale = 0.0
            self.state.load_filter_mode = "subcase"
            self.state.active_load_case = 0
            if self._beam_rnd:
                self._beam_rnd.beam_data = {}
            self.state.beam_loaded_subcase = 0
            self.state.beam_source_path = ""
            if _model_has_beams(self.state.model):
                self.camera.set_up_axis('z')
                self.camera._yaw = 0.0
                self.camera._pitch = -89.0
            print(f"[DAT] {len(self.state.model.nodes)} nodes, {len(self.state.model.elements)} elements")
            self._rebuild(); self.state.request_fit=True
        except Exception as e:
            import traceback; print(f"[DAT] Error: {e}"); traceback.print_exc()

    def _load_f06(self, path):
        ext = path.lower().split('.')[-1]
        if ext == 'op2':
            self._load_op2(path); return
        if ext == 'neu':
            self._load_neu(path); return
        try:
            from parser.f06_parser import load_f06 as _lf
            self.state.results = _lf(path)
            self.state.eigen_data = _parse_f06_eigen_data(path, self.state.results)
            sc = self.state.results.subcases
            print(f"[F06] Subcases: {sc}")
            if sc: self.state.subcase = sc[0]
            self._auto_static_deform_scale()
            self._rebuild()
        except Exception as e:
            import traceback; print(f"[F06] Error: {e}"); traceback.print_exc()

    def _fallback_to_f06_neu(self, op2_path):
        """Try companion F06 or NEU when OP2 has no useful results."""
        import os
        base = op2_path.rsplit('.', 1)[0]
        for ext, loader_fn in [('.F06', '_load_f06_direct'),
                                ('.f06', '_load_f06_direct'),
                                ('.NEU', '_load_neu'),
                                ('.neu', '_load_neu')]:
            candidate = base + ext
            if os.path.exists(candidate):
                print(f"[OP2] Falling back to {candidate}")
                if ext.lower() == '.neu':
                    self._load_neu(candidate)
                else:
                    self._load_f06_direct(candidate)
                return
        print("[OP2] No companion F06/NEU found - no results loaded")

    def _load_f06_direct(self, path):
        """Load F06 directly without routing check."""
        try:
            from parser.f06_parser import load_f06 as _lf
            self.state.results = _lf(path)
            self.state.eigen_data = _parse_f06_eigen_data(path, self.state.results)
            sc = self.state.results.subcases
            print(f"[F06] Subcases: {sc}")
            if sc: self.state.subcase = sc[0]
            self._auto_static_deform_scale()
            self._rebuild()
        except Exception as e:
            import traceback; print(f"[F06] Error: {e}"); traceback.print_exc()

    def _load_neu(self, path):
        try:
            from parser.neu_parser import load_neu
            self.state.results = load_neu(path)
            sc = self.state.results.subcases
            print(f"[NEU] Subcases: {sc}, disp: {len(self.state.results.displacements.get(sc[0] if sc else 1, {}))}")
            if sc: self.state.subcase = sc[0]
            self.state.eigen_data = {}
            self._auto_static_deform_scale()
            self._rebuild()
        except Exception as e:
            import traceback; print(f"[NEU] Error: {e}"); traceback.print_exc()

    def _load_op2(self, path):
        try:
            from pyNastran.op2.op2 import OP2
            from parser.f06_parser import F06Results, DisplacementResult, ElementStress
            import numpy as np
            op2 = OP2(debug=False)
            try:
                op2.read_op2(path, combine=True)
            except Exception as read_err:
                print(f"[OP2] Cannot parse OP2: {read_err}")
                self._fallback_to_f06_neu(path); return

            results = F06Results()

            # ── Subcases from displacements + eigenvectors ─────────
            subcases = sorted(set(
                list(op2.displacements.keys()) +
                list(op2.eigenvectors.keys())))
            if not subcases: subcases = [1]
            results.subcases = subcases
            for sc in subcases:
                results.displacements[sc] = {}
                results.stresses[sc]      = {}
                results.forces[sc]        = {}
                results.reactions[sc]     = {}

            # ── Static displacements ────────────────────────────────
            for isc, res in op2.displacements.items():
                if isc not in results.displacements:
                    results.displacements[isc] = {}; results.stresses[isc] = {}
                    if isc not in results.subcases: results.subcases.append(isc)
                self._read_disp_table(res, isc, results)

            # ── Eigenvectors (modal) ────────────────────────────────
            self.state.eigen_data = {}  # {mode_key: {mode, freq_hz, ...}}
            for isc, res in op2.eigenvectors.items():
                if isc not in results.displacements:
                    results.displacements[isc] = {}; results.stresses[isc] = {}
                    if isc not in results.subcases: results.subcases.append(isc)
                # eigenvectors.data shape: (nmodes, nnodes, 6)
                nids = res.node_gridtype[:, 0]
                for mode_idx in range(res.data.shape[0]):
                    mode_key = isc * 1000 + mode_idx + 1  # unique key per mode
                    if mode_key not in results.displacements:
                        results.displacements[mode_key] = {}
                        results.stresses[mode_key] = {}
                        if mode_key not in results.subcases:
                            results.subcases.append(mode_key)
                    data = res.data[mode_idx]
                    # Normalize eigenvector to reasonable scale
                    # Max translation = 10% of model bounding box
                    t_data = data[:,:3]
                    max_t = float(np.abs(t_data).max())
                    if max_t > 1e-30:
                        model_scale = 1.0  # will normalize in viewer
                        norm_scale = 1.0 / max_t  # normalize to unit max
                    else:
                        norm_scale = 1.0
                    for i, nid in enumerate(nids):
                        results.displacements[mode_key][int(nid)] = DisplacementResult(
                            subcase=mode_key, node_id=int(nid),
                            t1=float(data[i,0])*norm_scale,
                            t2=float(data[i,1])*norm_scale,
                            t3=float(data[i,2])*norm_scale,
                            r1=float(data[i,3])*norm_scale,
                            r2=float(data[i,4])*norm_scale,
                            r3=float(data[i,5])*norm_scale)

            # ── Eigenvalues table ───────────────────────────────────
            eig_sc_list = sorted(op2.eigenvectors.keys()) if getattr(op2, 'eigenvectors', None) else []
            for iev, (ev_name, ev_table) in enumerate(op2.eigenvalues.items()):
                try:
                    isc = eig_sc_list[min(iev, len(eig_sc_list)-1)] if eig_sc_list else (results.subcases[0] if results.subcases else 1)
                    cycles     = list(ev_table.cycles)
                    eigenvalues= list(ev_table.eigenvalues)
                    radians    = list(ev_table.radians)
                    is_buckling = ('BUCK' in str(ev_name).upper() or
                        (eigenvalues and all(v < 0 for v in eigenvalues[:3])))
                    for mode_idx, (freq, omega, ev_nm) in enumerate(
                            zip(cycles, radians, eigenvalues)):
                        mode_num = mode_idx + 1
                        mode_key = isc * 1000 + mode_idx + 1
                        if is_buckling:
                            self.state.eigen_data[mode_key] = {
                                'mode': mode_num,
                                'freq_hz': abs(float(ev_nm)),
                                'eigenvalue': abs(float(ev_nm)),
                                'load_factor': float(ev_nm),
                                'is_buckling': True, 'name': ev_name}
                        else:
                            self.state.eigen_data[mode_key] = {
                                'mode': mode_num, 'freq_hz': float(freq)/(2*3.14159),
                                'cycles': float(freq), 'omega': float(omega),
                                'eigenvalue': float(ev_nm), 'name': ev_name}
                except Exception as e:
                    print(f"[OP2] eigenvalue table: {e}")

            # ── Element stresses ────────────────────────────────────
            stress = op2.op2_results.stress
            strain = op2.op2_results.strain
            for attr in ('cquad4_stress','ctria3_stress','cquadr_stress','ctriar_stress',
                         'cquad8_stress','ctria6_stress'):
                tbl = getattr(stress, attr, None)
                if tbl:
                    etype = attr.split('_')[0].upper()
                    self._read_shell_stress(tbl, etype, results)

            # ── Solid element stresses ─────────────────────────────
            for attr in ('chexa_stress','cpenta_stress','ctetra_stress','cpyram_stress'):
                tbl = getattr(stress, attr, None)
                if tbl:
                    etype = attr.split('_')[0].upper()
                    self._read_solid_stress(tbl, etype, results)

            for attr in ('cquad4_strain','ctria3_strain','cquadr_strain',
                         'cquad8_strain','ctria6_strain'):
                tbl = getattr(strain, attr, None)
                if tbl:
                    etype = attr.split('_')[0].upper()
                    self._read_shell_stress(tbl, etype, results, is_strain=True)

            # ── Element forces ──────────────────────────────────────
            force = op2.op2_results.force
            for attr in ('cquad4_force','ctria3_force','cquadr_force','ctriar_force',
                         'cquad8_force','ctria6_force'):
                tbl = getattr(force, attr, None)
                if tbl:
                    etype = attr.split('_')[0].upper()
                    self._read_shell_forces(tbl, etype, results)

            # ── SPC reaction forces ─────────────────────────────────
            spc_forces = getattr(op2, 'spc_forces', None)
            if spc_forces:
                self._read_spc_forces(spc_forces, results)
            else:
                print("[OP2] No SPCFORCES table found; reaction display unavailable for this file")

            # ── Nodal averaged stress ───────────────────────────────
            if self.state.model:
                self._compute_nodal_avg(results, self.state.model)

            self.state.results = results
            # ── Summary ────────────────────────────────────────────
            has_disp = any(len(v)>0 for v in results.displacements.values())
            if not has_disp:
                self._fallback_to_f06_neu(path); return

            sc = results.subcases
            # Auto deform scale:
            # - eigen/buckling: 10% of bbox with normalized eigenvectors
            # - static: target a visible fraction of bbox from actual max displacement
            if self.state.eigen_data and self.state.model:
                coords = np.array([n.xyz for n in self.state.model.nodes.values()])
                if len(coords):
                    bbox_max = float((coords.max(axis=0)-coords.min(axis=0)).max())
                    # eigenvectors normalized to max=1, so bbox*0.1 = 10% deform
                    self.state.deform_scale = round(bbox_max * 0.1, 4)
                    print(f"[OP2] Auto deform scale: {self.state.deform_scale:.4f} (10% of {bbox_max:.4f}m bbox)")
            else:
                self._auto_static_deform_scale()
            print(f"[OP2] Subcases/modes: {sc}")
            for s in sc[:8]:
                nd = len(results.displacements.get(s,{}))
                ns = len(results.stresses.get(s,{}))
                info = ''
                if s in self.state.eigen_data:
                    ed = self.state.eigen_data[s]
                    info = " " + _eigen_result_summary(ed)
                print(f"  SC{s}: {nd} disp, {ns} stresses{info}")

            if sc: self.state.subcase = sc[0]
            self._rebuild()
            self._load_beam_diagrams(path, sc[0] if sc else 1)
        except ImportError:
            print("[OP2] pyNastran not installed: pip install pyNastran")
        except Exception as e:
            import traceback; print(f"[OP2] Error: {e}"); traceback.print_exc()

    def _read_disp_table(self, res, isc, results):
        from parser.f06_parser import DisplacementResult
        import numpy as np
        nids = res.node_gridtype[:, 0]
        data = res.data[-1]
        for i, nid in enumerate(nids):
            results.displacements[isc][int(nid)] = DisplacementResult(
                subcase=isc, node_id=int(nid),
                t1=float(data[i,0]), t2=float(data[i,1]), t3=float(data[i,2]),
                r1=float(data[i,3]), r2=float(data[i,4]), r3=float(data[i,5]))

    def _read_spc_forces(self, tbl, results):
        for isc, res in tbl.items():
            if isc not in results.reactions:
                results.reactions[isc] = {}
            node_ids = res.node_gridtype[:, 0]
            data = res.data[-1]
            for i, nid in enumerate(node_ids):
                results.reactions[isc][int(nid)] = {
                    'tx': float(data[i, 0]),
                    'ty': float(data[i, 1]),
                    'tz': float(data[i, 2]),
                    'rx': float(data[i, 3]),
                    'ry': float(data[i, 4]),
                    'rz': float(data[i, 5]),
                }
            if len(node_ids):
                print(f"  [SPCForce] SC{isc}: {len(node_ids)} nodes")

    def _read_solid_stress(self, tbl, etype, results):
        """Read solid element stress.
        - Element stress: centroid row (nid=0)
        - Nodal stress: average of corner values per node (more accurate than simple avg)
        """
        from parser.f06_parser import ElementStress
        import numpy as np
        stress_comps = ('von_mises', 'oxx', 'oyy', 'txy', 'omax', 'omin')
        for isc, res in tbl.items():
            if isc not in results.stresses: results.stresses[isc] = {}
            eids = res.element_node[:,0]
            nids = res.element_node[:,1]
            data = res.data[-1]
            # headers: oxx oyy ozz txy tyz txz omax omid omin von_mises
            elem_stress = {}   # eid -> ElementStress (centroid)
            node_acc = {}

            for i in range(len(eids)):
                eid = int(eids[i]); nid = int(nids[i])
                row = data[i]
                vm  = abs(float(row[-1]))
                comp_vals = {
                    'von_mises': vm,
                    'oxx': float(row[0]),
                    'oyy': float(row[1]),
                    'txy': float(row[3]),
                    'omax': float(row[6]),
                    'omin': float(row[8]),
                }
                if nid == 0:
                    # Centroid -> element stress
                    elem_stress[eid] = ElementStress(
                        subcase=isc, elem_id=eid, elem_type=etype,
                        values=comp_vals,
                        von_mises=vm)
                else:
                    # Corner node -> accumulate for nodal avg
                    if nid not in node_acc:
                        node_acc[nid] = {c: [] for c in stress_comps}
                    for comp, val in comp_vals.items():
                        node_acc[nid][comp].append(val)

            results.stresses[isc].update(elem_stress)

            # Store nodal avg from corner values (these are actual nodal values!)
            if node_acc:
                nodal_comp = {
                    nid: {comp: float(np.mean(vals)) for comp, vals in comp_map.items()}
                    for nid, comp_map in node_acc.items()
                }
                existing_comp = results.stresses[isc].get('_solver_nodal_avg_components', {})
                existing_comp.update(nodal_comp)
                results.stresses[isc]['_solver_nodal_avg_components'] = existing_comp
                results.stresses[isc]['_solver_nodal_avg'] = {
                    nid: vals['von_mises'] for nid, vals in existing_comp.items()
                }
                nav_vals = [vals['von_mises'] for vals in nodal_comp.values()]
                print(f"  [Stress] {etype} SC{isc}: {len(elem_stress)} elems (centroid), "
                      f"{len(nodal_comp)} nodes (corner avg), "
                      f"vm=[{min(nav_vals):.3e}, {max(nav_vals):.3e}]")
            elif elem_stress:
                vms = [v.von_mises for v in elem_stress.values()]
                print(f"  [Stress] {etype} SC{isc}: {len(elem_stress)} elems, "
                      f"vm=[{min(vms):.3e}, {max(vms):.3e}]")

    def _read_shell_stress(self, tbl, etype, results, is_strain=False):
        """Read shell stress/strain and preserve raw corner contributions."""
        from parser.f06_parser import ElementStress
        import numpy as np
        stress_comps = ('von_mises', 'oxx', 'oyy', 'txy', 'omax', 'omin')
        for isc, res in tbl.items():
            if isc not in results.stresses: results.stresses[isc] = {}
            elem_node = res.element_node
            eids = elem_node[:,0]
            nids = elem_node[:,1]
            data = res.data[-1]
            # Columns: fiber_dist oxx oyy txy angle omax omin von_mises
            seen = {}
            nodal_seen = {}
            corner_contribs = {}
            for i, eid in enumerate(eids):
                eid = int(eid)
                nid = int(nids[i])
                fiber_dist = float(data[i,0])
                vm  = abs(float(data[i,-1]))  # von_mises last col
                oxx = float(data[i,1]); oyy=float(data[i,2]); txy=float(data[i,3])
                omax = float(data[i,5]); omin = float(data[i,6])
                store = seen if nid == 0 else nodal_seen
                key = eid if nid == 0 else (eid, nid)
                if key not in store:
                    store[key] = {'fd':[],'vm':[],'oxx':[],'oyy':[],'txy':[],'omax':[],'omin':[]}
                store[key]['fd'].append(fiber_dist)
                store[key]['vm'].append(vm)
                store[key]['oxx'].append(oxx); store[key]['oyy'].append(oyy)
                store[key]['txy'].append(txy); store[key]['omax'].append(omax)
                store[key]['omin'].append(omin)
                if nid != 0:
                    if nid not in corner_contribs:
                        corner_contribs[nid] = []
                    corner_contribs[nid].append({
                        'eid': eid,
                        'elem_type': etype,
                        'fiber': 'top' if fiber_dist >= 0.0 else 'bottom',
                        'fiber_dist': fiber_dist,
                        'oxx': oxx,
                        'oyy': oyy,
                        'txy': txy,
                        'omax': omax,
                        'omin': omin,
                        'von_mises': vm,
                    })
            for eid, vals in seen.items():
                i_top = int(np.argmax(vals['fd'])) if vals['fd'] else 0
                i_bot = int(np.argmin(vals['fd'])) if vals['fd'] else 0
                vm_max = float(np.max(vals['vm']))
                results.stresses[isc][eid] = ElementStress(
                    subcase=isc, elem_id=eid, elem_type=etype,
                    values={'von_mises': vm_max,
                            'von_mises_top': float(vals['vm'][i_top]),
                            'von_mises_bottom': float(vals['vm'][i_bot]),
                            'oxx':  float(np.mean(vals['oxx'])),
                            'oyy':  float(np.mean(vals['oyy'])),
                            'txy':  float(np.mean(vals['txy'])),
                            'omax': float(np.max(vals.get('omax',[0]))),
                            'omin': float(np.min(vals.get('omin',[0])))},
                    von_mises=vm_max)
            if nodal_seen:
                nodal_comp = {}
                for (eid, nid), vals in nodal_seen.items():
                    comp_vals = {
                        'von_mises': float(np.max(vals['vm'])),
                        'oxx': float(np.mean(vals['oxx'])),
                        'oyy': float(np.mean(vals['oyy'])),
                        'txy': float(np.mean(vals['txy'])),
                        'omax': float(np.max(vals['omax'])),
                        'omin': float(np.min(vals['omin'])),
                    }
                    if nid not in nodal_comp:
                        nodal_comp[nid] = {c: [] for c in stress_comps}
                    for comp, val in comp_vals.items():
                        nodal_comp[nid][comp].append(val)
                merged = {
                    nid: {comp: float(np.mean(vals)) for comp, vals in comp_map.items()}
                    for nid, comp_map in nodal_comp.items()
                }
                existing_comp = results.stresses[isc].get('_solver_nodal_avg_components', {})
                existing_comp.update(merged)
                results.stresses[isc]['_solver_nodal_avg_components'] = existing_comp
                results.stresses[isc]['_solver_nodal_avg'] = {
                    nid: vals['von_mises'] for nid, vals in existing_comp.items()
                }
            if corner_contribs:
                existing_corner = results.stresses[isc].get('_shell_corner_contribs', {})
                for nid, vals in corner_contribs.items():
                    existing_corner.setdefault(nid, []).extend(vals)
                results.stresses[isc]['_shell_corner_contribs'] = existing_corner
            if seen:
                sample = list(seen.values())[0]
                avail_keys = [k for k,v in sample.items() if v]
                print(f"  [Stress] {etype} SC{isc}: {len(seen)} elems, "
                      f"keys={avail_keys}, "
                      f"vm_range=[{min(v['vm'][0] for v in seen.values()):.3e}, "
                      f"{max(v['vm'][0] for v in seen.values()):.3e}]")

    def _read_shell_forces(self, tbl, etype, results):
        """Read OP2 shell element forces: element -> (fx,fy,fxy,mx,my,mxy,qx,qy)"""
        from parser.f06_parser import ElementForce
        import numpy as np
        force_comps = ('fx','fy','fxy','mx','my','mxy','qx','qy')
        for isc, res in tbl.items():
            if isc not in results.forces: results.forces[isc] = {}
            data = res.data[-1]
            element_node = getattr(res, 'element_node', None)
            nodal_acc = {}
            elem_count = 0
            if element_node is not None:
                for i, row in enumerate(data):
                    eid = int(element_node[i, 0])
                    nid = int(element_node[i, 1])
                    comp_vals = {
                        'fx': float(row[0]), 'fy': float(row[1]), 'fxy': float(row[2]),
                        'mx': float(row[3]), 'my': float(row[4]), 'mxy': float(row[5]),
                        'qx': float(row[6]), 'qy': float(row[7]),
                    }
                    if nid == 0:
                        results.forces[isc][eid] = ElementForce(
                            subcase=isc, elem_id=eid, elem_type=etype, **comp_vals)
                        elem_count += 1
                    else:
                        if nid not in nodal_acc:
                            nodal_acc[nid] = {c: [] for c in force_comps}
                        for comp, val in comp_vals.items():
                            nodal_acc[nid][comp].append(val)
            else:
                eids = res.element
                for i, eid in enumerate(eids):
                    eid = int(eid); row = data[i]
                    results.forces[isc][eid] = ElementForce(
                        subcase=isc, elem_id=eid, elem_type=etype,
                        fx=float(row[0]), fy=float(row[1]), fxy=float(row[2]),
                        mx=float(row[3]), my=float(row[4]), mxy=float(row[5]),
                        qx=float(row[6]), qy=float(row[7]))
                elem_count = len(eids)
            if nodal_acc:
                merged = {
                    nid: {comp: float(np.mean(vals)) for comp, vals in comp_map.items()}
                    for nid, comp_map in nodal_acc.items()
                }
                existing = results.forces[isc].get('_solver_nodal_avg', {})
                existing.update(merged)
                results.forces[isc]['_solver_nodal_avg'] = existing
            if elem_count:
                print(f"  [Force] {etype} SC{isc}: {elem_count} elems, "
                      f"max|FY|={max(abs(data[:,1])):.3e} max|MX|={max(abs(data[:,3])):.3e}")

    def _compute_nodal_avg(self, results, model):
        """Compute conservative derived nodal stress and forces."""
        import numpy as np
        elem_nodes = {eid: elem.nodes for eid,elem in model.elements.items()}
        stress_comps = ('von_mises', 'oxx', 'oyy', 'txy', 'omax', 'omin')
        avg_scope = getattr(self.state, 'averaging_scope', 'property')
        same_family = getattr(self.state, 'average_same_family', True)
        angle_limit_deg = float(getattr(self.state, 'average_angle_deg', 20.0))

        def _elem_family(elem_type):
            if elem_type in ('CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6', 'CQUADR', 'CTRIAR'):
                return 'shell'
            if elem_type in ('CHEXA', 'CPENTA', 'CTETRA', 'CPYRAM'):
                return 'solid'
            if elem_type in ('CBAR', 'CBEAM', 'CROD'):
                return 'frame'
            return elem_type

        def _compat_signature(eid, elem_type):
            elem = model.elements.get(eid)
            if elem is None:
                return None
            prop = model.properties.get(elem.pid)
            return {
                'family': _elem_family(elem_type),
                'pid': elem.pid,
                'mid': prop.mid if prop else 0,
            }

        def _compatible(sig, seed):
            if sig is None:
                return False
            if seed is None:
                return True
            if same_family and sig['family'] != seed['family']:
                return False
            if avg_scope == 'property' and sig['pid'] != seed['pid']:
                return False
            if avg_scope == 'material' and sig['mid'] != seed['mid']:
                return False
            return True

        def _shell_normal(eid):
            elem = model.elements.get(eid)
            if elem is None or elem.type not in ('CQUAD4', 'CTRIA3', 'CQUAD8', 'CTRIA6', 'CQUADR', 'CTRIAR'):
                return None
            pts = [model.nodes[n].xyz for n in elem.nodes if n in model.nodes]
            if len(pts) < 3:
                return None
            v1 = np.asarray(pts[1], dtype=np.float64) - np.asarray(pts[0], dtype=np.float64)
            v2 = np.asarray(pts[-1], dtype=np.float64) - np.asarray(pts[0], dtype=np.float64)
            n = np.cross(v1, v2)
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                return None
            return n / nn

        def _compatible_angle(normal, seed_normal):
            if normal is None or seed_normal is None:
                return True
            dot = float(np.clip(abs(np.dot(normal, seed_normal)), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(dot)))
            return ang <= angle_limit_deg

        # ── Stress nodal avg ────────────────────────────────────────
        for isc, elem_stress in results.stresses.items():
            if not elem_stress: continue
            corner_contribs = elem_stress.get('_shell_corner_contribs', {})
            computed_comp = {}
            if corner_contribs:
                for nid, contribs in corner_contribs.items():
                    fiber_buckets = {'top': [], 'bottom': []}
                    seed_sig = {'top': None, 'bottom': None}
                    seed_normal = {'top': None, 'bottom': None}
                    for c in contribs:
                        sig = _compat_signature(c['eid'], c.get('elem_type', ''))
                        fiber = c.get('fiber', 'top')
                        normal = _shell_normal(c['eid'])
                        if not _compatible(sig, seed_sig[fiber]):
                            continue
                        if not _compatible_angle(normal, seed_normal[fiber]):
                            continue
                        if seed_sig[fiber] is None:
                            seed_sig[fiber] = sig
                            seed_normal[fiber] = normal
                        fiber_buckets[fiber].append(c)
                    fiber_results = {}
                    for fiber, vals in fiber_buckets.items():
                        if not vals:
                            continue
                        oxx = float(np.mean([v['oxx'] for v in vals]))
                        oyy = float(np.mean([v['oyy'] for v in vals]))
                        txy = float(np.mean([v['txy'] for v in vals]))
                        fiber_results[fiber] = {
                            'oxx': oxx,
                            'oyy': oyy,
                            'txy': txy,
                            'omax': float(np.mean([v['omax'] for v in vals])),
                            'omin': float(np.mean([v['omin'] for v in vals])),
                            'von_mises': float(np.sqrt(max(0.0, oxx*oxx - oxx*oyy + oyy*oyy + 3.0*txy*txy))),
                        }
                    if fiber_results:
                        best_fiber = max(fiber_results, key=lambda k: fiber_results[k]['von_mises'])
                        computed_comp[nid] = fiber_results[best_fiber]
            else:
                if not elem_stress.get('_solver_nodal_avg_components', {}):
                    continue
                node_comp = {}
                node_seed = {}
                for eid, es in elem_stress.items():
                    if not hasattr(es,'von_mises') or _is_beam_stress_obj(es):
                        continue
                    sig = _compat_signature(eid, getattr(es, 'elem_type', ''))
                    for nid in elem_nodes.get(eid, []):
                        seed = node_seed.get(nid)
                        if not _compatible(sig, seed):
                            continue
                        if seed is None:
                            node_seed[nid] = sig
                        if nid not in node_comp:
                            node_comp[nid] = {c: [] for c in stress_comps}
                        node_comp[nid]['oxx'].append(float(es.values.get('oxx', 0.0)))
                        node_comp[nid]['oyy'].append(float(es.values.get('oyy', 0.0)))
                        node_comp[nid]['txy'].append(float(es.values.get('txy', 0.0)))
                        node_comp[nid]['omax'].append(float(es.values.get('omax', es.von_mises)))
                        node_comp[nid]['omin'].append(float(es.values.get('omin', es.von_mises)))
                for nid, comp_map in node_comp.items():
                    oxx = float(np.mean(comp_map['oxx'])) if comp_map['oxx'] else 0.0
                    oyy = float(np.mean(comp_map['oyy'])) if comp_map['oyy'] else 0.0
                    txy = float(np.mean(comp_map['txy'])) if comp_map['txy'] else 0.0
                    computed_comp[nid] = {
                        'oxx': oxx,
                        'oyy': oyy,
                        'txy': txy,
                        'omax': float(np.mean(comp_map['omax'])) if comp_map['omax'] else 0.0,
                        'omin': float(np.mean(comp_map['omin'])) if comp_map['omin'] else 0.0,
                        'von_mises': float(np.sqrt(max(0.0, oxx*oxx - oxx*oyy + oyy*oyy + 3.0*txy*txy))),
                    }
            existing_comp = results.stresses[isc].get('_derived_nodal_avg_components', {})
            for nid, vals in computed_comp.items():
                existing_comp.setdefault(nid, vals)
            results.stresses[isc]['_derived_nodal_avg_components'] = existing_comp
            nodal_avg = {nid: vals['von_mises'] for nid, vals in existing_comp.items()}
            results.stresses[isc]['_derived_nodal_avg'] = nodal_avg
            if existing_comp:
                avg_vals = [vals['von_mises'] for vals in existing_comp.values()]
                print(f"  [Derived Nodal Stress] SC{isc}: {len(nodal_avg)} nodes, "
                      f"vm=[{min(avg_vals):.3e}, {max(avg_vals):.3e}]")

        # ── Force nodal avg ─────────────────────────────────────────
        force_comps = ('fx','fy','fxy','mx','my','mxy','qx','qy')
        for isc, elem_forces in results.forces.items():
            if not elem_forces: continue
            node_comp = {}
            node_seed = {}
            for eid, ef in elem_forces.items():
                if not hasattr(ef, 'values'):
                    continue
                sig = _compat_signature(eid, getattr(ef, 'elem_type', ''))
                for nid in elem_nodes.get(eid, []):
                    seed = node_seed.get(nid)
                    if not _compatible(sig, seed):
                        continue
                    if seed is None:
                        node_seed[nid] = sig
                    if nid not in node_comp:
                        node_comp[nid] = {c:[] for c in force_comps}
                    for c in force_comps:
                        node_comp[nid][c].append(ef.values.get(c, 0.0))
            nodal_force_avg = {
                nid: {c: float(np.mean(vals)) for c, vals in comp.items()}
                for nid, comp in node_comp.items()
            }
            existing = results.forces[isc].get('_derived_nodal_avg', {})
            for nid, vals in nodal_force_avg.items():
                existing.setdefault(nid, vals)
            results.forces[isc]['_derived_nodal_avg'] = existing
            if existing:
                mx_vals = [v['mx'] for v in existing.values()]
                print(f"  [Derived Nodal Force] SC{isc}: {len(nodal_force_avg)} nodes, "
                      f"MX=[{min(mx_vals):.3e}, {max(mx_vals):.3e}]")


    def _load_beam_diagrams(self, op2_path, subcase=1):
        try:
            if self._beam_rnd is None:
                self._beam_rnd = BeamDiagramRenderer(
                    self._ctx,
                    self._mesh_rnd._prog_line,
                    self._mesh_rnd._prog_fill)
                self.state.beam_diagram = self._beam_rnd
            ok = self._beam_rnd.load_op2(op2_path, subcase)
            if ok:
                self.state.show_beam_panel = True
                self.state.beam_source_path = op2_path
                self.state.beam_loaded_subcase = subcase
                self.state.request_beam_rebuild = True
                if self.state.display_mode in ('beam', 'beam_v2'):
                    self.state.request_fit = True
                print(f"[Beam] {len(self._beam_rnd.beam_data)} beams with diagram data")
        except Exception as e:
            import traceback; print(f"[Beam] {e}"); traceback.print_exc()

    def _rebuild(self):
        if not self.state.model: return
        try:
            if (self.state.beam_source_path and self.state.display_mode in ('beam', 'beam_v2') and
                    self.state.beam_loaded_subcase != self.state.subcase):
                self._load_beam_diagrams(self.state.beam_source_path, self.state.subcase)
            self._mesh_rnd.upload(
                self.state.model,
                results      = self.state.results,
                subcase      = self.state.subcase,
                display_mode = _mesh_display_mode(self.state.display_mode),
                result_type  = self.state.result_type,
                cmap_name    = self.state.cmap_name,
                deform_scale = _active_deform_scale(self.state),
                nodal_result_source = self.state.nodal_result_source,
                active_spc_sid = _resolved_spc_sid(self.state),
                beam_end_data = self._beam_rnd.beam_data if self._beam_rnd else None,
            )
            self._mesh_rnd.upload_undeformed(self.state.model)
        except Exception as e:
            import traceback; print(f"[Renderer] {e}"); traceback.print_exc()

    def _export_mp4(self):
        """Animate deform_scale sinusoidally, export to MP4 (no notation)."""
        import numpy as np, os, datetime
        try:
            import imageio
        except ImportError:
            print("[Export] pip install imageio[ffmpeg]"); return

        fps = 30; n_frames = 60
        base_scale = self.state.deform_scale if self.state.deform_scale > 0 else 0.05
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sc  = self.state.subcase
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"anim_sc{sc}_{ts}.mp4")
        print(f"[Export] {n_frames} frames -> {out}")
        self.state.exporting_mp4 = True

        # Save + disable notation
        orig_scale = self.state.deform_scale
        notation_flags = ['show_nodes','show_node_nums','show_elem_nums',
                          'show_constraints','show_reaction_forces',
                          'show_reaction_moments','show_local_axes',
                          'show_forces','show_moments','show_pressure']
        orig_notation = {k: getattr(self.state, k) for k in notation_flags}
        for k in notation_flags: setattr(self.state, k, False)

        fw, fh = glfw.get_framebuffer_size(self._window)
        iw, ih = glfw.get_window_size(self._window)

        try:
            writer = imageio.get_writer(out, fps=fps, codec='libx264', quality=8,
                                        macro_block_size=None)
            for fi in range(n_frames):
                phase = 2 * np.pi * fi / n_frames
                self.state.deform_scale = base_scale * float(np.sin(phase))
                self.state.export_progress = fi / n_frames * 100

                # Render current viewport-style frame
                self._mesh_rnd.upload(
                    self.state.model,
                    results=self.state.results, subcase=self.state.subcase,
                    display_mode=_mesh_display_mode(self.state.display_mode),
                    result_type=self.state.result_type,
                    cmap_name=self.state.cmap_name,
                    deform_scale=self.state.deform_scale,
                    nodal_result_source=self.state.nodal_result_source,
                    active_spc_sid=_resolved_spc_sid(self.state),
                    beam_end_data=self._beam_rnd.beam_data if self._beam_rnd else None)
                frame = self._render_export_frame(fw, fh, iw, ih)
                writer.append_data(frame)

            writer.close()
            print(f"[Export] Saved: {out}")
        except Exception as e:
            import traceback; print(f"[Export] Error: {e}"); traceback.print_exc()
        finally:
            self.state.deform_scale    = orig_scale
            self.state.exporting_mp4   = False
            self.state.export_progress = 0
            for k, v in orig_notation.items(): setattr(self.state, k, v)
            self._rebuild()

    def _export_png(self):
        """Export current view as a single PNG snapshot."""
        import numpy as np, os, datetime
        try:
            import imageio
        except ImportError:
            print("[Export] pip install imageio"); return

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sc = self.state.subcase
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"snapshot_sc{sc}_{ts}.png")
        fw, fh = glfw.get_framebuffer_size(self._window)
        iw, ih = glfw.get_window_size(self._window)

        try:
            self._mesh_rnd.upload(
                self.state.model,
                results=self.state.results, subcase=self.state.subcase,
                display_mode=_mesh_display_mode(self.state.display_mode),
                result_type=self.state.result_type,
                cmap_name=self.state.cmap_name,
                deform_scale=self.state.deform_scale,
                nodal_result_source=self.state.nodal_result_source,
                active_spc_sid=_resolved_spc_sid(self.state),
                beam_end_data=self._beam_rnd.beam_data if self._beam_rnd else None)
            frame = self._render_export_frame(fw, fh, iw, ih)
            imageio.imwrite(out, frame)
            print(f"[Export] Saved PNG: {out}")
        except Exception as e:
            import traceback; print(f"[Export] PNG error: {e}"); traceback.print_exc()

    def _cleanup(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()
        glfw.terminate()

    def _on_mouse_button(self, win, btn, act, mod):
        if not imgui.get_io().want_capture_mouse:
            x,y=glfw.get_cursor_pos(win)
            self.camera.on_mouse_button(btn, 1 if act==glfw.PRESS else 0, x, y)
            if act==glfw.PRESS: self._navigating=True; self._last_move=glfw.get_time()
            else:               self._navigating=False; self._last_move=glfw.get_time()

    def _on_mouse_move(self, win, x, y):
        self.state.mouse_x = float(x)
        self.state.mouse_y = float(y)
        if not imgui.get_io().want_capture_mouse:
            self.camera.on_mouse_move(x,y)
            if self._navigating: self._last_move=glfw.get_time()

    def _on_scroll(self, win, xoff, yoff):
        if not imgui.get_io().want_capture_mouse:
            self.camera.on_scroll(yoff); self._last_move=glfw.get_time()

    def _on_key(self, win, key, sc, act, mod):
        if act!=glfw.PRESS or imgui.get_io().want_capture_keyboard: return
        if   key==glfw.KEY_F:      self.state.request_fit=True
        elif key==glfw.KEY_R:      self.camera.reset()
        elif key==glfw.KEY_W:      self.state.display_mode='wireframe'; self.state.request_rebuild=True
        elif key==glfw.KEY_S:      self.state.display_mode='solid';     self.state.request_rebuild=True
        elif key==glfw.KEY_C:      self.state.display_mode='contour';   self.state.request_rebuild=True
        elif key==glfw.KEY_B:
            if self._beam_rnd and self._beam_rnd.beam_data:
                self.state.display_mode='beam'; self.state.request_rebuild=True; self.state.request_fit=True
        elif key==glfw.KEY_O:      self.state.ortho_mode=not self.state.ortho_mode
        elif key==glfw.KEY_ESCAPE: glfw.set_window_should_close(win,True)

    def _apply_dark_theme(self):
        style=imgui.get_style()
        style.window_rounding=2.0; style.frame_rounding=2.0
        style.grab_rounding=2.0;   style.window_border_size=0.0
        def sc(col,r,g,b,a=1.0): style.set_color_(col,imgui.ImVec4(r,g,b,a))
        sc(imgui.Col_.window_bg,         0.13,0.14,0.15,0.92)
        sc(imgui.Col_.frame_bg,          0.20,0.21,0.23)
        sc(imgui.Col_.frame_bg_hovered,  0.25,0.26,0.28)
        sc(imgui.Col_.frame_bg_active,   0.30,0.31,0.33)
        sc(imgui.Col_.title_bg,          0.10,0.11,0.12)
        sc(imgui.Col_.title_bg_active,   0.15,0.16,0.18)
        sc(imgui.Col_.menu_bar_bg,       0.10,0.11,0.12)
        sc(imgui.Col_.button,            0.25,0.40,0.60)
        sc(imgui.Col_.button_hovered,    0.35,0.50,0.75)
        sc(imgui.Col_.button_active,     0.20,0.35,0.55)
        sc(imgui.Col_.header,            0.25,0.40,0.60,0.7)
        sc(imgui.Col_.header_hovered,    0.30,0.45,0.65,0.8)
        sc(imgui.Col_.check_mark,        0.40,0.85,0.40)
        sc(imgui.Col_.slider_grab,       0.35,0.55,0.80)
        sc(imgui.Col_.slider_grab_active,0.45,0.65,0.90)
        sc(imgui.Col_.text,              0.90,0.90,0.90)
        sc(imgui.Col_.separator,         0.30,0.30,0.35)
        sc(imgui.Col_.popup_bg,          0.10,0.11,0.12,0.95)



if __name__=="__main__":
    import traceback
    dat=f06=""
    for a in sys.argv[1:]:
        al=a.lower()
        if   al.endswith(('.dat','.nas','.bdf')): dat=a
        elif al.endswith(('.f06','.op2','.neu')): f06=a
    if not dat:
        s=os.path.join(os.path.dirname(__file__),'samples','mixed_demo.dat')
        if os.path.exists(s): dat=s; print(f"[Info] Loading sample: {s}")
    try:
        MystranViewerApp().run(dat_file=dat, f06_file=f06)
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
