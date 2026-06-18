"""
Notation overlays via ImGui foreground draw list.
Renders in 2D screen space — no GL viewport change, always safe.

Overlays:
  - Node dots
  - Node numbers
  - Element numbers (at centroid)
  - SPC/constraint symbols (arrows per DOF)
  - Force/moment arrows
"""

import numpy as np
from typing import Optional, Dict, Tuple
from imgui_bundle import imgui

from parser.dat_parser import MystranModel


# ---------------------------------------------------------------------------
# Colors (RGBA as imgui uint32: 0xAABBGGRR)
# ---------------------------------------------------------------------------
C_NODE_DOT   = 0xFFFFFFCC   # light yellow
C_NODE_NUM   = 0xFFFFFFCC
C_ELEM_NUM   = 0xFF88FFFF   # cyan
C_SPC_PIN    = 0xFF3333FF   # red (constraint)
C_SPC_TEXT   = 0xFF5555FF
C_FORCE_ARR  = 0xFF00FFFF   # yellow
C_MOMENT_ARR = 0xFF00AAFF


def _rgba(r, g, b, a=1.0):
    """Float RGBA -> imgui uint32 (ABGR layout)."""
    ri = int(np.clip(r*255, 0, 255))
    gi = int(np.clip(g*255, 0, 255))
    bi = int(np.clip(b*255, 0, 255))
    ai = int(np.clip(a*255, 0, 255))
    return (ai << 24) | (bi << 16) | (gi << 8) | ri


# Colors as plain ints — computed at first use, not at import
def _lazy_colors():
    global _C_NODE, _C_ENUM, _C_SPC, _C_FORCE
    _C_NODE  = _rgba(1.0, 1.0, 0.6, 1.0)
    _C_ENUM  = _rgba(0.4, 1.0, 1.0, 1.0)
    _C_SPC   = _rgba(1.0, 0.3, 0.2, 1.0)
    _C_FORCE = _rgba(1.0, 0.9, 0.1, 1.0)
_C_NODE = _C_ENUM = _C_SPC = _C_FORCE = 0xFFFFFFFF  # placeholders


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _project(xyz, mvp, win_w, win_h, ortho=False):
    """
    Project 3D world point to 2D screen coords.
    Returns (sx, sy) or None if off-screen / behind camera.
    """
    p = mvp @ np.array([*xyz, 1.0], dtype=np.float32)
    pw = p[3]
    if not ortho and pw < 0.01:
        return None
    if abs(pw) < 1e-9:
        return None
    ndc_x = p[0] / pw
    ndc_y = p[1] / pw
    if abs(ndc_x) > 1.4 or abs(ndc_y) > 1.4:
        return None
    sx = (ndc_x + 1.0) * 0.5 * win_w
    sy = (1.0 - ndc_y) * 0.5 * win_h
    return (float(sx), float(sy))


# ---------------------------------------------------------------------------
# SPC DOF symbol
# ---------------------------------------------------------------------------

# DOF label positions around the node marker
_DOF_LABELS = ['Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz']
_DOF_DIRS   = [        # screen offset directions for each DOF label
    ( 8,  0),          # Tx right
    (-8,  0),          # Ty left
    ( 0, -8),          # Tz up
    (12,  6),          # Rx lower-right
    (-12, 6),          # Ry lower-left
    ( 0, 10),          # Rz down
]

def _dof_string_from_spc(dofs_str: str):
    """Parse SPC dofs string e.g. '123456' -> list of active DOF indices [0..5]."""
    active = []
    for ch in str(dofs_str):
        if ch.isdigit():
            d = int(ch)
            if 1 <= d <= 6:
                active.append(d - 1)
    return active


# ---------------------------------------------------------------------------
# Main draw function
# ---------------------------------------------------------------------------

def draw_notation(
    model:      MystranModel,
    mvp:        np.ndarray,
    win_w:      int,
    win_h:      int,
    ortho:      bool = True,
    *,
    show_nodes:       bool = False,
    show_node_nums:   bool = False,
    show_elem_nums:   bool = False,
    show_constraints: bool = True,
    show_forces:      bool = False,
    node_dot_size:    float = 3.0,
    font_scale:       float = 1.0,
    # left/right panel margins to avoid drawing under UI
    margin_left:  int = 275,
    margin_right: int = 260,
):
    _lazy_colors()
    if not any([show_nodes, show_node_nums, show_elem_nums,
               show_constraints, show_forces]):
        return

    dl = imgui.get_foreground_draw_list()
    pos = {nid: n.xyz for nid, n in model.nodes.items()}

    # Panel clip region (avoid drawing under left/right panels)
    clip_x0 = float(margin_left)
    clip_x1 = float(win_w - margin_right)
    clip_y0 = 20.0
    clip_y1 = float(win_h)
    dl.push_clip_rect((clip_x0, clip_y0), (clip_x1, clip_y1), True)

    # ── Node dots ──────────────────────────────────────────────────
    if show_nodes:
        for nid, node in model.nodes.items():
            sc = _project(node.xyz, mvp, win_w, win_h, ortho)
            if sc is None: continue
            dl.add_circle_filled(sc, node_dot_size, _C_NODE)

    # ── Node numbers ───────────────────────────────────────────────
    if show_node_nums:
        for nid, node in model.nodes.items():
            sc = _project(node.xyz, mvp, win_w, win_h, ortho)
            if sc is None: continue
            label = str(nid)
            dl.add_text(
                (sc[0] + node_dot_size + 2, sc[1] - 6),
                _C_NODE, label
            )

    # ── Element numbers (at centroid) ──────────────────────────────
    if show_elem_nums:
        for eid, elem in model.elements.items():
            nds = [pos.get(n) for n in elem.nodes if pos.get(n) is not None]
            if not nds: continue
            centroid = np.mean(nds, axis=0)
            sc = _project(centroid, mvp, win_w, win_h, ortho)
            if sc is None: continue
            dl.add_text((sc[0] - 6, sc[1] - 5), _C_ENUM, str(eid))

    # ── Constraints ────────────────────────────────────────────────
    if show_constraints:
        # Group SPCs by node
        spc_by_node: Dict[int, set] = {}
        for spc in model.spcs:
            nid = spc.node_id
            if nid not in spc_by_node:
                spc_by_node[nid] = set()
            for d in _dof_string_from_spc(spc.dofs):
                spc_by_node[nid].add(d)

        model_scale = model.scale()
        for nid, dofs in spc_by_node.items():
            node = model.nodes.get(nid)
            if node is None: continue
            sc = _project(node.xyz, mvp, win_w, win_h, ortho)
            if sc is None: continue

            # Draw triangle pin symbol
            s = 8.0
            dl.add_triangle_filled(
                (sc[0], sc[1]),
                (sc[0] - s, sc[1] + s*1.5),
                (sc[0] + s, sc[1] + s*1.5),
                _C_SPC
            )
            dl.add_triangle(
                (sc[0], sc[1]),
                (sc[0] - s, sc[1] + s*1.5),
                (sc[0] + s, sc[1] + s*1.5),
                _rgba(0,0,0,0.8), 1.0
            )

            # DOF labels (only if 1-5 DOFs constrained, not all 6)
            if 0 < len(dofs) < 6:
                dof_str = ''.join(str(d+1) for d in sorted(dofs))
                dl.add_text(
                    (sc[0] + s + 2, sc[1] + 2),
                    _C_SPC, dof_str
                )

    # ── Force arrows ───────────────────────────────────────────────
    if show_forces:
        for force in model.forces:
            node = model.nodes.get(force.node_id)
            if node is None: continue
            sc = _project(node.xyz, mvp, win_w, win_h, ortho)
            if sc is None: continue

            # Project force direction
            tip_world = node.xyz + force.direction * (force.magnitude / (abs(force.magnitude) + 1e-30))
            sc_tip = _project(tip_world, mvp, win_w, win_h, ortho)
            if sc_tip is None: continue

            # Normalize to fixed screen length
            dx = sc_tip[0] - sc[0]; dy = sc_tip[1] - sc[1]
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6: continue
            arrow_len = 30.0
            dx = dx/length * arrow_len; dy = dy/length * arrow_len

            # Shaft
            dl.add_line(
                (sc[0], sc[1]),
                (sc[0]+dx, sc[1]+dy),
                _C_FORCE, 1.5
            )
            # Arrowhead
            angle = np.arctan2(dy, dx)
            for da in [2.6, -2.6]:
                ax = sc[0]+dx - np.cos(angle+da)*8
                ay = sc[1]+dy - np.sin(angle+da)*8
                dl.add_line((sc[0]+dx, sc[1]+dy), (ax,ay), _C_FORCE, 1.5)

    dl.pop_clip_rect()


# ---------------------------------------------------------------------------
# Local element axis arrows
# ---------------------------------------------------------------------------

def _elem_local_axes(elem, model):
    """
    Compute (centroid, x1, x2, x3) for an element.
    Returns None if not computable.
    x1=red, x2=green, x3=blue (normal for shell, strong for beam)
    """
    pos = {nid: model.nodes[nid].xyz
           for nid in elem.nodes if nid in model.nodes}
    if len(pos) < 2:
        return None

    pts = [pos[n] for n in elem.nodes if n in pos]
    centroid = np.mean(pts, axis=0).astype(np.float32)

    if elem.type in ('CQUAD4', 'CTRIA3'):
        # x1 = first edge, x3 = normal, x2 = x3 × x1
        p0, p1 = pts[0], pts[1]
        p3 = pts[3] if elem.type == 'CQUAD4' and len(pts) >= 4 else pts[2]
        x1 = p1 - p0
        n1 = np.linalg.norm(x1)
        if n1 < 1e-12: return None
        x1 = (x1 / n1).astype(np.float32)

        x3 = np.cross(p1 - p0, p3 - p0).astype(np.float32)
        n3 = np.linalg.norm(x3)
        if n3 < 1e-12: return None
        x3 = x3 / n3

        x2 = np.cross(x3, x1).astype(np.float32)
        x2 = x2 / (np.linalg.norm(x2) + 1e-12)
        return centroid, x1, x2, x3

    elif elem.type in ('CBAR', 'CBEAM', 'CROD'):
        p0, p1 = pts[0], pts[1]
        x1 = p1 - p0
        n1 = np.linalg.norm(x1)
        if n1 < 1e-12: return None
        x1 = (x1 / n1).astype(np.float32)

        v_orient = getattr(elem, 'v_orient', None)
        v = (v_orient / np.linalg.norm(v_orient)).astype(np.float32) \
            if v_orient is not None and np.linalg.norm(v_orient) > 1e-9 \
            else (np.array([0,0,1],dtype=np.float32)
                  if abs(x1[2]) < 0.9 else np.array([0,1,0],dtype=np.float32))

        x3 = v - np.dot(v, x1) * x1
        n3 = np.linalg.norm(x3)
        if n3 < 1e-12: return None
        x3 = (x3 / n3).astype(np.float32)
        x2 = np.cross(x3, x1).astype(np.float32)
        x2 = x2 / (np.linalg.norm(x2) + 1e-12)
        return centroid, x1, x2, x3

    elif elem.type in ('CHEXA', 'CPENTA', 'CTETRA'):
        # Global aligned for solids (approximate)
        x1 = np.array([1,0,0], dtype=np.float32)
        x2 = np.array([0,1,0], dtype=np.float32)
        x3 = np.array([0,0,1], dtype=np.float32)
        return centroid, x1, x2, x3

    return None


def draw_local_axes(
    model:   MystranModel,
    mvp:     np.ndarray,
    win_w:   int,
    win_h:   int,
    ortho:   bool  = True,
    length:  float = 0.0,   # 0 = auto (5% of model scale)
    *,
    show_shell: bool = True,
    show_frame: bool = True,
    show_solid: bool = False,
    margin_left:  int = 275,
    margin_right: int = 260,
):
    """Draw RGB local axis arrows at element centroids."""
    if not any([show_shell, show_frame, show_solid]):
        return

    model_scale = model.scale()
    arrow_len   = length if length > 0 else model_scale * 0.07

    dl = imgui.get_foreground_draw_list()
    dl.push_clip_rect(
        (float(margin_left), 20.0),
        (float(win_w - margin_right), float(win_h)),
        True
    )

    # Colors: x1=red, x2=green, x3=blue (standard RGB)
    col_x1 = _rgba(1.0, 0.2, 0.2, 0.95)   # red
    col_x2 = _rgba(0.2, 0.9, 0.2, 0.95)   # green
    col_x3 = _rgba(0.3, 0.5, 1.0, 0.95)   # blue

    for eid, elem in model.elements.items():
        # Filter by element type category
        if elem.type in ('CQUAD4','CTRIA3') and not show_shell: continue
        if elem.type in ('CBAR','CBEAM','CROD') and not show_frame: continue
        if elem.type in ('CHEXA','CPENTA','CTETRA') and not show_solid: continue

        axes = _elem_local_axes(elem, model)
        if axes is None: continue
        centroid, x1, x2, x3 = axes

        sc0 = _project(centroid, mvp, win_w, win_h, ortho)
        if sc0 is None: continue

        axis_glyphs = [
            (x1, col_x1, '1'),
            (x2, col_x2, '2'),
            (x3, col_x3, '3'),
        ]
        if elem.type in ('CBAR', 'CBEAM', 'CROD'):
            axis_glyphs = [
                (x1, col_x1, 'x'),
                (x2, col_x2, 'z'),
                (x3, col_x3, 'y'),
            ]
        for vec, col, label in axis_glyphs:
            tip_world = centroid + vec * arrow_len
            sc1 = _project(tip_world, mvp, win_w, win_h, ortho)
            if sc1 is None: continue

            # Shaft
            dl.add_line(sc0, sc1, col, 1.5)

            # Arrowhead (small triangle)
            dx = sc1[0]-sc0[0]; dy = sc1[1]-sc0[1]
            ln = np.sqrt(dx**2+dy**2)
            if ln < 2: continue
            dx /= ln; dy /= ln
            ah = 6.0
            aw = 3.0
            # Arrow tip points
            lx = -dy*aw; ly = dx*aw
            ax1 = (sc1[0]-dx*ah+lx, sc1[1]-dy*ah+ly)
            ax2 = (sc1[0]-dx*ah-lx, sc1[1]-dy*ah-ly)
            dl.add_triangle_filled(sc1, ax1, ax2, col)

    dl.pop_clip_rect()
