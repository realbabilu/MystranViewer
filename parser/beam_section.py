"""
Beam/bar cross-section geometry extractor.

Converts PBAR / PBEAM / PBARL / PROD property cards into a
BeamSection descriptor with:
  - b, h      : equivalent rectangular width × height (plane 2, plane 1)
  - shape     : 'rect' | 'circle' | 'pipe' | 'I' | 'T' | 'L' | 'box' | 'rod' | 'unknown'
  - profile   : list of (y, z) corner vertices in local section frame
                (origin = centroid, y = axis-2, z = axis-1/strong)
  - area      : cross-sectional area
  - label     : human-readable description string

Local section axes convention (NASTRAN):
  Beam x-axis  = along element (node1 → node2)
  Beam y-axis  = plane 1 normal (axis 1 = strong = h direction)
  Beam z-axis  = plane 2 normal (axis 2 = weak  = b direction)
  I1 = bh³/12  (strong, about y-axis)
  I2 = hb³/12  (weak,   about z-axis)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BeamSection:
    shape:   str   = 'unknown'   # 'rect','circle','pipe','I','T','L','box','rod','unknown'
    b:       float = 0.0         # width  (axis-2 direction, weak)
    h:       float = 0.0         # height (axis-1 direction, strong)
    area:    float = 0.0
    label:   str   = ''
    cap_ends: bool = True
    dims:    List[float] = field(default_factory=list)
    # polygon in (y, z) local coords — y=weak, z=strong
    profile: List[Tuple[float,float]] = field(default_factory=list)

    def render_half_extents(self):
        """Return (b/2, h/2) for fast box rendering."""
        return self.b * 0.5, self.h * 0.5


def section_area_inertias(section: BeamSection) -> Tuple[float, float, float]:
    """
    Return (A, Iy, Iz) about the centroidal local section axes.

    Local profile coordinates:
      y -> weak axis direction
      z -> strong axis direction

    Therefore:
      Iy = integral(z^2 dA)  (strong-axis bending inertia)
      Iz = integral(y^2 dA)  (weak-axis bending inertia)
    """
    prof = getattr(section, 'profile', None) or []
    if len(prof) >= 3:
        area2 = 0.0
        cy_num = 0.0
        cz_num = 0.0
        iy0 = 0.0
        iz0 = 0.0
        n = len(prof)
        for i in range(n):
            y0, z0 = prof[i]
            y1, z1 = prof[(i + 1) % n]
            cr = y0 * z1 - y1 * z0
            area2 += cr
            cy_num += (y0 + y1) * cr
            cz_num += (z0 + z1) * cr
            iy0 += (z0 * z0 + z0 * z1 + z1 * z1) * cr
            iz0 += (y0 * y0 + y0 * y1 + y1 * y1) * cr
        area = 0.5 * area2
        if abs(area) > 1e-12:
            cy = cy_num / (6.0 * area)
            cz = cz_num / (6.0 * area)
            iy = iy0 / 12.0 - area * cz * cz
            iz = iz0 / 12.0 - area * cy * cy
            return abs(float(area)), abs(float(iy)), abs(float(iz))

    A = float(getattr(section, 'area', 0.0) or 0.0)
    b = float(getattr(section, 'b', 0.0) or 0.0)
    h = float(getattr(section, 'h', 0.0) or 0.0)
    if A > 1e-12 and b > 1e-12 and h > 1e-12:
        iy = b * h * h * h / 12.0
        iz = h * b * b * b / 12.0
        return A, iy, iz
    return A, 0.0, 0.0


# ---------------------------------------------------------------------------
# Section profiles for known PBARL / PBEAML types
# Reference: MSC NASTRAN QRG, PBARL DIM definitions
# ---------------------------------------------------------------------------

def _rect_profile(b, h):
    """Axis-aligned rectangle centred at origin. y=weak(b), z=strong(h)."""
    hb, hh = b*0.5, h*0.5
    return [(-hb,-hh),(hb,-hh),(hb,hh),(-hb,hh)]


def _circle_profile(r, n=16):
    return [(r*math.cos(2*math.pi*i/n), r*math.sin(2*math.pi*i/n)) for i in range(n)]


def _pipe_profile(r_outer, r_inner, n=16):
    """Two concentric polygons — renderer draws outer filled, then inner cutout.
       For simplicity return outer polygon; renderer will use b=2r for cross-hatch."""
    return _circle_profile(r_outer, n)


def _I_profile(bf_top, tf_top, bf_bot, tf_bot, hw, tw):
    """
    I / H section.
    Dims:  bf_top=flange width top, tf_top=flange thick top,
           bf_bot=flange width bot, tf_bot=flange thick bot,
           hw=web height, tw=web thick.
    Total height H = tf_top + hw + tf_bot.
    Centroid assumed at mid-height.
    y = weak (horizontal), z = strong (vertical).
    """
    H  = tf_top + hw + tf_bot
    hH = H * 0.5
    # z coords from bottom: 0..H, shift by -hH
    z0  = -hH
    z1  = z0 + tf_bot
    z2  = z1 + hw
    z3  = z2 + tf_top
    hbt = bf_top * 0.5
    hbb = bf_bot * 0.5
    htw = tw * 0.5
    # CCW polygon
    return [
        (-hbb, z0), (hbb, z0),   # bottom flange bottom
        (hbb, z1), (htw, z1),    # bottom flange top → web
        (htw, z2), (hbt, z2),    # web top → top flange
        (hbt, z3), (-hbt, z3),   # top flange top
        (-hbt, z2), (-htw, z2),  # top flange → web
        (-htw, z1), (-hbb, z1),  # web → bottom flange
    ]


def _box_profile(b, h, t1, t2):
    """Closed box section. b=width, h=height, t1=side thick, t2=top/bot thick."""
    hb, hh = b*0.5, h*0.5
    # outer
    outer = [(-hb,-hh),(hb,-hh),(hb,hh),(-hb,hh)]
    return outer   # renderer will use outer bbox; full hollow not supported yet


def _T_profile(bf, tf, hw, tw):
    """T section. Flange on top."""
    H   = hw + tf
    # centroid approximately at distance from bottom:
    A_web  = hw * tw
    A_fl   = bf * tf
    A_tot  = A_web + A_fl
    z_cg   = (A_web*(hw/2) + A_fl*(hw + tf/2)) / A_tot
    hb     = bf * 0.5
    htw    = tw * 0.5
    z_bot  = -z_cg
    z_junc = z_bot + hw
    z_top  = z_junc + tf
    return [
        (-htw, z_bot), (htw, z_bot),
        (htw, z_junc), (hb, z_junc),
        (hb, z_top), (-hb, z_top),
        (-hb, z_junc), (-htw, z_junc),
    ]


def _L_profile(b, h, tb, th):
    """L (angle) section. b=horizontal leg, h=vertical leg."""
    # centroid approx
    A1 = b * tb      # horizontal
    A2 = (h-tb) * th # vertical
    A  = A1 + A2
    y_cg = (A1*(b/2) + A2*(th/2)) / A
    z_cg = (A1*(tb/2) + A2*(tb + (h-tb)/2)) / A
    # profile in (y, z) shifted by centroid
    pts = [
        (0,   0),   (b,   0),
        (b,   tb),  (th,  tb),
        (th,  h),   (0,   h),
    ]
    return [(y - y_cg, z - z_cg) for y, z in pts]


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def _safe_sqrt(x):
    return math.sqrt(max(x, 0.0))


def _param_tokens(params: dict):
    items = []
    for k, v in params.items():
        if not k.startswith('f'):
            continue
        try:
            idx = int(k[1:])
        except ValueError:
            continue
        sv = str(v).strip()
        if sv:
            items.append((idx, sv))
    items.sort(key=lambda kv: kv[0])
    return [v for _, v in items]


def _section_dim_count(sec_type: str) -> int:
    t = sec_type.upper()
    if t == 'ROD':
        return 1
    if t in ('TUBE', 'PIPE', 'BAR', 'RECT'):
        return 2
    if t in ('BOX', 'T', 'T1', 'L', 'CHAN', 'C', 'H'):
        return 4
    if t in ('I', 'I1'):
        return 6
    return 0


def _extract_pbeaml_dims(params: dict, sec_type: str):
    tokens = _param_tokens(params)
    sec_u = sec_type.upper()
    try:
        sec_pos = next(i for i, tok in enumerate(tokens) if tok.upper() == sec_u)
    except StopIteration:
        return []
    tail = tokens[sec_pos + 1:]
    dim_count = _section_dim_count(sec_u)
    if dim_count <= 0 or len(tail) < dim_count:
        return []

    def _is_so(tok: str) -> bool:
        return tok.upper() in ('YES', 'YESA', 'NO')

    stations = []
    i = 0

    # First station: DIM... NSM [SO]
    try:
        first_dims = [float(tok) for tok in tail[i:i + dim_count]]
    except ValueError:
        return []
    stations.append(first_dims)
    i += dim_count
    if i < len(tail):
        try:
            float(tail[i])
            i += 1  # NSM
        except ValueError:
            pass
    if i < len(tail) and _is_so(tail[i]):
        i += 1

    # Remaining stations: XXB DIM... NSM [SO]
    while i < len(tail):
        if i < len(tail):
            try:
                float(tail[i])
                i += 1  # XXB
            except ValueError:
                if _is_so(tail[i]):
                    i += 1
                    continue
                break
        if i + dim_count > len(tail):
            break
        try:
            dims = [float(tok) for tok in tail[i:i + dim_count]]
        except ValueError:
            break
        stations.append(dims)
        i += dim_count
        if i < len(tail):
            try:
                float(tail[i])
                i += 1  # NSM
            except ValueError:
                pass
        if i < len(tail) and _is_so(tail[i]):
            i += 1

    if not stations:
        return []
    return [max(vals[j] for vals in stations) for j in range(dim_count)]


def extract_section(ptype: str, mid: int, params: dict) -> BeamSection:
    """
    Convert property card fields to BeamSection.
    params keys: f3, f4, f5, ... as stored by dat_parser.
    """
    def g(key, default=0.0):
        v = params.get(key, '')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).strip()) if str(v).strip() else default
        except ValueError:
            return default

    # ---- PROD (rod / circular solid) ----
    if ptype == 'PROD':
        A  = g('f3')
        J  = g('f4')
        r  = _safe_sqrt(A / math.pi) if A > 0 else 0.0
        dia = 2*r
        prof = _circle_profile(r) if r > 0 else _rect_profile(0.01, 0.01)
        return BeamSection(
            shape='circle', b=dia, h=dia, area=A, cap_ends=True, dims=[dia],
            label=f'ROD A={A:.4g}',
            profile=prof
        )

    # ---- PBAR ----
    # PBAR  PID MID  A   I1   I2   J   NSM
    #             f3  f4   f5   f6  f7
    if ptype == 'PBAR':
        A  = g('f3')
        I1 = g('f4')   # strong (h direction)
        I2 = g('f5')   # weak   (b direction)
        if A > 1e-30 and I1 > 1e-30 and I2 > 1e-30:
            h = _safe_sqrt(12 * I1 / A)
            b = _safe_sqrt(12 * I2 / A)
        elif A > 1e-30 and I1 > 1e-30:
            h = _safe_sqrt(12 * I1 / A)
            b = A / h if h > 0 else h
        elif A > 1e-30:
            side = _safe_sqrt(A)
            b = h = side
        else:
            b = h = 0.1
        prof = _rect_profile(b, h)
        return BeamSection(
            shape='rect', b=b, h=h, area=A, cap_ends=True, dims=[b, h],
            label=f'BAR b={b:.4g} h={h:.4g}',
            profile=prof
        )

    # ---- PBEAM ----
    # PBEAM PID MID  A   I1   I2   I12  J  NSM  (per station)
    #              f3  f4   f5   f6   f7  f8
    if ptype == 'PBEAM':
        A  = g('f3')
        I1 = g('f4')
        I2 = g('f5')
        if A > 1e-30 and I1 > 1e-30 and I2 > 1e-30:
            h = _safe_sqrt(12 * I1 / A)
            b = _safe_sqrt(12 * I2 / A)
        elif A > 1e-30:
            side = _safe_sqrt(A)
            b = h = side
        else:
            b = h = 0.1
        prof = _rect_profile(b, h)
        return BeamSection(
            shape='rect', b=b, h=h, area=A, cap_ends=True, dims=[b, h],
            label=f'BEAM b={b:.4g} h={h:.4g}',
            profile=prof
        )

    # ---- PBARL / PBEAML ----
    # PBARL  PID MID  — GROUP TYPE
    #              f3  f4    f5
    # Dimensions start at f6, f7, f8, ...
    # GROUP default = 'MSCBML0'
    if ptype in ('PBARL', 'PBEAML'):
        # f3=group(optional), f4=type or f3=type
        # Find the section type string
        sec_type = ''
        for k in ('f5', 'f4', 'f3'):
            v = str(params.get(k, '')).strip().upper()
            if v and not v.replace('.','').replace('-','').replace('+','').isnumeric():
                sec_type = v
                break

        if ptype == 'PBEAML':
            dims = _extract_pbeaml_dims(params, sec_type)
        else:
            # Collect dimension values (all numeric fields after sec_type key)
            dims = []
            started = False
            for tok in _param_tokens(params):
                if not started:
                    if tok.upper() == sec_type:
                        started = True
                    continue
                try:
                    dims.append(float(tok))
                except ValueError:
                    pass

        return _pbarl_section(sec_type, dims)

    # Fallback
    return BeamSection(shape='unknown', b=0.1, h=0.1, area=0.01, cap_ends=True, dims=[0.1, 0.1],
                       label=f'{ptype} (unknown)',
                       profile=_rect_profile(0.1, 0.1))


def _pbarl_section(sec_type: str, dims: list) -> BeamSection:
    """
    Parse PBARL/PBEAML section type and dimensions.
    Reference: MSC NASTRAN QRG Table 3-12
    """
    def d(i, default=0.0):
        return dims[i] if i < len(dims) else default

    t = sec_type.upper()

    # ROD: DIM1=r
    if t == 'ROD':
        r = d(0)
        A = math.pi * r**2
        return BeamSection(shape='circle', b=2*r, h=2*r, area=A, cap_ends=True, dims=[2*r],
                           label=f'ROD r={r:.4g}',
                           profile=_circle_profile(r))

    # TUBE: DIM1=r_outer, DIM2=r_inner
    if t in ('TUBE', 'PIPE'):
        ro, ri = d(0), d(1)
        A = math.pi*(ro**2 - ri**2)
        return BeamSection(shape='pipe', b=2*ro, h=2*ro, area=A, cap_ends=False, dims=[2*ro, 2*ri],
                           label=f'TUBE ro={ro:.4g} ri={ri:.4g}',
                           profile=_circle_profile(ro))

    # BAR / RECT: DIM1=b(width), DIM2=h(height)
    if t in ('BAR', 'RECT'):
        b, h = d(0), d(1) if len(dims) > 1 else d(0)
        return BeamSection(shape='rect', b=b, h=h, area=b*h, cap_ends=True, dims=[b, h],
                           label=f'RECT {b:.4g}×{h:.4g}',
                           profile=_rect_profile(b, h))

    # BOX: DIM1=b, DIM2=h, DIM3=t1(side), DIM4=t2(top/bot)
    if t == 'BOX':
        b, h, t1, t2 = d(0), d(1), d(2), d(3)
        A = b*h - (b-2*t1)*(h-2*t2)
        return BeamSection(shape='box', b=b, h=h, area=A, cap_ends=True, dims=[b, h, t1, t2],
                           label=f'BOX {b:.4g}×{h:.4g} t={t1:.4g}/{t2:.4g}',
                           profile=_rect_profile(b, h))

    # I: DIM1=H, DIM2=bf_top, DIM3=bf_bot, DIM4=tf_top, DIM5=tf_bot, DIM6=tw
    if t in ('I', 'I1'):
        H, bf1, bf2, tf1, tf2, tw = d(0), d(1), d(2), d(3), d(4), d(5)
        hw = max(H - tf1 - tf2, 0.0)
        A = bf1*tf1 + bf2*tf2 + hw*tw
        B = max(bf1, bf2)
        return BeamSection(shape='I', b=B, h=H, area=A, cap_ends=True, dims=[H, bf1, bf2, tf1, tf2, tw],
                           label=f'I {B:.4g}×{H:.4g}',
                           profile=_I_profile(bf1, tf1, bf2, tf2, hw, tw))

    # H: DIM1=H, DIM2=bf, DIM3=tf, DIM4=tw
    if t == 'H':
        H, bf, tf, tw = d(0), d(1), d(2), d(3)
        hw = max(H - 2*tf, 0.0)
        A = 2*bf*tf + hw*tw
        return BeamSection(shape='I', b=bf, h=H, area=A, cap_ends=True, dims=[H, bf, bf, tf, tf, tw],
                           label=f'H {bf:.4g}×{H:.4g}',
                           profile=_I_profile(bf, tf, bf, tf, hw, tw))

    # T: DIM1=bf, DIM2=hw, DIM3=tw, DIM4=tf
    if t in ('T', 'T1'):
        bf, hw, tw, tf = d(0), d(1), d(2), d(3)
        A = bf*tf + hw*tw
        H = hw + tf
        return BeamSection(shape='T', b=bf, h=H, area=A, cap_ends=True, dims=[bf, H, tf, tw],
                           label=f'T {bf:.4g}×{H:.4g}',
                           profile=_T_profile(bf, tf, hw, tw))

    # L: DIM1=b(horiz leg), DIM2=h(vert leg), DIM3=tb, DIM4=th
    if t == 'L':
        b, h, tb, th = d(0), d(1), d(2), d(3)
        A = b*tb + (h-tb)*th
        return BeamSection(shape='L', b=b, h=h, area=A, cap_ends=True, dims=[b, h, tb, th],
                           label=f'L {b:.4g}×{h:.4g}',
                           profile=_L_profile(b, h, tb, th))

    # CHAN: DIM1=b(flange), DIM2=h(height), DIM3=tw, DIM4=tf
    if t in ('CHAN', 'C'):
        b, h, tw, tf = d(0), d(1), d(2), d(3)
        A = 2*b*tf + (h-2*tf)*tw
        return BeamSection(shape='rect', b=b, h=h, area=A, cap_ends=True, dims=[b, h, tw, tf],
                           label=f'CHAN {b:.4g}×{h:.4g}',
                           profile=_rect_profile(b, h))

    # Fallback: use whatever dims we have as b×h
    if len(dims) >= 2:
        b, h = dims[0], dims[1]
        A = b * h
    elif len(dims) == 1:
        b = h = dims[0]
        A = b * h
    else:
        b = h = 0.1; A = 0.01

    return BeamSection(shape='rect', b=b, h=h, area=A, cap_ends=True, dims=[b, h],
                       label=f'{sec_type} {b:.4g}×{h:.4g}',
                       profile=_rect_profile(b, h))
