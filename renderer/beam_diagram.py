"""
Beam internal force diagram renderer.

Reads CBEAM/CBAR force results from pyNastran OP2 and draws:
  - Bending moment diagrams (M1, M2)
  - Shear force diagrams (V1, V2)  
  - Axial force diagram (N)
  - Torque diagram (T)
  - Deformed beam shape (from interstation displacements if available)

Each beam can have up to 11 interstation points (stations 0.0..1.0).
Diagrams are drawn as filled areas perpendicular to the beam axis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from parser.beam_section import section_area_inertias
from parser.beam_stress import beam_stress_at_station, pbeam_cdef_points


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BeamStationData:
    """Force/moment data at one station along a beam element."""
    eid:    int
    nid:    int       # internal station node id (0 = end A, >0 = interstation)
    sd:     float     # station distance 0.0 (end A) → 1.0 (end B)
    bm1:    float     # bending moment plane 1 (about axis 2)
    bm2:    float     # bending moment plane 2 (about axis 3)
    ts1:    float     # shear force plane 1
    ts2:    float     # shear force plane 2
    af:     float     # axial force
    ttrq:   float     # total torque
    wtrq:   float     # warping torque
    sxc:    float = 0.0
    sxd:    float = 0.0
    sxe:    float = 0.0
    sxf:    float = 0.0
    smax:   float = 0.0
    smin:   float = 0.0


@dataclass
class BeamDiagramData:
    """All station data for one beam element."""
    eid:      int
    stations: List[BeamStationData] = field(default_factory=list)

    @property
    def sd_arr(self): return np.array([s.sd for s in self.stations])

    def values(self, key):
        return np.array([getattr(s, key) for s in self.stations])


def _empty_station(eid: int, sd: float) -> BeamStationData:
    return BeamStationData(
        eid=eid, nid=0, sd=sd,
        bm1=0.0, bm2=0.0, ts1=0.0, ts2=0.0,
        af=0.0, ttrq=0.0, wtrq=0.0,
        sxc=0.0, sxd=0.0, sxe=0.0, sxf=0.0, smax=0.0, smin=0.0)


def _reduced_station_data(bd: BeamDiagramData, result_key: str) -> BeamDiagramData:
    vals = bd.values(result_key)
    n = len(vals)
    if n <= 3:
        return bd
    out = BeamDiagramData(eid=bd.eid)
    out.stations.append(bd.stations[0])

    # If the member crosses zero between the ends, add one interpolated zero point
    # for a cleaner comparison view without carrying the whole station list.
    v0 = float(vals[0])
    v1 = float(vals[-1])
    if abs(v0) > 1e-12 and abs(v1) > 1e-12 and ((v0 < 0.0 < v1) or (v1 < 0.0 < v0)):
        t = abs(v0) / (abs(v0) + abs(v1))
        sd0 = float(bd.stations[0].sd)
        sd1 = float(bd.stations[-1].sd)
        st = _empty_station(bd.eid, sd0 + (sd1 - sd0) * t)
        setattr(st, result_key, 0.0)
        out.stations.append(st)

    out.stations.append(bd.stations[-1])
    return out


_BEAM_STRESS_KEYS = ('sxc', 'sxd', 'sxe', 'sxf', 'smax', 'smin')


# ---------------------------------------------------------------------------
# OP2 reader
# ---------------------------------------------------------------------------

def read_beam_forces_op2(filepath: str, subcase: int = 1) -> Dict[int, BeamDiagramData]:
    """
    Read CBEAM force results from OP2 file.
    Returns dict: eid -> BeamDiagramData
    """
    try:
        from pyNastran.op2.op2 import OP2
    except ImportError:
        raise ImportError("pyNastran required: pip install pyNastran")

    op2 = OP2(debug=False)
    op2.read_op2(filepath, combine=True)

    result = {}

    # Try CBEAM first (has interstation), then CBAR (end A/B only)
    force = op2.op2_results.force
    candidates = [
        (getattr(force, 'cbeam_force',        None), 'cbeam_force'),
        (getattr(force, 'cbar_force_10nodes', None), 'cbar_force_10nodes'),
        (getattr(force, 'cbar_force',         None), 'cbar_force'),
    ]

    for table_dict, name in candidates:
        if not table_dict: continue

        sc_data = table_dict.get(subcase)
        if sc_data is None and table_dict:
            sc_data = table_dict[sorted(table_dict.keys())[0]]
        if sc_data is None: continue

        # element_node: (nrows,2) col0=eid col1=station_nid
        # data: (ntimes, nrows, 8): sd bm1 bm2 ts1 ts2 af ttrq wtrq
        eids = sc_data.element_node[:, 0]
        nids = sc_data.element_node[:, 1]
        data = sc_data.data[-1]   # last time step

        for i in range(len(eids)):
            eid = int(eids[i]); nid = int(nids[i])
            row = data[i]
            sd  = float(row[0])
            st  = BeamStationData(
                eid=eid, nid=nid, sd=sd,
                bm1=float(row[1]), bm2=float(row[2]),
                ts1=float(row[3]), ts2=float(row[4]),
                af =float(row[5]), ttrq=float(row[6]), wtrq=float(row[7])
            )
            if eid not in result:
                result[eid] = BeamDiagramData(eid=eid)
            result[eid].stations.append(st)

        for bd in result.values():
            bd.stations.sort(key=lambda s: s.sd)

        print(f"[BeamDiagram] {name}: {len(result)} beams, "
              f"{sum(len(b.stations) for b in result.values())} total stations")
        if result: break

    return result


def read_beam_stress_op2(filepath: str, subcase: int = 1) -> Dict[int, List[BeamStationData]]:
    """
    Read CBEAM stress results from OP2 file.
    Returns dict: eid -> list[BeamStationData] with sd/nid + stress-only fields filled.
    """
    try:
        from pyNastran.op2.op2 import OP2
    except ImportError:
        raise ImportError("pyNastran required: pip install pyNastran")

    op2 = OP2(debug=False)
    op2.read_op2(filepath, combine=True)
    stress = op2.op2_results.stress
    table_dict = getattr(stress, 'cbeam_stress', None)
    if not table_dict:
        return {}
    sc_data = table_dict.get(subcase)
    if sc_data is None and table_dict:
        sc_data = table_dict[sorted(table_dict.keys())[0]]
    if sc_data is None:
        return {}

    result: Dict[int, List[BeamStationData]] = {}
    eids = sc_data.element_node[:, 0]
    nids = sc_data.element_node[:, 1]
    data = sc_data.data[-1]

    # RealBeamStressArray headers: sxc,sxd,sxe,sxf,smax,smin,MS_tension,MS_compression
    for i in range(len(eids)):
        eid = int(eids[i]); nid = int(nids[i])
        row = data[i]
        sd = 0.0
        if eid in result and result[eid]:
            sd = 1.0
        st = BeamStationData(
            eid=eid, nid=nid, sd=sd,
            bm1=0.0, bm2=0.0, ts1=0.0, ts2=0.0,
            af=0.0, ttrq=0.0, wtrq=0.0,
            sxc=float(row[0]), sxd=float(row[1]), sxe=float(row[2]), sxf=float(row[3]),
            smax=float(row[4]), smin=float(row[5]))
        result.setdefault(eid, []).append(st)

    for sts in result.values():
        sts.sort(key=lambda s: s.sd)
    print(f"[BeamDiagram] cbeam_stress: {len(result)} beams, {sum(len(v) for v in result.values())} total stations")
    return result


def beam_stress_available_op2(filepath: str, subcase: int = 1) -> bool:
    try:
        from pyNastran.op2.op2 import OP2
    except ImportError:
        return False
    op2 = OP2(debug=False)
    op2.read_op2(filepath, combine=True)
    stress = op2.op2_results.stress
    candidates = [
        getattr(stress, 'cbeam_stress', None),
        getattr(stress, 'cbar_stress_10nodes', None),
        getattr(stress, 'cbar_stress', None),
    ]
    for table_dict in candidates:
        if not table_dict:
            continue
        sc_data = table_dict.get(subcase)
        if sc_data is None and table_dict:
            sc_data = table_dict[sorted(table_dict.keys())[0]]
        if sc_data is not None:
            return True
    return False


def _getattr_deep(obj, path):
    """Safe deep attribute access: 'a.b.c' -> obj.a.b.c"""
    try:
        for part in path.split('.'):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return None


def _beam_local_frame(p0: np.ndarray, p1: np.ndarray, v_orient=None):
    ex = p1 - p0
    L = np.linalg.norm(ex)
    if L < 1e-30:
        return (
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        )
    ex = (ex / L).astype(np.float32)
    if v_orient is not None and np.linalg.norm(v_orient) > 1e-9:
        v = (v_orient / np.linalg.norm(v_orient)).astype(np.float32)
    else:
        v = np.array([0, 0, 1], dtype=np.float32) if abs(ex[2]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
    ez = v - np.dot(v, ex) * ex
    nez = np.linalg.norm(ez)
    if nez < 1e-12:
        p = np.array([1, 0, 0], dtype=np.float32) if abs(ex[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        ez = np.cross(ex, p)
        nez = np.linalg.norm(ez)
    ez = (ez / (nez + 1e-30)).astype(np.float32)
    ey = np.cross(ez, ex).astype(np.float32)
    ey /= (np.linalg.norm(ey) + 1e-30)
    return ex, ey, ez


def _pload1_result_component(pl, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray, result_key: str) -> Optional[float]:
    typ = str(pl.get('type', '')).strip().upper()
    if not typ:
        return None
    ex64 = np.asarray(ex, dtype=np.float64)
    ey64 = np.asarray(ey, dtype=np.float64)
    ez64 = np.asarray(ez, dtype=np.float64)

    if typ in ('FX', 'FY', 'FZ'):
        gvec = {
            'FX': np.array([1.0, 0.0, 0.0], dtype=np.float64),
            'FY': np.array([0.0, 1.0, 0.0], dtype=np.float64),
            'FZ': np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }[typ]
        if result_key in ('bm1', 'ts1'):
            return float(np.dot(gvec, ez64))
        if result_key in ('bm2', 'ts2'):
            return float(np.dot(gvec, ey64))
        if result_key == 'af':
            return float(np.dot(gvec, ex64))
        return None

    if typ in ('FXE', 'FYE', 'FZE'):
        if result_key == 'af' and typ == 'FXE':
            return 1.0
        if result_key in ('bm1', 'ts1') and typ == 'FZE':
            return 1.0
        if result_key in ('bm2', 'ts2') and typ == 'FYE':
            return 1.0
        return None

    if typ in ('MX', 'MY', 'MZ'):
        gvec = {
            'MX': np.array([1.0, 0.0, 0.0], dtype=np.float64),
            'MY': np.array([0.0, 1.0, 0.0], dtype=np.float64),
            'MZ': np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }[typ]
        if result_key == 'ttrq':
            return float(np.dot(gvec, ex64))
        return None

    if typ in ('MXE', 'MYE', 'MZE'):
        if result_key == 'ttrq' and typ == 'MXE':
            return 1.0
        return None

    return None


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


def _beam_timoshenko_props(model, elem, length: float):
    if model is None or elem is None or length <= 1e-12:
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
    return float(mat.E), float(G), float(area), float(iy), float(iz), float(kappa)


def _float_param(params: dict, key: str, default: float = 0.0) -> float:
    v = params.get(key, default)
    if isinstance(v, (int, float)):
        return float(v)
    txt = str(v).strip()
    if not txt:
        return default
    try:
        return float(txt)
    except ValueError:
        return default


def _property_cdef_points(prop):
    params = getattr(prop, 'params', {}) or {}
    ptype = str(getattr(prop, 'type', '')).upper()
    if ptype == 'PBEAM':
        pts = pbeam_cdef_points(params)
        if any(abs(v) > 1e-12 for pair in pts.values() for v in pair):
            return pts
    sec = getattr(prop, 'section', None)
    if sec is None:
        return None
    dims = list(getattr(sec, 'dims', []) or [])
    shp = str(getattr(sec, 'shape', '')).upper()
    if shp in ('BAR', 'RECT', 'BOX') and len(dims) >= 2:
        b = float(dims[0]); h = float(dims[1])
        hb = 0.5 * b; hh = 0.5 * h
        return {
            'C': ( hb,  hh),
            'D': (-hb,  hh),
            'E': (-hb, -hh),
            'F': ( hb, -hh),
        }
    if ptype == 'PBAR':
        b = _float_param(params, 'f7', 0.0)
        h = _float_param(params, 'f8', 0.0)
        if abs(b) > 1e-12 and abs(h) > 1e-12:
            hb = 0.5 * b; hh = 0.5 * h
            return {
                'C': ( hb,  hh),
                'D': (-hb,  hh),
                'E': (-hb, -hh),
                'F': ( hb, -hh),
            }
    return None


def _exact_beam_stress_data(
    beam_data: Dict[int, BeamDiagramData],
    model,
) -> Dict[int, BeamDiagramData]:
    out: Dict[int, BeamDiagramData] = {}
    for eid, bd in beam_data.items():
        elem = model.elements.get(eid)
        if elem is None:
            out[eid] = bd
            continue
        prop = model.properties.get(getattr(elem, 'pid', 0))
        if prop is None or str(getattr(prop, 'type', '')).upper() not in ('PBEAM', 'PBAR'):
            out[eid] = bd
            continue
        params = getattr(prop, 'params', {}) or {}
        A = _float_param(params, 'f3', 0.0)
        I1 = _float_param(params, 'f4', 0.0)
        I2 = _float_param(params, 'f5', 0.0)
        I12 = _float_param(params, 'f6', 0.0)
        pts = _property_cdef_points(prop)
        if A <= 1e-12 or I1 <= 1e-16 or I2 <= 1e-16 or pts is None:
            out[eid] = bd
            continue

        exact = BeamDiagramData(eid=eid)
        for st in bd.stations:
            sig = beam_stress_at_station(
                st.bm1, st.bm2, st.af, A, I1, I2, I12,
                pts['C'][0], pts['C'][1],
                pts['D'][0], pts['D'][1],
                pts['E'][0], pts['E'][1],
                pts['F'][0], pts['F'][1],
            )
            new_st = BeamStationData(
                eid=st.eid, nid=st.nid, sd=st.sd,
                bm1=st.bm1, bm2=st.bm2, ts1=st.ts1, ts2=st.ts2,
                af=st.af, ttrq=st.ttrq, wtrq=st.wtrq,
                sxc=float(sig['C']), sxd=float(sig['D']),
                sxe=float(sig['E']), sxf=float(sig['F']),
            )
            vals = [new_st.sxc, new_st.sxd, new_st.sxe, new_st.sxf]
            new_st.smax = max(vals)
            new_st.smin = min(vals)
            exact.stations.append(new_st)
        out[eid] = exact
    return out


def _transverse_pload_components(model, elem, load_sid: int, xs: np.ndarray,
                                 ex: np.ndarray, ey: np.ndarray, ez: np.ndarray):
    qy = np.zeros_like(xs, dtype=np.float64)
    qz = np.zeros_like(xs, dtype=np.float64)
    if model is None or elem is None or len(getattr(elem, 'nodes', [])) < 2:
        return qy, qz
    n0 = model.nodes.get(elem.nodes[0]); n1 = model.nodes.get(elem.nodes[1])
    if n0 is None or n1 is None:
        return qy, qz
    L = float(np.linalg.norm(n1.xyz - n0.xyz))
    if L <= 1e-12:
        return qy, qz

    type_global = {
        'FY': np.array([0.0, 1.0, 0.0], dtype=np.float64),
        'FZ': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    type_local = {
        'FXE': np.array([1.0, 0.0, 0.0], dtype=np.float64),
        'FYE': np.array([0.0, 1.0, 0.0], dtype=np.float64),
        'FZE': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }

    local_y = np.asarray(ey, dtype=np.float64)
    local_z = np.asarray(ez, dtype=np.float64)

    for pl in getattr(model, 'pload1s', []):
        if int(pl.get('eid', 0)) != int(elem.id):
            continue
        if load_sid > 0 and int(pl.get('sid', 0)) != load_sid:
            continue
        typ = str(pl.get('type', '')).strip().upper()
        if typ in type_local:
            comp_y = float(type_local[typ][1])
            comp_z = float(type_local[typ][2])
        elif typ in type_global:
            gv = type_global[typ]
            comp_y = float(np.dot(gv, local_y))
            comp_z = float(np.dot(gv, local_z))
        else:
            continue

        a, b, p1, p2 = _pload1_span(pl, L)
        if abs(b - a) <= 1e-12:
            continue
        span = b - a
        for i, x in enumerate(xs):
            if x < a or x > b:
                continue
            t = (x - a) / span
            p = float(p1 + (p2 - p1) * t)
            qy[i] += comp_y * p
            qz[i] += comp_z * p
    return qy, qz


def _concentrated_pload_components(model, elem, load_sid: int, length: float,
                                   ex: np.ndarray, ey: np.ndarray, ez: np.ndarray):
    loads_y = []
    loads_z = []
    if model is None or elem is None or length <= 1e-12:
        return loads_y, loads_z

    type_global = {
        'FY': np.array([0.0, 1.0, 0.0], dtype=np.float64),
        'FZ': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    type_local = {
        'FXE': np.array([1.0, 0.0, 0.0], dtype=np.float64),
        'FYE': np.array([0.0, 1.0, 0.0], dtype=np.float64),
        'FZE': np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    local_y = np.asarray(ey, dtype=np.float64)
    local_z = np.asarray(ez, dtype=np.float64)

    for pl in getattr(model, 'pload1s', []):
        if int(pl.get('eid', 0)) != int(elem.id):
            continue
        if load_sid > 0 and int(pl.get('sid', 0)) != load_sid:
            continue
        a, b, p1, p2 = _pload1_span(pl, length)
        if abs(b - a) > 1e-10 * max(length, 1.0):
            continue
        typ = str(pl.get('type', '')).strip().upper()
        if typ in type_local:
            comp_y = float(type_local[typ][1])
            comp_z = float(type_local[typ][2])
        elif typ in type_global:
            gv = type_global[typ]
            comp_y = float(np.dot(gv, local_y))
            comp_z = float(np.dot(gv, local_z))
        else:
            continue
        pos = 0.5 * (a + b)
        if abs(comp_y) > 1e-12:
            loads_y.append((pos, comp_y * float(p1)))
        if abs(comp_z) > 1e-12:
            loads_z.append((pos, comp_z * float(p1)))
    return loads_y, loads_z


def _beam_end_force_components(beam_end_data, eid: int):
    if not beam_end_data:
        return None
    bd = beam_end_data.get(int(eid))
    if bd is None or not getattr(bd, 'stations', None):
        return None
    st0 = bd.stations[0]
    return {
        'bm1_A': float(getattr(st0, 'bm1', 0.0)),
        'bm2_A': float(getattr(st0, 'bm2', 0.0)),
        'ts1_A': float(getattr(st0, 'ts1', 0.0)),
        'ts2_A': float(getattr(st0, 'ts2', 0.0)),
    }


def _double_integrate_point_loads(xs: np.ndarray, u0: float, u1: float,
                                  bmA: float, tsA: float, E: float, I: float,
                                  point_loads, G: float = 0.0, A: float = 0.0,
                                  kappa: float = 5.0 / 6.0):
    if len(xs) < 2 or E <= 1e-12 or I <= 1e-16:
        return None
    EI = E * I
    L = float(xs[-1] - xs[0])
    if L <= 1e-12:
        return None

    V_A = -float(tsA)
    M_A = float(bmA)
    conc_loads = [(float(a), float(p)) for a, p in (point_loads or [])]

    def _EI_w_raw(x):
        val = M_A * x * x / 2.0 + V_A * x * x * x / 6.0
        for a_pos, pval in conc_loads:
            if x > a_pos:
                val += pval * (x - a_pos) ** 3 / 6.0
        return val

    C2 = EI * float(u0)
    C1 = (EI * float(u1) - C2 - _EI_w_raw(L)) / L
    w = np.array([(_EI_w_raw(float(x)) + C1 * float(x) + C2) / EI for x in xs], dtype=np.float64)

    kGA = kappa * G * A
    if kGA > 1e-12:
        w_sh = np.zeros_like(w)
        for i in range(1, len(xs)):
            s = 0.5 * (xs[i - 1] + xs[i])
            dx = xs[i] - xs[i - 1]
            Vs = V_A
            for a_pos, pval in conc_loads:
                if s > a_pos:
                    Vs += pval
            w_sh[i] = w_sh[i - 1] + (Vs / kGA) * dx
        w -= w_sh
    return w.astype(np.float32)


def _timoshenko_transverse_from_loads(xs: np.ndarray, q: np.ndarray,
                                      u0: float, th0: float, u1: float, th1: float,
                                      E: float, I: float, G: float, A: float, kappa: float):
    n = len(xs)
    if n < 2 or E <= 1e-12 or I <= 1e-16:
        return None
    EI = E * I
    kGA = max(kappa * G * A, 1e-12)
    dx = np.diff(xs)
    Vp = np.zeros(n, dtype=np.float64)
    Mp = np.zeros(n, dtype=np.float64)
    tp = np.zeros(n, dtype=np.float64)
    wp = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        Vp[i] = Vp[i-1] - 0.5 * (q[i-1] + q[i]) * dx[i-1]
    for i in range(1, n):
        Mp[i] = Mp[i-1] + 0.5 * (Vp[i-1] + Vp[i]) * dx[i-1]
    for i in range(1, n):
        tp[i] = tp[i-1] + 0.5 * (Mp[i-1] + Mp[i]) * dx[i-1] / EI
    for i in range(1, n):
        f0 = tp[i-1] + Vp[i-1] / kGA
        f1 = tp[i] + Vp[i] / kGA
        wp[i] = wp[i-1] + 0.5 * (f0 + f1) * dx[i-1]

    L = float(xs[-1] - xs[0])
    a11 = L / EI
    a12 = (L * L) / (2.0 * EI)
    a21 = (L * L) / (2.0 * EI)
    a22 = (L * L * L) / (6.0 * EI) + L / kGA
    rhs1 = float(th1 - th0 - tp[-1])
    rhs2 = float(u1 - u0 - th0 * L - wp[-1])
    det = a11 * a22 - a12 * a21
    if abs(det) <= 1e-18:
        return None
    M0 = (rhs1 * a22 - rhs2 * a12) / det
    V0 = (a11 * rhs2 - a21 * rhs1) / det

    x = xs - xs[0]
    theta = th0 + M0 * (x / EI) + V0 * ((x * x) / (2.0 * EI)) + tp
    w = u0 + th0 * x + M0 * ((x * x) / (2.0 * EI)) + V0 * (((x * x * x) / (6.0 * EI)) + x / kGA) + wp
    return w.astype(np.float32)


def _hermite_beam_points(p0, p1, d0, d1, deform_scale: float, nsamp: int = 17, v_orient=None,
                         result_type: str = 'displacement', phi_yz: Optional[Tuple[float, float]] = None,
                         model=None, elem=None, load_sid: int = 0, beam_end_data=None):
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)
    ex, ey, ez = _beam_local_frame(p0, p1, v_orient)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-12:
        return np.array([p0, p1], dtype=np.float32), np.array([p0, p1], dtype=np.float32)

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
    tim_props = _beam_timoshenko_props(model, elem, L) if (model is not None and elem is not None) else None
    if tim_props is not None:
        E, G, A, Iy, Iz, kappa = tim_props
        qy, qz = _transverse_pload_components(model, elem, load_sid, xs, ex, ey, ez)
        py, pz = _concentrated_pload_components(model, elem, load_sid, L, ex, ey, ez)
        ef = _beam_end_force_components(beam_end_data, getattr(elem, 'id', 0))
        vy_exact = _timoshenko_transverse_from_loads(xs, qy, u0y, s0y, u1y, s1y, E, Iz, G, A, kappa) if np.max(np.abs(qy)) > 1e-12 else None
        vz_exact = _timoshenko_transverse_from_loads(xs, qz, u0z, s0z, u1z, s1z, E, Iy, G, A, kappa) if np.max(np.abs(qz)) > 1e-12 else None
        if vy_exact is None and ef is not None and py:
            vy_exact = _double_integrate_point_loads(xs, u0y, u1y, ef.get('bm2_A', 0.0), ef.get('ts2_A', 0.0), E, Iz, py, G, A, kappa)
        if vz_exact is None and ef is not None and pz:
            vz_exact = _double_integrate_point_loads(xs, u0z, u1z, ef.get('bm1_A', 0.0), ef.get('ts1_A', 0.0), E, Iy, pz, G, A, kappa)
    else:
        vy_exact = None
        vz_exact = None
    base_pts = []
    def_pts = []
    for i, s in enumerate(svals):
        axial = (1.0 - s) * u0x + s * u1x
        vy = float(vy_exact[i]) if vy_exact is not None else _timoshenko_interp(float(s), u0y, s0y, u1y, s1y, L, phi_y)
        vz = float(vz_exact[i]) if vz_exact is not None else _timoshenko_interp(float(s), u0z, s0z, u1z, s1z, L, phi_z)
        base = p0 + ex * (L * float(s))
        disp = ex * axial + ey * vy + ez * vz
        base_pts.append(base)
        def_pts.append(base + disp)
    return np.array(base_pts, dtype=np.float32), np.array(def_pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3D diagram geometry builder
# ---------------------------------------------------------------------------

def build_beam_diagram_geometry(
    beam_data:  Dict[int, BeamDiagramData],
    model,                              # MystranModel for node positions
    result_key: str = 'bm1',            # which result to diagram
    scale:      float = 0.0,            # 0 = auto-scale
    show_fill:  bool = True,
    show_curve: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build OpenGL line geometry for beam diagrams.

    Returns:
        line_verts:  (N, 3) float32 — diagram outline + baselines
        line_colors: (N, 3) float32 — color per vertex
        fill_verts:  (M, 3) float32 — filled triangles
        fill_colors: (M, 3) float32
    """
    # Auto-scale: diagram height = fraction of typical beam length
    if scale <= 0:
        # Find max value across all beams
        all_vals = []
        for bd in beam_data.values():
            all_vals.extend(np.abs(bd.values(result_key)).tolist())
        max_val = max(all_vals) if all_vals else 1.0
        if max_val < 1e-30:
            return (np.zeros((0,3),dtype=np.float32),)*4

        # Target: diagram height = 50% of avg beam length for clearer moment shape
        avg_len = _avg_beam_length(beam_data, model)
        scale = (avg_len * 0.50) / max_val if max_val > 0 else 1.0

    # Colors
    C_POS  = np.array([0.3, 0.8, 1.0], dtype=np.float32)   # positive: cyan
    C_NEG  = np.array([1.0, 0.5, 0.2], dtype=np.float32)   # negative: orange
    C_BASE = np.array([0.7, 0.7, 0.7], dtype=np.float32)   # baseline: gray

    line_v = []; line_c = []
    fill_v = []; fill_c = []

    for eid, bd in beam_data.items():
        if len(bd.stations) < 2:
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
        L    = np.linalg.norm(axis)
        if L < 1e-9: continue
        ex = (axis / L).astype(np.float32)

        # Local perpendicular for diagram plane
        # Use element local y-axis (from v_orient if available)
        vo = getattr(elem, 'v_orient', None)
        if vo is not None and np.linalg.norm(vo) > 1e-9:
            v = (vo / np.linalg.norm(vo)).astype(np.float32)
        else:
            v = np.array([0,0,1],dtype=np.float32) if abs(ex[2])<0.9 \
                else np.array([0,1,0],dtype=np.float32)
        ez = v - np.dot(v,ex)*ex
        nez = np.linalg.norm(ez)
        if nez < 1e-12: continue
        ey = (ez/nez).astype(np.float32)  # diagram direction (local y)

        # Station points along beam
        sds   = bd.sd_arr
        vals  = bd.values(result_key)
        pts   = np.array([p0 + ex*L*sd for sd in sds], dtype=np.float32)
        disp  = np.array([ey * v * scale for v in vals], dtype=np.float32)
        tips  = pts + disp

        # Baseline: line along beam
        for i in range(len(pts)-1):
            line_v += [pts[i], pts[i+1]]
            line_c += [C_BASE, C_BASE]

        # Diagram outline (the curve)
        for i in range(len(tips)-1):
            col = C_POS if vals[i] >= 0 else C_NEG
            line_v += [tips[i], tips[i+1]]
            line_c += [col, col]

        # Sparse divider lines from baseline to tip. Dense exact sampling in v2
        # is useful for the curve itself, but not for drawing every single stripe.
        vert_idx = _diagram_vertical_indices(vals)
        for i in vert_idx:
            col = C_POS if vals[i] >= 0 else C_NEG
            line_v += [pts[i], tips[i]]
            line_c += [col, col]

        # Fill triangles (baseline to curve). Split at zero crossing so
        # positive/negative regions do not create crossed fill wedges.
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            ta, tb = tips[i], tips[i+1]
            va = float(vals[i])
            vb = float(vals[i+1])
            same_sign = (
                va == 0.0 or vb == 0.0 or
                (va > 0.0 and vb >= 0.0) or
                (va < 0.0 and vb <= 0.0)
            )
            if same_sign:
                col = C_POS if (va + vb) >= 0.0 else C_NEG
                fill_v += [a, b, tb, a, tb, ta]
                fill_c += [col] * 6
            else:
                t = abs(va) / (abs(va) + abs(vb))
                mid = a + (b - a) * t
                fill_v += [a, mid, ta]
                fill_c += ([C_POS] * 3) if va > 0.0 else ([C_NEG] * 3)
                fill_v += [mid, b, tb]
                fill_c += ([C_POS] * 3) if vb > 0.0 else ([C_NEG] * 3)

    def arr(lst):
        return np.array(lst, dtype=np.float32).reshape(-1,3) if lst \
               else np.zeros((0,3),dtype=np.float32)

    return arr(line_v), arr(line_c), arr(fill_v), arr(fill_c)


def _diagram_vertical_indices(vals: np.ndarray) -> List[int]:
    n = len(vals)
    if n <= 0:
        return []
    if n <= 10:
        return list(range(n))

    idx = {0, n - 1}

    # Local extrema
    for i in range(1, n - 1):
        a = float(vals[i - 1])
        b = float(vals[i])
        c = float(vals[i + 1])
        if (b >= a and b >= c) or (b <= a and b <= c):
            idx.add(i)

    # Sign changes / near-zero crossings
    for i in range(n - 1):
        a = float(vals[i])
        b = float(vals[i + 1])
        if a == 0.0 or b == 0.0 or (a < 0.0 < b) or (b < 0.0 < a):
            idx.add(i)
            idx.add(i + 1)

    # If the curve is monotone and has no sign change, keep only a few
    # structural dividers instead of dense stripes.
    if len(idx) <= 2:
        idx.update({max(1, n // 3), max(1, (2 * n) // 3)})
    elif len(idx) <= 4:
        idx.add(max(1, n // 2))

    return sorted(i for i in idx if 0 <= i < n)


def _avg_beam_length(beam_data, model) -> float:
    lengths = []
    for eid in beam_data:
        elem = model.elements.get(eid)
        if elem and len(elem.nodes) >= 2:
            n0 = model.nodes.get(elem.nodes[0])
            n1 = model.nodes.get(elem.nodes[1])
            if n0 and n1:
                lengths.append(float(np.linalg.norm(n1.xyz - n0.xyz)))
    return float(np.mean(lengths)) if lengths else 1.0


def _pload1_span(pl, length: float) -> Tuple[float, float, float, float]:
    use_fraction = str(pl.get('scale', '')).strip().upper().startswith('FR')
    x1 = float(pl.get('x1', 0.0))
    x2 = float(pl.get('x2', x1))
    p1 = float(pl.get('p1', 0.0))
    p2 = float(pl.get('p2', p1))
    if use_fraction:
        a = float(np.clip(x1, 0.0, 1.0)) * length
        b = float(np.clip(x2, 0.0, 1.0)) * length
    else:
        a = float(np.clip(x1, 0.0, length))
        b = float(np.clip(x2, 0.0, length))
    if a <= b:
        return a, b, p1, p2
    return b, a, p2, p1


def _q_and_k_at_x(x: float, a: float, b: float, pa: float, pb: float) -> Tuple[float, float]:
    if x <= a:
        return 0.0, 0.0
    if abs(b - a) <= 1e-12:
        p = pa if abs(pa) >= abs(pb) else pb
        dx = x - a
        if dx <= 0.0:
            return 0.0, 0.0
        return p, p * dx
    Ls = b - a
    k = (pb - pa) / Ls
    if x < b:
        t = x - a
        q = pa * t + 0.5 * k * t * t
        kk = 0.5 * pa * t * t + (k * t * t * t) / 6.0
        return q, kk
    q_tot = pa * Ls + 0.5 * k * Ls * Ls
    base_m = 0.5 * pa * Ls * Ls + (k * Ls * Ls * Ls) / 3.0
    kk = q_tot * (x - a) - base_m
    return q_tot, kk


def build_exact_beam_data(
    beam_data: Dict[int, BeamDiagramData],
    model,
    result_key: str = 'bm1',
    load_sid: int = 0,
    nsamp: int = 161,
) -> Dict[int, BeamDiagramData]:
    if not beam_data:
        return {}
    if result_key in _BEAM_STRESS_KEYS:
        return _exact_beam_stress_data(beam_data, model)
    pair_map = {
        'bm1': ('bm1', 'ts1'),
        'ts1': ('bm1', 'ts1'),
        'bm2': ('bm2', 'ts2'),
        'ts2': ('bm2', 'ts2'),
        'af':  ('af',  'af'),
        'ttrq':('ttrq','ttrq'),
    }
    if result_key not in pair_map:
        return beam_data

    moment_key, force_key = pair_map[result_key]
    out: Dict[int, BeamDiagramData] = {}

    for eid, bd in beam_data.items():
        if len(bd.stations) < 2:
            out[eid] = bd
            continue
        elem = model.elements.get(eid)
        if elem is None or len(elem.nodes) < 2:
            out[eid] = bd
            continue
        n0 = model.nodes.get(elem.nodes[0])
        n1 = model.nodes.get(elem.nodes[1])
        if n0 is None or n1 is None:
            out[eid] = bd
            continue

        p0 = n0.xyz.astype(np.float32)
        p1 = n1.xyz.astype(np.float32)
        L = float(np.linalg.norm(p1 - p0))
        if L <= 1e-12:
            out[eid] = bd
            continue
        ex, ey, ez = _beam_local_frame(p0, p1, getattr(elem, 'v_orient', None))

        first = bd.stations[0]
        last = bd.stations[-1]
        f0 = float(getattr(first, force_key, 0.0))
        f1 = float(getattr(last, force_key, 0.0))
        m0 = float(getattr(first, moment_key, 0.0))
        m1 = float(getattr(last, moment_key, 0.0))

        beam_loads = []
        for pl in getattr(model, 'pload1s', []):
            if int(pl.get('eid', 0)) != eid:
                continue
            if load_sid and int(pl.get('sid', 0)) != load_sid:
                continue
            comp = _pload1_result_component(pl, ex, ey, ez, result_key)
            if comp is None or abs(comp) <= 1e-12:
                continue
            beam_loads.append((pl, comp))

        if not beam_loads:
            out[eid] = _reduced_station_data(bd, result_key)
            continue

        q_end = 0.0
        k_end = 0.0
        for pl, comp in beam_loads:
            a, b, pa, pb = _pload1_span(pl, L)
            q_i, k_i = _q_and_k_at_x(L, a, b, comp * pa, comp * pb)
            q_end += q_i
            k_end += k_i
        if result_key in ('bm1', 'bm2', 'ts1', 'ts2'):
            score_pos = abs((f0 + q_end) - f1) + abs((m0 - f0 * L - k_end) - m1)
            score_neg = abs((f0 - q_end) - f1) + abs((m0 - f0 * L + k_end) - m1)
            sgn = 1.0 if score_pos <= score_neg else -1.0
        else:
            score_pos = abs((f0 + q_end) - f1)
            score_neg = abs((f0 - q_end) - f1)
            sgn = 1.0 if score_pos <= score_neg else -1.0

        all_point = True
        x_samples = [0.0, L]
        for pl, _comp in beam_loads:
            a, b, _, _ = _pload1_span(pl, L)
            x_samples.extend([a, b])
            if abs(b - a) > 1e-12:
                all_point = False

        if not all_point:
            x_samples.extend(np.linspace(0.0, L, max(5, int(nsamp)), dtype=np.float64).tolist())
        elif result_key in ('ts1', 'ts2', 'af', 'ttrq'):
            # For point-load/shear-like plots, keep a tiny split around the jump
            # so the step is visible without flooding the beam with samples.
            eps = max(L * 1e-6, 1e-6)
            extra = []
            for pl, _comp in beam_loads:
                a, b, _, _ = _pload1_span(pl, L)
                x0 = 0.5 * (a + b)
                extra.extend([max(0.0, x0 - eps), min(L, x0 + eps)])
            x_samples.extend(extra)
        x_samples = sorted(set(round(float(x), 9) for x in x_samples))

        exact = BeamDiagramData(eid=eid)
        for x in x_samples:
            qx = 0.0
            kx = 0.0
            for pl, comp in beam_loads:
                a, b, pa, pb = _pload1_span(pl, L)
                qi, ki = _q_and_k_at_x(float(x), a, b, comp * pa, comp * pb)
                qx += qi
                kx += ki

            sval = _empty_station(eid, float(x / L))
            if result_key in ('bm1', 'bm2', 'ts1', 'ts2'):
                force_val = f0 + sgn * qx
                moment_val = m0 - f0 * float(x) - sgn * kx
                setattr(sval, force_key, force_val)
                setattr(sval, moment_key, moment_val)
            else:
                force_val = f0 + sgn * qx
                setattr(sval, force_key, force_val)
            exact.stations.append(sval)
        out[eid] = exact

    return out


def auto_beam_scale(beam_data, model, result_key: str) -> float:
    all_vals = []
    for bd in beam_data.values():
        all_vals.extend(np.abs(bd.values(result_key)).tolist())
    max_val = max(all_vals) if all_vals else 1.0
    if max_val < 1e-30:
        return 1.0
    avg_len = _avg_beam_length(beam_data, model)
    return (avg_len * 0.50) / max_val if max_val > 0 else 1.0


def beam_diagram_bounds(
    beam_data: Dict[int, BeamDiagramData],
    model,
    result_key: str = 'bm1',
    scale: float = 0.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not beam_data:
        return None
    if scale <= 0:
        scale = auto_beam_scale(beam_data, model, result_key)
    lv, _, fv, _ = build_beam_diagram_geometry(beam_data, model, result_key, scale)
    pts = []
    if len(lv):
        pts.append(lv)
    if len(fv):
        pts.append(fv)
    if not pts:
        return None
    arr = np.vstack(pts)
    return arr.min(axis=0), arr.max(axis=0)


# ---------------------------------------------------------------------------
# Deformed beam shape from interstation displacements
# ---------------------------------------------------------------------------

def build_beam_deformed_shape(
    beam_data:  Dict[int, BeamDiagramData],
    model,
    disp_results,           # F06Results.displacements[subcase]
    deform_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build smooth deformed beam shape from end-node beam DOFs.
    Uses cubic Hermitian interpolation in the transverse directions.
    Returns (line_verts, line_colors).
    """
    C_GHOST  = np.array([0.45, 0.55, 0.70], dtype=np.float32)
    C_DEFORM = np.array([0.3, 1.0, 0.5], dtype=np.float32)

    line_v = []; line_c = []

    for eid, bd in beam_data.items():
        elem = model.elements.get(eid)
        if elem is None or len(elem.nodes) < 2: continue
        n0 = model.nodes.get(elem.nodes[0])
        n1 = model.nodes.get(elem.nodes[1])
        if n0 is None or n1 is None: continue

        p0 = n0.xyz.astype(np.float32)
        p1 = n1.xyz.astype(np.float32)
        d0 = disp_results.get(elem.nodes[0])
        d1 = disp_results.get(elem.nodes[1])
        if d0 is None or d1 is None: continue
        nsamp = max(17, len(getattr(bd, 'stations', []) or []) * 2 + 1)
        base_pts, deformed_pts = _hermite_beam_points(
            p0, p1, d0, d1, deform_scale,
            nsamp=nsamp,
            v_orient=getattr(elem, 'v_orient', None))

        for i in range(len(base_pts)-1):
            line_v += [base_pts[i], base_pts[i+1]]
            line_c += [C_GHOST, C_GHOST]

        for i in range(len(deformed_pts)-1):
            line_v += [deformed_pts[i], deformed_pts[i+1]]
            line_c += [C_DEFORM, C_DEFORM]

    def arr(lst):
        return np.array(lst, dtype=np.float32).reshape(-1,3) if lst \
               else np.zeros((0,3),dtype=np.float32)

    return arr(line_v), arr(line_c)


# ---------------------------------------------------------------------------
# OpenGL VAO renderer (integrates with existing MeshRenderer pattern)
# ---------------------------------------------------------------------------

class BeamDiagramRenderer:
    """
    Renders beam internal force diagrams using OpenGL lines + alpha tris.
    Uses the same shaders as MeshRenderer (passed in as refs).
    """
    def __init__(self, ctx, prog_line, prog_fill):
        self.ctx       = ctx
        self._prog_line = prog_line
        self._prog_fill = prog_fill
        self._vaos     = {}   # key -> (vao, vbo, count, mode)
        self.beam_data : Dict[int, BeamDiagramData] = {}
        self.active_data: Dict[int, BeamDiagramData] = {}
        self.beam_stress_data: Dict[int, List[BeamStationData]] = {}
        self.active_key = 'bm1'
        self.scale      = 0.0
        self.alpha      = 0.45
        self.mode       = 'station'
        self.beam_stress_available = False

    def load_op2(self, filepath: str, subcase: int = 1):
        self.beam_data = read_beam_forces_op2(filepath, subcase)
        try:
            self.beam_stress_data = read_beam_stress_op2(filepath, subcase)
        except Exception:
            self.beam_stress_data = {}
        if self.beam_stress_data:
            for eid, bd in self.beam_data.items():
                sts = self.beam_stress_data.get(eid)
                if not sts:
                    continue
                by_sd = {round(float(st.sd), 6): st for st in sts}
                for st in bd.stations:
                    src = by_sd.get(round(float(st.sd), 6))
                    if src is None:
                        continue
                    st.sxc = src.sxc; st.sxd = src.sxd; st.sxe = src.sxe; st.sxf = src.sxf
                    st.smax = src.smax; st.smin = src.smin
        try:
            self.beam_stress_available = beam_stress_available_op2(filepath, subcase)
        except Exception:
            self.beam_stress_available = False
        self.active_data = dict(self.beam_data)
        self._vaos.clear()
        print(f"[BeamDiagram] Loaded {len(self.beam_data)} beam elements")
        return len(self.beam_data) > 0

    def upload(self, model, result_key='bm1', scale=0.0, alpha=0.45):
        if not self.beam_data: return
        self.active_key = result_key
        self.scale      = scale
        self.alpha      = alpha
        self.mode       = 'station'
        self.active_data = dict(self.beam_data)
        import moderngl

        lv, lc, fv, fc = build_beam_diagram_geometry(
            self.active_data, model, result_key, scale)

        self._vaos.clear()

        if len(lv) > 0:
            import numpy as np
            n   = len(lv)
            buf = np.empty((n,6), dtype=np.float32)
            buf[:,0:3] = lv; buf[:,3:6] = lc
            vbo = self.ctx.buffer(buf.flatten().tobytes())
            vao = self.ctx.vertex_array(self._prog_line,
                      [(vbo,'3f 3f','in_position','in_color')])
            self._vaos['lines'] = (vao, vbo, n)

        if len(fv) > 0:
            import numpy as np
            n   = len(fv)
            nrm = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
            buf = np.empty((n,9), dtype=np.float32)
            buf[:,0:3] = fv; buf[:,3:6] = fc; buf[:,6:9] = nrm
            vbo = self.ctx.buffer(buf.flatten().tobytes())
            vao = self.ctx.vertex_array(self._prog_fill,
                      [(vbo,'3f 3f 3f','in_position','in_color','in_normal')])
            self._vaos['fill'] = (vao, vbo, n)

    def upload_exact(self, model, result_key='bm1', scale=0.0, alpha=0.45, load_sid: int = 0):
        if not self.beam_data:
            return
        self.active_key = result_key
        self.scale = scale
        self.alpha = alpha
        self.mode = 'exact'
        self.active_data = build_exact_beam_data(
            self.beam_data, model, result_key=result_key, load_sid=load_sid)
        import moderngl

        lv, lc, fv, fc = build_beam_diagram_geometry(
            self.active_data, model, result_key, scale)

        self._vaos.clear()

        if len(lv) > 0:
            import numpy as np
            n   = len(lv)
            buf = np.empty((n,6), dtype=np.float32)
            buf[:,0:3] = lv; buf[:,3:6] = lc
            vbo = self.ctx.buffer(buf.flatten().tobytes())
            vao = self.ctx.vertex_array(self._prog_line,
                      [(vbo,'3f 3f','in_position','in_color')])
            self._vaos['lines'] = (vao, vbo, n)

        if len(fv) > 0:
            import numpy as np
            n   = len(fv)
            nrm = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
            buf = np.empty((n,9), dtype=np.float32)
            buf[:,0:3] = fv; buf[:,3:6] = fc; buf[:,6:9] = nrm
            vbo = self.ctx.buffer(buf.flatten().tobytes())
            vao = self.ctx.vertex_array(self._prog_fill,
                      [(vbo,'3f 3f 3f','in_position','in_color','in_normal')])
            self._vaos['fill'] = (vao, vbo, n)

    def upload_deformed(self, model, disp_results, deform_scale=1.0):
        data = self.active_data if self.active_data else self.beam_data
        if not data: return
        import moderngl, numpy as np
        lv, lc = build_beam_deformed_shape(
            data, model, disp_results, deform_scale)
        if len(lv) > 0:
            n = len(lv)
            buf = np.empty((n,6), dtype=np.float32)
            buf[:,0:3]=lv; buf[:,3:6]=lc
            vbo = self.ctx.buffer(buf.flatten().tobytes())
            vao = self.ctx.vertex_array(self._prog_line,
                      [(vbo,'3f 3f','in_position','in_color')])
            self._vaos['deform'] = (vao, vbo, n)

    def diagram_data(self) -> Dict[int, BeamDiagramData]:
        return self.active_data if self.active_data else self.beam_data

    def draw(self, mvp_bytes, alpha=None):
        import moderngl
        if not self._vaos: return
        a = alpha if alpha is not None else self.alpha

        # Lines (baseline + outline + verticals)
        if 'lines' in self._vaos:
            vao,_,cnt = self._vaos['lines']
            self._prog_line['u_mvp'].write(mvp_bytes)
            self.ctx.line_width = 1.5
            vao.render(moderngl.LINES)
            self.ctx.line_width = 1.0

        # Fill (alpha blend)
        if 'fill' in self._vaos:
            vao,_,cnt = self._vaos['fill']
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._prog_fill['u_mvp'].write(mvp_bytes)
            self._prog_fill['u_alpha'].value = a
            vao.render(moderngl.TRIANGLES)
            self.ctx.disable(moderngl.BLEND)

        # Deformed shape
        if 'deform' in self._vaos:
            vao,_,cnt = self._vaos['deform']
            self._prog_line['u_mvp'].write(mvp_bytes)
            self.ctx.line_width = 2.5
            vao.render(moderngl.LINES)
            self.ctx.line_width = 1.0

    def available_results(self) -> List[Tuple[str,str]]:
        items = [
            ('bm1',  'Bending Moment 1 (M1)'),
            ('bm2',  'Bending Moment 2 (M2)'),
            ('ts1',  'Shear Force 1 (V1)'),
            ('ts2',  'Shear Force 2 (V2)'),
            ('af',   'Axial Force (N)'),
            ('ttrq', 'Total Torque (T)'),
        ]
        if self.beam_stress_available:
            items.extend([
                ('sxc',  'Stress C'),
                ('sxd',  'Stress D'),
                ('sxe',  'Stress E'),
                ('sxf',  'Stress F'),
                ('smax', 'Stress Smax'),
                ('smin', 'Stress Smin'),
            ])
        return items

    def max_value(self, key: str) -> float:
        data = self.diagram_data()
        if not data: return 0.0
        vals = []
        for bd in data.values():
            vals.extend(np.abs(bd.values(key)).tolist())
        return max(vals) if vals else 0.0
