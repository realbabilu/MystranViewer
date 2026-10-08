"""
MYSTRAN F06 Parser — whitespace-split based.
Covers: CQUAD4, CQUAD8, CTRIA3, CTRIA6
  - Displacements (T1,T2,T3,R1,R2,R3)
  - Grid-point stresses (surface system, Z1/Z2/MID fibers)
  - Element stresses CENTER + CORNER (local coord system)
  - Element engineering forces CENTER + CORNER (Nxx,Nyy,Nxy,Mxx,Myy,Mxy,Qx)
  - Grid-point forces (membrane+bending)
"""
from __future__ import annotations
import re, sys
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DisplacementRow:
    subcase: int
    grid_id: int
    coord_sys: int
    t1: float; t2: float; t3: float
    r1: float; r2: float; r3: float

@dataclass
class GridStressRow:
    subcase: int; surface_id: int
    grid_id: int; elem_id: int; fiber: str
    nx: float; ny: float; sxy: float
    angle: float; major: float; minor: float; shear: float; von_mises: float

@dataclass
class ElemStressRow:
    subcase: int; elem_type: str; elem_id: int
    location: str; fiber_dist: float
    nx: float; ny: float; sxy: float
    angle: float; major: float; minor: float; von_mises: float
    sxz: float = 0.0; syz: float = 0.0

@dataclass
class ElemForceRow:
    subcase: int; elem_type: str; elem_id: int
    location: str; grid_id: int  # 0 for CENTER
    nxx: float; nyy: float; nxy: float
    mxx: float; myy: float; mxy: float
    qx: float; qy: float = 0.0

@dataclass
class GridForceRow:
    subcase: int; surface_id: int
    grid_id: int; elem_id: int
    nxx: float; nyy: float; nxy: float
    mxx: float; myy: float; mxy: float
    qx: float; qy: float

@dataclass
class F06Result:
    title: str = ""
    displacements: list[DisplacementRow] = field(default_factory=list)
    grid_stresses: list[GridStressRow] = field(default_factory=list)
    elem_stresses: list[ElemStressRow] = field(default_factory=list)
    elem_forces: list[ElemForceRow] = field(default_factory=list)
    grid_forces: list[GridForceRow] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

RE_OUTPUT_SC = re.compile(r'^\s*OUTPUT FOR SUBCASE\s+(\d+)\s*$')
RE_SURFACE_HDR = re.compile(r'S T R E S S E S   A T   G R I D   P O I N T S.*S U R F A C E\s+(\d+)')
RE_FORCE_HDR = re.compile(r'F O R C E S   A T   G R I D   P O I N T S.*S U R F A C E\s+(\d+)')
RE_ELEM_STRESS_L1 = re.compile(r'E L E M E N T\s+S T R E S S E S\s+I N\s+L O C A L')
RE_ELEM_FORCE_L1 = re.compile(r'E L E M E N T\s+E N G I N E E R I N G\s+F O R C E S')
# Element type line: "F O R   E L E M E N T   T Y P E   Q U A D" on one line,
# "4" (or 8/3/6) on the next physical line.
RE_ELEM_TYPE_L2 = re.compile(r'F O R\s+E L E M E N T\s+T Y P E\s+(.+?)\s*$')
RE_SUMMARY = re.compile(r'^\s*(MAX|MIN|ABS)\s*\*')
RE_SEPARATOR = re.compile(r'^\s*-{3,}')
RE_BLANK = re.compile(r'^\s*$')

def _f(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s.replace('D', 'E').replace('d', 'e'))
    except ValueError:
        return 0.0

def _norm_elem_type(s: str) -> str:
    s = s.upper().replace(' ', '').strip()
    m = {'QUAD4': 'CQUAD4', 'QUADR': 'CQUAD4', 'QUAD8': 'CQUAD8', 'TRIA3': 'CTRIA3', 'TRIAR': 'CTRIA3', 'TRIA6': 'CTRIA6'}
    return m.get(s, s)

# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_f06(path: str) -> F06Result:
    with open(path, 'r', encoding='latin-1') as fh:
        lines = fh.readlines()

    result = F06Result()
    state = 'top'
    cur_subcase: Optional[int] = None
    cur_surface: Optional[int] = None
    cur_elem_type: Optional[str] = None
    cur_elem_id: Optional[int] = None
    cur_location: str = 'CENTER'
    cur_grid: Optional[int] = None
    cur_gs_elem: Optional[int] = None
    pending_elem_header: Optional[str] = None

    for i, raw in enumerate(lines):
        line = raw.rstrip('\n\r')

        if RE_BLANK.match(line) or RE_SEPARATOR.match(line):
            continue

        # ── "OUTPUT FOR SUBCASE N"
        m = RE_OUTPUT_SC.match(line)
        if m:
            cur_subcase = int(m.group(1))
            state = 'disp'
            pending_elem_header = None
            continue

        # ── Surface header (grid-point stress)
        m = RE_SURFACE_HDR.search(line)
        if m:
            cur_surface = int(m.group(1))
            state = 'grid_stress'
            pending_elem_header = None
            continue

        # ── Surface header (grid-point force)
        m = RE_FORCE_HDR.search(line)
        if m:
            cur_surface = int(m.group(1))
            state = 'grid_force'
            pending_elem_header = None
            continue

        # ── Two-line element headers
        if RE_ELEM_STRESS_L1.search(line):
            pending_elem_header = 'stress'
            continue
        if RE_ELEM_FORCE_L1.search(line):
            pending_elem_header = 'force'
            continue

        # ── Element type line (may span 2 physical lines: "Q U A D" + "4")
        m = RE_ELEM_TYPE_L2.search(line)
        if m and pending_elem_header:
            raw_type = m.group(1).replace(' ', '')
            # Look ahead for digit on next line
            nxt = lines[i + 1].rstrip('\n\r').strip() if i + 1 < len(lines) else ''
            if nxt and nxt[0].isdigit():
                raw_type += nxt[0]
            cur_elem_type = _norm_elem_type(raw_type)
            state = 'elem_stress' if pending_elem_header == 'stress' else 'elem_force'
            pending_elem_header = None
            continue

        # ── Summary rows
        if RE_SUMMARY.match(line):
            continue

        parts = line.split()
        if not parts:
            continue

        # ── Displacement (8 fields: grid coord t1..r3)
        if state == 'disp' and cur_subcase is not None:
            if len(parts) == 8 and parts[0].isdigit() and parts[1].isdigit():
                result.displacements.append(DisplacementRow(
                    subcase=cur_subcase,
                    grid_id=int(parts[0]), coord_sys=int(parts[1]),
                    t1=_f(parts[2]), t2=_f(parts[3]), t3=_f(parts[4]),
                    r1=_f(parts[5]), r2=_f(parts[6]), r3=_f(parts[7])
                ))
            continue

        # ── Grid-point stress
        # Primary: 11 fields [grid, elem, fiber, nx, ny, sxy, angle, major, minor, shear, von]
        # Continuation: 9 fields [fiber, nx, ny, sxy, angle, major, minor, shear, von]
        if state == 'grid_stress' and cur_subcase is not None and cur_surface is not None:
            if (len(parts) == 11 and parts[0].isdigit() and parts[1].isdigit()
                    and parts[2] in ('Z1', 'Z2', 'MID')):
                cur_grid = int(parts[0])
                cur_gs_elem = int(parts[1])
                result.grid_stresses.append(GridStressRow(
                    subcase=cur_subcase, surface_id=cur_surface,
                    grid_id=cur_grid, elem_id=cur_gs_elem, fiber=parts[2],
                    nx=_f(parts[3]), ny=_f(parts[4]), sxy=_f(parts[5]),
                    angle=_f(parts[6]), major=_f(parts[7]), minor=_f(parts[8]),
                    shear=_f(parts[9]), von_mises=_f(parts[10])
                ))
            elif len(parts) == 9 and parts[0] in ('Z1', 'Z2', 'MID'):
                result.grid_stresses.append(GridStressRow(
                    subcase=cur_subcase, surface_id=cur_surface,
                    grid_id=cur_grid, elem_id=cur_gs_elem, fiber=parts[0],
                    nx=_f(parts[1]), ny=_f(parts[2]), sxy=_f(parts[3]),
                    angle=_f(parts[4]), major=_f(parts[5]), minor=_f(parts[6]),
                    shear=_f(parts[7]), von_mises=_f(parts[8])
                ))
            continue

        # ── Grid-point force (10 fields: grid elem nxx nyy nxy mxx myy mxy qx qy)
        if state == 'grid_force' and cur_subcase is not None and cur_surface is not None:
            if len(parts) == 10 and parts[0].isdigit() and parts[1].isdigit():
                result.grid_forces.append(GridForceRow(
                    subcase=cur_subcase, surface_id=cur_surface,
                    grid_id=int(parts[0]), elem_id=int(parts[1]),
                    nxx=_f(parts[2]), nyy=_f(parts[3]), nxy=_f(parts[4]),
                    mxx=_f(parts[5]), myy=_f(parts[6]), mxy=_f(parts[7]),
                    qx=_f(parts[8]), qy=_f(parts[9])
                ))
            continue

        # ── Element stress
        # CENTER primary (12 fields): [id, CENTER, dist, nx, ny, sxy, angle, maj, min, von, sxz, syz]
        # CENTER cont (8 fields):    [dist, nx, ny, sxy, angle, maj, min, von]  (QUAD)
        # CENTER cont (10 fields):   [dist, nx, ny, sxy, angle, maj, min, von, sxz, syz] (TRIA)
        # CORNER primary (11 fields): [id, GRD, grid_id, dist, nx, ny, sxy, angle, maj, min, von]
        # CORNER cont (8 fields):   same as CENTER cont
        if state == 'elem_stress' and cur_subcase is not None and cur_elem_type is not None:
            is_quad = cur_elem_type in ('CQUAD4', 'CQUAD8')
            is_tria = cur_elem_type in ('CTRIA3', 'CTRIA6')

            # CENTER primary
            if (len(parts) == 12 and parts[0].isdigit()
                    and parts[1] in ('CENTER', 'GAUSS')):
                cur_elem_id = int(parts[0])
                cur_location = parts[1]
                result.elem_stresses.append(ElemStressRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=cur_elem_id, location=cur_location,
                    fiber_dist=_f(parts[2]),
                    nx=_f(parts[3]), ny=_f(parts[4]), sxy=_f(parts[5]),
                    angle=_f(parts[6]), major=_f(parts[7]), minor=_f(parts[8]),
                    von_mises=_f(parts[9]),
                    sxz=_f(parts[10]), syz=_f(parts[11])
                ))
            # CORNER primary (11 fields: id, GRD, grid_id, dist, nx, ny, sxy, ang, maj, min, von)
            elif (len(parts) == 11 and parts[0].isdigit()
                    and parts[1] == 'GRD'):
                cur_elem_id = int(parts[0])
                cur_location = 'GRD'
                try:
                    corner_grid = int(parts[2])
                except (ValueError, IndexError):
                    corner_grid = 0
                result.elem_stresses.append(ElemStressRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=cur_elem_id, location=cur_location,
                    fiber_dist=_f(parts[3]),
                    nx=_f(parts[4]), ny=_f(parts[5]), sxy=_f(parts[6]),
                    angle=_f(parts[7]), major=_f(parts[8]), minor=_f(parts[9]),
                    von_mises=_f(parts[10])
                ))
            # QUAD continuation (8 fields)
            elif is_quad and len(parts) == 8 and cur_elem_id is not None:
                result.elem_stresses.append(ElemStressRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=cur_elem_id, location=cur_location,
                    fiber_dist=_f(parts[0]),
                    nx=_f(parts[1]), ny=_f(parts[2]), sxy=_f(parts[3]),
                    angle=_f(parts[4]), major=_f(parts[5]), minor=_f(parts[6]),
                    von_mises=_f(parts[7])
                ))
            # TRIA continuation (10 fields)
            elif is_tria and len(parts) == 10 and cur_elem_id is not None:
                result.elem_stresses.append(ElemStressRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=cur_elem_id, location=cur_location,
                    fiber_dist=_f(parts[0]),
                    nx=_f(parts[1]), ny=_f(parts[2]), sxy=_f(parts[3]),
                    angle=_f(parts[4]), major=_f(parts[5]), minor=_f(parts[6]),
                    von_mises=_f(parts[7]),
                    sxz=_f(parts[8]), syz=_f(parts[9])
                ))
            continue

        # ── Element engineering force
        # CENTER (9 fields): [id, nxx, nyy, nxy, mxx, myy, mxy, qx, qy]
        # CENTER (8 fields): [id, nxx, nyy, nxy, mxx, myy, mxy, qx]  (membrane only)
        # CORNER (10 fields): [id, GRD, grid_id, nxx, nyy, nxy, mxx, myy, mxy, qx]
        # Header rows start with 'Element', 'ID', 'Nxx' — skip them
        if state == 'elem_force' and cur_subcase is not None and cur_elem_type is not None:
            first = parts[0]
            n = len(parts)
            # QUAD8 / QUAD4 force CENTER: 10 fields [id, CENTER, nxx, nyy, nxy, mxx, myy, mxy, qx]
            if n == 10 and first.isdigit() and parts[1] == 'CENTER':
                result.elem_forces.append(ElemForceRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=int(parts[0]), location='CENTER', grid_id=0,
                    nxx=_f(parts[2]), nyy=_f(parts[3]), nxy=_f(parts[4]),
                    mxx=_f(parts[5]), myy=_f(parts[6]), mxy=_f(parts[7]),
                    qx=_f(parts[8])
                ))
            # QUAD8 / QUAD4 force CORNER: 10 fields [GRD, grid_id, nxx, nyy, nxy, mxx, myy, mxy, qx]
            # Only match when elem_type is QUAD
            elif (n == 10 and first == 'GRD' and cur_elem_type in ('CQUAD4', 'CQUAD8')):
                try:
                    gid = int(float(parts[2]))  # grid_id may be int or float
                except (ValueError, IndexError):
                    gid = 0
                result.elem_forces.append(ElemForceRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=int(parts[1]), location='GRD', grid_id=gid,
                    nxx=_f(parts[3]), nyy=_f(parts[4]), nxy=_f(parts[5]),
                    mxx=_f(parts[6]), myy=_f(parts[7]), mxy=_f(parts[8]),
                    qx=_f(parts[9])
                ))
            # CTRIA6 / CTRIA3 force CENTER (9 fields: id, nxx, nyy, nxy, mxx, myy, mxy, qx, qy)
            elif n == 9 and first.isdigit():
                result.elem_forces.append(ElemForceRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=int(parts[0]), location='CENTER', grid_id=0,
                    nxx=_f(parts[1]), nyy=_f(parts[2]), nxy=_f(parts[3]),
                    mxx=_f(parts[4]), myy=_f(parts[5]), mxy=_f(parts[6]),
                    qx=_f(parts[7]), qy=_f(parts[8])
                ))
            # CTRIA6 / CTRIA3 force CENTER without qy (8 fields, membrane only)
            elif n == 8 and first.isdigit():
                result.elem_forces.append(ElemForceRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=int(parts[0]), location='CENTER', grid_id=0,
                    nxx=_f(parts[1]), nyy=_f(parts[2]), nxy=_f(parts[3]),
                    mxx=_f(parts[4]), myy=_f(parts[5]), mxy=_f(parts[6]),
                    qx=_f(parts[7])
                ))
            # CTRIA6 / CTRIA3 force CORNER: 10 fields [id, GRD, grid_id, nxx, nyy, nxy, mxx, myy, mxy, qx]
            elif n == 10 and first == 'GRD' and parts[1].isdigit():
                try:
                    gid = int(float(parts[2]))  # grid_id
                except (ValueError, IndexError):
                    gid = 0
                result.elem_forces.append(ElemForceRow(
                    subcase=cur_subcase, elem_type=cur_elem_type,
                    elem_id=int(parts[1]), location='GRD', grid_id=gid,
                    nxx=_f(parts[3]), nyy=_f(parts[4]), nxy=_f(parts[5]),
                    mxx=_f(parts[6]), myy=_f(parts[7]), mxy=_f(parts[8]),
                    qx=_f(parts[9])
                ))
            # skip header rows
            elif first[0].isalpha():
                pass
            continue

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(r: F06Result) -> None:
    from collections import defaultdict
    by_sc: dict = defaultdict(lambda: defaultdict(int))
    for d in r.displacements:
        by_sc[d.subcase]['disp'] += 1
    for s in r.grid_stresses:
        by_sc[s.subcase]['gstress'] += 1
    for s in r.elem_stresses:
        key = f'estress_{s.elem_type}_{s.location}'
        by_sc[s.subcase][key] += 1
    for f in r.elem_forces:
        key = f'eforce_{f.elem_type}_{f.location}'
        by_sc[f.subcase][key] += 1
    for f in r.grid_forces:
        by_sc[f.subcase]['gforce'] += 1

    print("=" * 60)
    print("F06 PARSE SUMMARY")
    print("=" * 60)
    for sc in sorted(by_sc.keys()):
        print(f"\n  SUBCASE {sc}:")
        for k, v in sorted(by_sc[sc].items()):
            print(f"    {k:35s}: {v:4d} rows")

    print(f"\n  Total displacements: {len(r.displacements)}")
    print(f"  Total grid stresses : {len(r.grid_stresses)}")
    print(f"  Total elem stresses  : {len(r.elem_stresses)}")
    print(f"  Total elem forces    : {len(r.elem_forces)}")
    print(f"  Total grid forces    : {len(r.grid_forces)}")

    print("\n--- ELEMENT STRESS SAMPLES ---")
    for et in ('CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6'):
        for loc in ('CENTER', 'GRD'):
            rows = [s for s in r.elem_stresses
                     if s.elem_type == et and s.location == loc and s.fiber_dist < 0]
            if rows:
                print(f"\n  {et} {loc} (first 3):")
                for s in rows[:3]:
                    print(f"    ID={s.elem_id:3d} dist={s.fiber_dist:.3e}  "
                          f"NX={s.nx:.5e} NY={s.ny:.5e} SXY={s.sxy:.5e}  "
                          f"ang={s.angle:7.2f} maj={s.major:.5e}")

    print("\n--- ELEMENT FORCE SAMPLES ---")
    for et in ('CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6'):
        for loc in ('CENTER', 'GRD'):
            rows = [f for f in r.elem_forces if f.elem_type == et and f.location == loc]
            if rows:
                print(f"\n  {et} {loc} (first 3):")
                for f in rows[:3]:
                    print(f"    ID={f.elem_id:3d} grid={f.grid_id}  "
                          f"NXX={f.nxx:.5e} NYY={f.nyy:.5e} NXY={f.nxy:.5e}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV exporter
# ─────────────────────────────────────────────────────────────────────────────
import csv, os, pathlib

def _csv_path(f06_path: str, suffix: str) -> str:
    stem = pathlib.Path(f06_path).stem
    out_dir = pathlib.Path(f06_path).parent
    return str(out_dir / f"{stem}_{suffix}.csv")

def write_csvs(r: F06Result, f06_path: str) -> dict[str, str]:
    written = {}

    # Displacements
    p = _csv_path(f06_path, 'displacements')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','grid_id','coord_sys','t1','t2','t3','r1','r2','r3'])
        for d in r.displacements:
            w.writerow([d.subcase, d.grid_id, d.coord_sys,
                        d.t1, d.t2, d.t3, d.r1, d.r2, d.r3])
    written['displacements'] = p

    # Grid stresses
    p = _csv_path(f06_path, 'grid_stresses')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','surface_id','grid_id','elem_id','fiber',
                    'nx','ny','sxy','angle','major','minor','shear','von_mises'])
        for s in r.grid_stresses:
            w.writerow([s.subcase, s.surface_id, s.grid_id, s.elem_id, s.fiber,
                        s.nx, s.ny, s.sxy, s.angle, s.major, s.minor,
                        s.shear, s.von_mises])
    written['grid_stresses'] = p

    # Element stresses
    p = _csv_path(f06_path, 'elem_stresses')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','elem_type','elem_id','location','fiber_dist',
                    'nx','ny','sxy','angle','major','minor','von_mises','sxz','syz'])
        for s in r.elem_stresses:
            w.writerow([s.subcase, s.elem_type, s.elem_id, s.location, s.fiber_dist,
                        s.nx, s.ny, s.sxy, s.angle, s.major, s.minor,
                        s.von_mises, s.sxz, s.syz])
    written['elem_stresses'] = p

    # Element forces
    p = _csv_path(f06_path, 'elem_forces')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','elem_type','elem_id','location','grid_id',
                    'nxx','nyy','nxy','mxx','myy','mxy','qx','qy'])
        for f_ in r.elem_forces:
            w.writerow([f_.subcase, f_.elem_type, f_.elem_id, f_.location, f_.grid_id,
                        f_.nxx, f_.nyy, f_.nxy, f_.mxx, f_.myy, f_.mxy, f_.qx, f_.qy])
    written['elem_forces'] = p

    # Grid forces
    p = _csv_path(f06_path, 'grid_forces')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','surface_id','grid_id','elem_id',
                    'nxx','nyy','nxy','mxx','myy','mxy','qx','qy'])
        for g in r.grid_forces:
            w.writerow([g.subcase, g.surface_id, g.grid_id, g.elem_id,
                        g.nxx, g.nyy, g.nxy, g.mxx, g.myy, g.mxy, g.qx, g.qy])
    written['grid_forces'] = p

    return written


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'duel3a.F06'
    print(f"Parsing: {path}")
    result = parse_f06(path)
    print_summary(result)

    csvs = write_csvs(result, path)
    print("\n--- CSV EXPORT ---")
    for name, fp in csvs.items():
        nrows = sum(1 for _ in open(fp)) - 1
        print(f"  {name:20s}: {fp}  ({nrows} data rows)")
