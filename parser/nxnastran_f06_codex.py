"""
NX Nastran (Simcenter) F06 Parser.
Covers: CQUAD4, CQUAD8, CTRIA3, CTRIA6
  - Displacements (T1,T2,T3,R1,R2,R3)
  - Grid-point stresses (surface system, Z1/Z2/MID fibers)
  - Element stresses CENTER + CORNER (local coord system)
  - Element engineering forces CENTER + CORNER (FX,FY,FXY,MX,MY,MXY,QX,QY)

Parsing notes:
  - Carriage control is removed only from column 1; numeric zero is preserved.
  - Element context survives repeated subcase/table headers at page breaks.
  - Corner rows carry grid IDs; fiber continuation rows retain their location.
  - CQUAD4 BILIN and CTRIA3 center-only output are distinguished by columns.
  - Optional input .dat supplies title, labels, case-control and ID validation.
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
    location: str; grid_id: int; fiber_dist: float
    nx: float; ny: float; sxy: float
    angle: float; major: float; minor: float; von_mises: float

@dataclass
class ElemForceRow:
    subcase: int; elem_type: str; elem_id: int
    location: str; grid_id: int
    fx: float; fy: float; fxy: float
    mx: float; my_field: float; mxy: float
    qx: float; qy: float

@dataclass
class F06Result:
    title: str = ""
    displacements: list[DisplacementRow] = field(default_factory=list)
    grid_stresses: list[GridStressRow] = field(default_factory=list)
    elem_stresses: list[ElemStressRow] = field(default_factory=list)
    elem_forces: list[ElemForceRow] = field(default_factory=list)
    subcase_labels: dict[int, str] = field(default_factory=dict)
    case_control: dict[int, dict[str, str]] = field(default_factory=dict)
    element_types: dict[int, str] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_RE_BLANK   = re.compile(r'^\s*$')
_RE_SEPDASH = re.compile(r'^\s*-{3,}')

def _f(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    try:
        return float(s.replace('D', 'E').replace('d', 'e'))
    except ValueError:
        return 0.0

def _is_float(tok: str) -> bool:
    try:
        float(tok.replace('D', 'E').replace('d', 'e'))
        return True
    except ValueError:
        return False

def _norm_etype(s: str) -> str:
    m = {'QUAD4': 'CQUAD4', 'QUADR': 'CQUAD4', 'QUAD8': 'CQUAD8', 'TRIA3': 'CTRIA3', 'TRIAR': 'CTRIA3', 'TRIA6': 'CTRIA6'}
    return m.get(s.upper().replace(' ', '').strip(), s)

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

_RE_SUBCASE     = re.compile(r'\sSUBCASE\s+(\d+)\s*$')
_RE_SUBCASE_EQ  = re.compile(r'\sSUBCASE\s*=\s*(\d+)')
# S T R E S S E S ... S U R F A C E 100
_RE_SURFACE_HDR = re.compile(
    r'S\s+T\s+R\s+E\s+S\s+S\s+E\s+S\s+.*S\s+U\s+R\s+F\s+A\s+C\s+E\s+(\d+)')
# S T R E S S E S ... ( Q U A D 8 )
_RE_ESTR_HDR = re.compile(r'S\s+T\s+R\s+E\s+S\s+S\s+E\s+S\s+.*\((.+?)\)')
# F O R C E S ... ( Q U A D 8 )
_RE_EFOR_HDR = re.compile(r'F\s+O\s+R\s+C\s+E\s+S\s+.*\((.+?)\)')

# ─────────────────────────────────────────────────────────────────────────────
# State labels
# ─────────────────────────────────────────────────────────────────────────────
_ST_TOP       = 'top'
_ST_DISP      = 'disp'
_ST_GRID_STR  = 'grid_stress'
_ST_ELEM_STR  = 'elem_stress'
_ST_ELEM_FOR  = 'elem_force'

# ─────────────────────────────────────────────────────────────────────────────
# Main parser
# ─────────────────────────────────────────────────────────────────────────────

def _number(token: str) -> float:
    """Read printed Nastran numbers without silently replacing bad values by zero."""
    token = token.strip().replace('D', 'E').replace('d', 'e')
    if not token:
        raise ValueError('Empty numeric field')
    if 'e' not in token.lower():
        token = re.sub(r'(?<=\d)([+-]\d+)$', r'E\1', token)
    return float(token)


def _numeric(tokens) -> bool:
    try:
        for token in tokens:
            _number(token)
        return True
    except ValueError:
        return False


def _read_deck(path, result):
    """Read case-control metadata and primary shell-card IDs from this input deck.

    This is a metadata reader, not a replacement for a general BDF reader.
    It accepts free-field and small/large fixed-field primary GRID/shell cards.
    """
    in_bulk = False
    subcase = None
    grid_ids = set()
    with open(path, encoding='latin-1') as stream:
        for raw in stream:
            line = raw.split('$', 1)[0].strip()
            if not line:
                continue
            if line.upper().startswith('BEGIN BULK'):
                in_bulk = True
                continue
            if not in_bulk:
                match = re.match(r'SUBCASE\s*(?:=\s*)?(\d+)\s*$', line, re.I)
                if match:
                    subcase = int(match.group(1))
                    result.subcase_labels.setdefault(subcase, '')
                    continue
                if '=' in line:
                    key, value = (part.strip() for part in line.split('=', 1))
                    if key.upper() == 'TITLE':
                        result.title = value
                    elif subcase is not None and key.upper() == 'LABEL':
                        result.subcase_labels[subcase] = value
                    elif subcase is not None:
                        result.case_control.setdefault(subcase, {})[key.upper()] = value
                continue
            if ',' in line:
                fields = [part.strip() for part in line.split(',')]
            else:
                # Preserve columns; whitespace stripping only removed indentation.
                width = 16 if line[:8].rstrip().endswith('*') else 8
                fields = [line[:8].strip()] + [line[i:i+width].strip() for i in range(8, len(line), width)]
            card = fields[0].rstrip('*').upper()
            if len(fields) > 1 and fields[1].isdigit():
                if card == 'GRID':
                    grid_ids.add(int(fields[1]))
                elif card in ('CQUAD4', 'CQUADR', 'CQUAD8', 'CTRIA3', 'CTRIAR', 'CTRIA6'):
                    result.element_types[int(fields[1])] = {'CQUADR': 'CQUAD4', 'CTRIAR': 'CTRIA3'}.get(card, card)
    return grid_ids


def parse_f06(path: str, dat_path: Optional[str] = None) -> F06Result:
    """Read real static SC1/SC2 results, preserving element context across pages.

    The companion .dat is read automatically when present, or can be supplied
    explicitly. No files are written by this function. Rows retain NX field names
    and contain one entry per printed fiber/location, never invented results.
    """
    from pathlib import Path
    result = F06Result()
    companion = Path(dat_path) if dat_path is not None else Path(path).with_suffix('.dat')
    grid_ids = _read_deck(companion, result) if dat_path is not None or companion.exists() else set()
    state = _ST_TOP
    subcase = surface = etype = eid = None
    location, grid = 'CENTER', 0
    has_grid_column = False
    grid_stress_context = None

    with open(path, encoding='latin-1') as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.rstrip('\r\n')
            # Page titles contain dates and PAGE numbers, not result data.
            if re.search(r'\bPAGE\s+\d+\s*$', line):
                continue
            # Carriage control occupies column 1; a numeric token beginning
            # with zero (e.g. 0.0 or 0.993) must remain intact.
            if len(line) > 1 and line[0] in '01' and line[1].isspace():
                line = line[1:]
            stripped = line.strip()
            if not stripped or _RE_SEPDASH.match(stripped):
                continue
            match = re.search(r'\bSUBCASE\s*(?:=\s*)?(\d+)\s*$', stripped)
            if match:
                new_case = int(match.group(1))
                if new_case != subcase:
                    state, surface, etype, eid = _ST_TOP, None, None, None
                    grid_stress_context = None
                subcase = new_case
                result.subcase_labels.setdefault(subcase, '')
                continue
            compact = re.sub(r'\s+', '', stripped).upper()
            if compact == 'DISPLACEMENTVECTOR':
                state, eid = _ST_DISP, None
                continue
            match = _RE_SURFACE_HDR.search(stripped)
            if match:
                new_surface = int(match.group(1))
                if state != _ST_GRID_STR or surface != new_surface:
                    grid_stress_context = None
                surface, state = new_surface, _ST_GRID_STR
                continue
            match_s = _RE_ESTR_HDR.search(stripped)
            match_f = _RE_EFOR_HDR.search(stripped)
            if match_s or match_f:
                match = match_s or match_f
                new_state = _ST_ELEM_STR if match_s else _ST_ELEM_FOR
                new_type = _norm_etype(match.group(1))
                if state != new_state or etype != new_type:
                    eid, grid, location = None, 0, 'CENTER'
                    has_grid_column = False
                state, etype = new_state, new_type
                continue
            if 'GRID-ID' in stripped and state in (_ST_ELEM_STR, _ST_ELEM_FOR):
                has_grid_column = True
                continue
            tokens = stripped.split()
            if subcase is None:
                continue
            try:
                if state == _ST_DISP:
                    if len(tokens) == 8 and tokens[0].isdigit() and tokens[1] in ('G', 'S', 'O', 'C', 'L', 'R'):
                        result.displacements.append(DisplacementRow(subcase, int(tokens[0]), *map(_number, tokens[2:])))
                elif state == _ST_GRID_STR and surface is not None:
                    if len(tokens) == 11 and tokens[0].isdigit() and tokens[1].isdigit() and tokens[2] in ('Z1', 'Z2', 'MID'):
                        grid_stress_context = int(tokens[0]), int(tokens[1])
                        result.grid_stresses.append(GridStressRow(subcase, surface, *grid_stress_context, tokens[2], *map(_number, tokens[3:])))
                    elif len(tokens) == 9 and tokens[0] in ('Z1', 'Z2', 'MID') and grid_stress_context is not None:
                        result.grid_stresses.append(GridStressRow(subcase, surface, *grid_stress_context, tokens[0], *map(_number, tokens[1:])))
                elif state in (_ST_ELEM_STR, _ST_ELEM_FOR) and etype in ('CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6'):
                    values = None
                    # Explicit element and location: CEN/n or a printed grid ID.
                    if len(tokens) == 10 and tokens[0].isdigit() and (tokens[1].startswith('CEN/') or tokens[1].isdigit()) and _numeric(tokens[2:]):
                        eid = int(tokens[0])
                        location = 'CENTER' if tokens[1].startswith('CEN/') else 'CORNER'
                        grid = 0 if location == 'CENTER' else int(tokens[1])
                        has_grid_column = True
                        values = tokens[2:]
                    elif len(tokens) == 9 and tokens[0].isdigit() and _numeric(tokens[1:]):
                        if has_grid_column:
                            if eid is None:
                                raise ValueError('Corner row has no preceding element')
                            grid, location = int(tokens[0]), 'CORNER'
                        else:
                            eid, grid, location = int(tokens[0]), 0, 'CENTER'
                        values = tokens[1:]
                    elif state == _ST_ELEM_STR and len(tokens) == 8 and eid is not None and _numeric(tokens):
                        # The second fiber retains the exact preceding location.
                        values = tokens
                    if values is not None:
                        if state == _ST_ELEM_STR:
                            result.elem_stresses.append(ElemStressRow(subcase, etype, eid, location, grid, *map(_number, values)))
                        else:
                            result.elem_forces.append(ElemForceRow(subcase, etype, eid, location, grid, *map(_number, values)))
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: {error}: {stripped}') from error

    # Check parsed IDs against the provided input; this catches lost element
    # context rather than allowing a node number to become an element number.
    for row in result.elem_stresses + result.elem_forces:
        if result.element_types and result.element_types.get(row.elem_id) != row.elem_type:
            raise ValueError(f'Element {row.elem_id} / {row.elem_type} does not match input {companion}')
        if row.grid_id and grid_ids and row.grid_id not in grid_ids:
            raise ValueError(f'Grid {row.grid_id} does not match input {companion}')
    for row in result.displacements:
        if grid_ids and row.grid_id not in grid_ids:
            raise ValueError(f'Displacement grid {row.grid_id} does not match input {companion}')
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
        by_sc[s.subcase][f'estress_{s.elem_type}_{s.location}'] += 1
    for f in r.elem_forces:
        by_sc[f.subcase][f'eforce_{f.elem_type}_{f.location}'] += 1

    print("=" * 60)
    print("NX NASTRAN F06 PARSE SUMMARY")
    print("=" * 60)
    for sc in sorted(by_sc.keys()):
        print(f"\n  SUBCASE {sc}:")
        for k, v in sorted(by_sc[sc].items()):
            print(f"    {k:38s}: {v:4d} rows")

    print(f"\n  Total displacements: {len(r.displacements)}")
    print(f"  Total grid stresses : {len(r.grid_stresses)}")
    print(f"  Total elem stresses  : {len(r.elem_stresses)}")
    print(f"  Total elem forces   : {len(r.elem_forces)}")

    print("\n--- ELEMENT STRESS SAMPLES (CENTER Z1 fiber) ---")
    for et in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6'):
        rows = [s for s in r.elem_stresses
                 if s.elem_type == et and s.location == 'CENTER' and s.fiber_dist < 0]
        if rows:
            print(f"\n  {et} CENTER (first 3):")
            for s in rows[:3]:
                print(f"    ID={s.elem_id:3d} dist={s.fiber_dist:.3e}  "
                      f"NX={s.nx:.5e} NY={s.ny:.5e} SXY={s.sxy:.5e}")

    print("\n--- ELEMENT STRESS SAMPLES (CORNER) ---")
    for et in ('CQUAD8','CTRIA6'):
        rows = [s for s in r.elem_stresses
                 if s.elem_type == et and s.location == 'CORNER']
        if rows:
            print(f"\n  {et} CORNER (first 3):")
            for s in rows[:3]:
                print(f"    ID={s.elem_id:3d} grid={s.grid_id:3d} dist={s.fiber_dist:.3e}  "
                      f"NX={s.nx:.5e} NY={s.ny:.5e} SXY={s.sxy:.5e}")

    print("\n--- ELEMENT FORCE SAMPLES ---")
    for et in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6'):
        rows = [f for f in r.elem_forces if f.elem_type == et]
        if rows:
            print(f"\n  {et} (first 3):")
            for f in rows[:3]:
                print(f"    ID={f.elem_id:3d} loc={f.location:7s} grid={f.grid_id:3d}  "
                      f"MX={f.mx:.5e} MY={f.my_field:.5e}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV exporter
# ─────────────────────────────────────────────────────────────────────────────
import csv, pathlib

def _csv_path(f06_path: str, suffix: str) -> str:
    stem = pathlib.Path(f06_path).stem
    return str(pathlib.Path(f06_path).parent / f"{stem}_{suffix}.csv")

def write_csvs(r: F06Result, f06_path: str, output_dir: Optional[str] = None) -> dict[str, str]:
    if output_dir is not None:
        directory = pathlib.Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        f06_path = str(directory / pathlib.Path(f06_path).name)
    written = {}

    p = _csv_path(f06_path, 'displacements')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','grid_id','t1','t2','t3','r1','r2','r3'])
        for d in r.displacements:
            w.writerow([d.subcase, d.grid_id, d.t1,d.t2,d.t3,d.r1,d.r2,d.r3])
    written['displacements'] = p

    p = _csv_path(f06_path, 'grid_stresses')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','surface_id','grid_id','elem_id','fiber',
                    'nx','ny','sxy','angle','major','minor','shear','von_mises'])
        for s in r.grid_stresses:
            w.writerow([s.subcase,s.surface_id,s.grid_id,s.elem_id,s.fiber,
                        s.nx,s.ny,s.sxy,s.angle,s.major,s.minor,s.shear,s.von_mises])
    written['grid_stresses'] = p

    p = _csv_path(f06_path, 'elem_stresses')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','elem_type','elem_id','location','grid_id','fiber_dist',
                    'nx','ny','sxy','angle','major','minor','von_mises'])
        for s in r.elem_stresses:
            w.writerow([s.subcase,s.elem_type,s.elem_id,s.location,s.grid_id,s.fiber_dist,
                        s.nx,s.ny,s.sxy,s.angle,s.major,s.minor,s.von_mises])
    written['elem_stresses'] = p

    p = _csv_path(f06_path, 'elem_forces')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['subcase','elem_type','elem_id','location','grid_id',
                    'fx','fy','fxy','mx','my','mxy','qx','qy'])
        for f_ in r.elem_forces:
            w.writerow([f_.subcase,f_.elem_type,f_.elem_id,f_.location,f_.grid_id,
                        f_.fx,f_.fy,f_.fxy,f_.mx,f_.my_field,f_.mxy,f_.qx,f_.qy])
    written['elem_forces'] = p

    return written


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read NX Nastran shell results for each subcase.')
    parser.add_argument('f06', nargs='?', default=r'C:\PROJECTAI\18a\test\duel3a_nastran.f06')
    parser.add_argument('--dat', help='Input deck (default: matching .dat when present)')
    parser.add_argument('--output-dir', help='CSV directory (default: beside F06)')
    args = parser.parse_args()
    print(f'Parsing: {args.f06}')
    result = parse_f06(args.f06, args.dat)
    print_summary(result)
    csvs = write_csvs(result, args.f06, args.output_dir)
    print('\n--- CSV EXPORT ---')
    for name, fp in csvs.items():
        with open(fp, newline='', encoding='utf8') as stream:
            nrows = sum(1 for _ in csv.reader(stream)) - 1
        print(f'  {name:20s}: {fp}  ({nrows} data rows)')
