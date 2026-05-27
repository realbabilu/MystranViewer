"""
MYSTRAN/NASTRAN .f06 output parser.
Handles both MYSTRAN format (8 fields: nid coord T1..R3)
and MSC NASTRAN format (7 fields: nid type T1..R3).
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DisplacementResult:
    subcase: int
    node_id: int
    t1: float; t2: float; t3: float
    r1: float; r2: float; r3: float

    @property
    def translation(self):
        return np.array([self.t1, self.t2, self.t3], dtype=np.float32)

    @property
    def magnitude(self):
        return float(np.sqrt(self.t1**2 + self.t2**2 + self.t3**2))


@dataclass
class ElementStress:
    subcase: int
    elem_id: int
    elem_type: str
    values: Dict = field(default_factory=dict)
    von_mises: float = 0.0


@dataclass
class ElementForce:
    subcase: int; elem_id: int; elem_type: str
    fx:float=0.0; fy:float=0.0; fxy:float=0.0
    mx:float=0.0; my:float=0.0; mxy:float=0.0
    qx:float=0.0; qy:float=0.0
    af:float=0.0; sf1:float=0.0; sf2:float=0.0
    bm1:float=0.0; bm2:float=0.0; tq:float=0.0

    @property
    def values(self):
        return {'fx':self.fx,'fy':self.fy,'fxy':self.fxy,
                'mx':self.mx,'my':self.my,'mxy':self.mxy,
                'qx':self.qx,'qy':self.qy}


@dataclass
class F06Results:
    subcases: List[int] = field(default_factory=list)
    displacements: Dict[int, Dict[int, DisplacementResult]] = field(default_factory=dict)
    stresses: Dict[int, Dict[int, ElementStress]] = field(default_factory=dict)
    forces: Dict[int, Dict] = field(default_factory=dict)
    reactions: Dict[int, Dict[int, Dict[str, float]]] = field(default_factory=dict)

    def max_displacement(self, subcase: int) -> float:
        if subcase not in self.displacements: return 0.0
        vals = [d.magnitude for d in self.displacements[subcase].values()]
        return max(vals) if vals else 0.0

    def displacement_array(self, subcase: int):
        if subcase not in self.displacements: return np.array([]), np.array([])
        items = sorted(self.displacements[subcase].items())
        return np.array([i for i,_ in items]), np.array([d.magnitude for _,d in items])

    def von_mises_array(self, subcase: int):
        if subcase not in self.stresses: return np.array([]), np.array([])
        items = sorted(self.stresses[subcase].items())
        return np.array([i for i,_ in items]), np.array([s.von_mises for _,s in items])


def _parse_float(s):
    s = s.strip()
    if not s: return 0.0
    s = re.sub(r'([0-9])([+-])([0-9])', r'\1e\2\3', s)
    try: return float(s)
    except: return 0.0


class F06Parser:
    _RE_SUBCASE  = re.compile(r'(?:SUBCASE|OUTPUT FOR SUBCASE)\s+(\d+)', re.I)
    _RE_DISP_HDR = re.compile(r'D\s*I\s*S\s*P\s*L\s*A\s*C\s*E\s*M\s*E\s*N\s*T', re.I)
    _RE_FORCE2D  = re.compile(
        r'F[\s]*O[\s]*R[\s]*C[\s]*E[\s]*S.{1,200}[(]([\w\s]+)[)]', re.I)
    _RE_STRESS2D = re.compile(
        r'S[\s]*T[\s]*R[\s]*E[\s]*S[\s]*S[\s]*E[\s]*S.{1,200}[(]([\w\s]+)[)]',
        re.I)

    def parse(self, filepath: str) -> F06Results:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()

        results = F06Results()
        current_sc = 1
        results.subcases = [1]
        results.displacements[1] = {}
        results.stresses[1] = {}

        i = 0
        while i < len(lines):
            line = lines[i]

            # Track subcase
            m = self._RE_SUBCASE.search(line)
            if m:
                current_sc = int(m.group(1))
                if current_sc not in results.subcases:
                    results.subcases.append(current_sc)
                    results.displacements[current_sc] = {}
                    results.stresses[current_sc] = {}
                    if current_sc not in results.forces: results.forces[current_sc] = {}

            # Displacement table header
            if self._RE_DISP_HDR.search(line):
                i = self._parse_displacements(lines, i+1, current_sc, results)
                continue

            # Force table (plate/shell element forces)
            m3 = self._RE_FORCE2D.search(line)
            if m3:
                raw3 = re.sub(r'\s+','',m3.group(1)).upper()
                fmap3 = {'QUADR':'CQUAD4','QUAD4':'CQUAD4','TRIAR':'CTRIA3','TRIA3':'CTRIA3'}
                ftype = fmap3.get(raw3, raw3)
                if current_sc not in results.forces: results.forces[current_sc]={}
                i = self._parse_force_2d(lines, i+1, current_sc, ftype, results)
                continue

            # Stress table
            m2 = self._RE_STRESS2D.search(line)
            if m2:
                # Strip spaces from spaced format: 'Q U A D R' -> 'QUADR'
                raw = re.sub(r'\s+', '', m2.group(1)).upper()
                emap = {'QUADR':'CQUAD4','QUAD4':'CQUAD4','QUAD8':'CQUAD8',
                        'TRIAR':'CTRIA3','TRIA3':'CTRIA3','TRIA6':'CTRIA6',
                        'CQUAD4':'CQUAD4','CTRIA3':'CTRIA3',
                        'PYRAM':'CPYRAM','CPYRA5':'CPYRAM','HEXA':'CHEXA',
                        'PENTA':'CPENTA','TETRA':'CTETRA'}
                etype = emap.get(raw, raw)
                # Route solid vs shell
                if etype in ('CPYRAM','CHEXA','CPENTA','CTETRA'):
                    i = self._parse_stress_solid(lines, i+1, current_sc, etype, results)
                else:
                    i = self._parse_stress_2d(lines, i+1, current_sc, etype, results)
                continue

            i += 1

        # Ensure at least subcase 1
        if not results.subcases:
            results.subcases = [1]

        nd = sum(len(v) for v in results.displacements.values())
        ns = sum(len(v) for v in results.stresses.values())
        print(f"[F06] subcases={results.subcases} nodes={nd} stresses={ns}")
        return results

    def _parse_displacements(self, lines, start, subcase, results):
        """Parse MYSTRAN/NASTRAN displacement block.
        MYSTRAN: nid  coord_sys  T1  T2  T3  R1  R2  R3  (8 tokens, coord_sys=int)
        NASTRAN: nid  type       T1  T2  T3  R1  R2  R3  (7 tokens, type=G/S string)
        Key: first token must be a valid integer node id.
        """
        i = start
        while i < len(lines):
            line = lines[i]; stripped = line.strip(); i += 1
            if not stripped: continue
            # Hard end markers
            if any(stripped.startswith(k) for k in ('---','===','>>',' >>','**')):
                break
            # New section header (S T R E S S E S, FORCES, etc.) - stop
            if re.search(r'S\s+T\s+R\s+E\s+S\s+S|F\s+O\s+R\s+C\s+E\s+S', stripped):
                i -= 1  # put the line back
                break
            # Summary rows - skip but don't end
            if any(stripped.startswith(k) for k in ('MAX','MIN','ABS','*for')):
                continue
            parts = stripped.split()
            if not parts: continue
            # First token MUST be integer node id
            try:
                nid = int(parts[0])
            except ValueError:
                continue  # header or label line - skip
            # Need enough fields
            if len(parts) < 7:
                continue
            # Both MYSTRAN and NX/MSC NASTRAN have 2 fields before T1:
            # MYSTRAN: nid coord_sys T1 T2 T3 R1 R2 R3  (coord_sys is int)
            # NX/MSC:  nid type     T1 T2 T3 R1 R2 R3  (type is 'G','S', etc)
            # Always skip both -> data starts at index 2
            offset = 2
            if len(parts) < offset + 6:
                continue
            try:
                vals = [_parse_float(parts[offset+j]) for j in range(6)]
                results.displacements[subcase][nid] = DisplacementResult(
                    subcase=subcase, node_id=nid,
                    t1=vals[0], t2=vals[1], t3=vals[2],
                    r1=vals[3], r2=vals[4], r3=vals[5]
                )
            except (ValueError, IndexError):
                pass
        return i

    def _parse_stress_2d(self, lines, start, subcase, etype, results):
        """Parse NX/MSC shell stress: 2 rows/elem (top+bot fiber)."""
        i = start
        pending = {}   # eid -> {vm:[], oxx:[], ...}
        last_eid = None

        while i < len(lines):
            line = lines[i]; i += 1
            stripped = line.strip()
            if not stripped: continue
            if re.search(r'PAGE\s+\d+', stripped): continue

            # Skip column header lines (ELEMENT ID., FIBER DISTANCE, etc.)
            if re.match(r'^[A-Z]', stripped) and not re.match(r'^[0-9-]', stripped):
                # Only break on SPACED stress header (S T R E S S E S) or SUBCASE line
                if (re.search(r'S[ ]+T[ ]+R[ ]+E[ ]+S[ ]+S', stripped) or
                        re.match(r'SUBCASE|LOAD STEP|NASTRAN FORT', stripped)):
                    break
                continue  # column headers - skip

            # Remove leading '0' page marker
            s = stripped.lstrip('0').strip()
            parts = s.split()
            if len(parts) < 8: continue

            # Skip lines starting with non-numeric/non-minus chars (e.g. '(TOTAL...')
            if parts[0] and parts[0][0] not in '0123456789-':
                # Could be a continuation if it starts with a float
                try: float(parts[0].replace('E','e').replace('D','e'))
                except ValueError: continue  # e.g. '(TOTAL', 'LOAD' etc

            # Determine eid vs continuation
            try:
                eid = int(parts[0])
                v_start = 1
                last_eid = eid
            except ValueError:
                if last_eid is None: continue
                # Validate continuation: parts[0] must be a float (fiber_dist)
                try: float(parts[0].replace('E','e').replace('D','e'))
                except ValueError: continue  # not a data line
                eid = last_eid
                v_start = 0

            try:
                oxx  = _parse_float(parts[v_start+1])
                oyy  = _parse_float(parts[v_start+2])
                txy  = _parse_float(parts[v_start+3])
                omax = _parse_float(parts[v_start+5])
                omin = _parse_float(parts[v_start+6])
                vm   = abs(_parse_float(parts[v_start+7])) if len(parts) > v_start+7 else abs(omax)
            except (IndexError, ValueError):
                continue

            if eid not in pending:
                pending[eid] = {'vm':[],'oxx':[],'oyy':[],'txy':[],'omax':[],'omin':[]}
            p = pending[eid]
            p['vm'].append(vm); p['oxx'].append(oxx); p['oyy'].append(oyy)
            p['txy'].append(txy); p['omax'].append(omax); p['omin'].append(omin)

        # Store all accumulated
        import numpy as np
        for eid, p in pending.items():
            if not p['vm']: continue
            vm_max = float(max(p['vm']))
            results.stresses[subcase][eid] = ElementStress(
                subcase=subcase, elem_id=eid, elem_type=etype,
                values={'von_mises': vm_max,
                        'oxx':  float(np.mean(p['oxx'])),
                        'oyy':  float(np.mean(p['oyy'])),
                        'txy':  float(np.mean(p['txy'])),
                        'omax': float(np.max(p['omax'])),
                        'omin': float(np.min(p['omin']))},
                von_mises=vm_max)
        return i

    def _parse_force_2d(self, lines, start, subcase, etype, results):
        """Shell element forces: one row per element FX FY FXY MX MY MXY QX QY"""
        i = start
        while i < len(lines):
            line = lines[i]; i += 1
            stripped = line.strip()
            if not stripped: continue
            if re.search(r'PAGE\s+\d+', stripped): continue
            if re.match(r'^[A-Z(]', stripped) and not re.match(r'^[0-9-]', stripped):
                if re.search(r'F[ ]+O[ ]+R[ ]+C[ ]+E[ ]+S|S[ ]+T[ ]+R[ ]+E[ ]+S[ ]+S|'
                             r'S[ ]+T[ ]+R[ ]+A[ ]+I[ ]+N|SUBCASE|LOAD STEP', stripped):
                    i -= 1; break
                continue
            s = stripped.lstrip('0').strip()
            parts = s.split()
            if len(parts) < 8: continue
            if parts[0] and parts[0][0] not in '0123456789-':
                try: float(parts[0].replace('E','e'))
                except: continue
            try: eid = int(parts[0])
            except: continue
            try:
                fx =_parse_float(parts[1]); fy =_parse_float(parts[2])
                fxy=_parse_float(parts[3]); mx =_parse_float(parts[4])
                my =_parse_float(parts[5]); mxy=_parse_float(parts[6])
                qx =_parse_float(parts[7])
                qy = _parse_float(parts[8]) if len(parts) > 8 else 0.0
            except (IndexError, ValueError): continue
            results.forces[subcase][eid] = ElementForce(
                subcase=subcase, elem_id=eid, elem_type=etype,
                fx=fx, fy=fy, fxy=fxy, mx=mx, my=my, mxy=mxy, qx=qx, qy=qy)
        return i


    def _parse_stress_solid(self, lines, start, subcase, etype, results):
        """Parse solid element stresses: CENTER->element, corners->nodal avg.
        Each element block: header, CENTER (3 lines), then N corner groups (3 lines each).
        The first line of each group contains VM as last float.
        """
        import numpy as np
        i = start
        cur_eid   = None
        stage     = None   # 'center' or 'corner'
        stage_line = 0     # 1=X, 2=Y, 3=Z within current group
        cur_vm    = None
        cur_nid   = None
        corner_vals = {}   # nid -> [vm, ...]

        while i < len(lines):
            line = lines[i]; i += 1
            stripped = line.strip()
            if not stripped: continue

            # Strip leading Fortran page marker '0'
            s = stripped.lstrip('0').strip()
            if not s: continue

            # Structural headers -> end block
            if re.search(r'S[ ]+T[ ]+R[ ]+E[ ]+S|S[ ]+T[ ]+R[ ]+A[ ]+I|'
                         r'F[ ]+O[ ]+R[ ]+C|SUBCASE|LOAD STEP|PAGE', s):
                if re.search(r'PAGE\s+\d+', s): continue
                i -= 1; break

            parts = s.split()
            if not parts: continue

            # --- Line 2 or 3 of current group (Y line or Z line) ---
            # stage_line=1 after X line, so Y is stage_line=1, Z is stage_line=2
            if stage_line > 0 and re.match(r'^[YZ]\s', s):
                stage_line += 1
                if stage_line == 3:  # just finished Z line (3rd line) -> group complete
                    if stage == 'corner' and cur_nid is not None and cur_vm is not None:
                        if cur_nid not in corner_vals: corner_vals[cur_nid] = []
                        corner_vals[cur_nid].append(cur_vm)
                        cur_nid = None; cur_vm = None
                    stage_line = 0
                continue

            # --- CENTER line (first line of center group) ---
            if s.upper().startswith('CENTER'):
                stage = 'center'; stage_line = 1
                nums = re.findall(r'[-]?\d+\.\d+E[+-]?\d+|[-]?\d+\.\d+', s)
                cur_vm = abs(float(nums[-1])) if nums else None
                # Get oxx from X value
                m = re.search(r'\bX\b\s+([-\d.E+]+)', s, re.I)
                m2 = re.search(r'\bXY\b\s+([-\d.E+]+)', s, re.I)
                if cur_eid is not None:
                    cv = {'vm': cur_vm, 'sx': float(m.group(1)) if m else 0,
                          'txy': float(m2.group(1)) if m2 else 0}
                    results.stresses[subcase][cur_eid] = ElementStress(
                        subcase=subcase, elem_id=cur_eid, elem_type=etype,
                        values={'von_mises':cur_vm or 0,
                                'oxx':cv['sx'],'oyy':0,'txy':cv['txy']},
                        von_mises=cur_vm or 0)
                continue

            # --- Try integer: either element header or corner node line ---
            try:
                val = int(parts[0])
                if len(parts) > 1 and not parts[1].replace('E','').replace('-','').replace('.','').replace('+','').isdigit():
                    # Second field non-numeric -> could be element header ('0GRID') or corner line ('X')
                    if parts[1].upper() == 'X' or re.match(r'^X\b', parts[1]):
                        # Corner node line: 'nid  X  val  XY  val  A  val  LX...  vm'
                        stage = 'corner'; stage_line = 1; cur_nid = val
                        nums = re.findall(r'[-]?\d+\.\d+E[+-]?\d+|[-]?\d+\.\d+', s)
                        cur_vm = abs(float(nums[-1])) if nums else None
                    else:
                        # Element header: 'eid  0GRID CS ...'
                        cur_eid = val; stage = None; stage_line = 0
                        cur_nid = None; cur_vm = None
                else:
                    # Purely numeric second field is unusual here - skip
                    pass
            except ValueError:
                # Non-integer first field that's not Y/Z/CENTER -> column header, skip
                pass

        # Store any remaining corner values
        if stage == 'corner' and cur_nid is not None and cur_vm is not None:
            if cur_nid not in corner_vals: corner_vals[cur_nid] = []
            corner_vals[cur_nid].append(cur_vm)

        # Accumulate nodal corner values (merge across sections)
        if corner_vals:
            acc = results.stresses[subcase].get('_nodal_acc', {})
            for nid, vms in corner_vals.items():
                if nid not in acc: acc[nid] = []
                acc[nid].extend(vms)
            results.stresses[subcase]['_nodal_acc'] = acc
            # Recompute nodal avg
            results.stresses[subcase]['_nodal_avg'] = {
                nid: float(np.mean(vms)) for nid, vms in acc.items()}

        st = {k:v for k,v in results.stresses[subcase].items()
              if k!='_nodal_avg' and hasattr(v,'von_mises')}
        nav = results.stresses[subcase].get('_nodal_avg', {})
        if st:
            vms = [v.von_mises for v in st.values()]
            print(f"  [Stress] {etype} SC{subcase}: {len(st)} elems, "
                  f"{len(nav)} nodes (corner), "
                  f"vm=[{min(vms):.3e}, {max(vms):.3e}]")
        return i




def load_f06(filepath: str) -> F06Results:
    """Parse a NASTRAN/MYSTRAN F06 output file."""
    return F06Parser().parse(filepath)
