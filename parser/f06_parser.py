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
class ElementGPStress:
    """Grid-point surface stress in global coordinates (from OP2 GPSTRESS)."""
    subcase: int
    node_id: int
    sxx: float = 0.0   # normal x (global)
    syy: float = 0.0   # normal y (global)
    txy: float = 0.0   # shear xy (global)
    angle: float = 0.0 # principal angle (deg)
    s1: float = 0.0    # max principal
    s2: float = 0.0    # min principal
    s12: float = 0.0   # shear on principal plane
    ovm: float = 0.0   # von Mises

    @property
    def von_mises(self):
        return self.ovm


@dataclass
class F06Results:
    subcases: List[int] = field(default_factory=list)
    displacements: Dict[int, Dict[int, DisplacementResult]] = field(default_factory=dict)
    stresses: Dict[int, Dict[int, ElementStress]] = field(default_factory=dict)
    forces: Dict[int, Dict] = field(default_factory=dict)
    reactions: Dict[int, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    gp_stresses: Dict[int, Dict[tuple, 'ElementGPStress']] = field(default_factory=dict)  # subcase -> (node_id, fiber) -> GP stress
    _gp_forces: Dict[int, Dict[int, Dict[str, float]]] = field(default_factory=dict)  # subcase -> node_id -> {nxx,nyy,nxy,mxx,myy,mxy,qx,qy}

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
    _RE_SUBCASE  = re.compile(r'SUBCASE\s+(\d+)', re.I)
    _RE_SUBCASE2 = re.compile(r'SUBCASE\s*=\s*(\d+)', re.I)
    _RE_SUBCASE3 = re.compile(r'OUTPUT FOR SUBCASE\s+(\d+)', re.I)
    _RE_DISP_HDR = re.compile(r'D\s*I\s*S\s*P\s*L\s*A\s*C\s*E\s*M\s*E\s*N\s*T', re.I)
    _RE_FORCE2D  = re.compile(
        r'F[\s]*O[\s]*R[\s]*C[\s]*E[\s]*S.{1,200}[(]([\w\s]+)[)]', re.I)
    _RE_STRESS2D = re.compile(
        r'S[\s]*T[\s]*R[\s]*E[\s]*S[\s]*S[\s]*E[\s]*S.{1,200}[(]([\w\s]+)[)]',
        re.I)
    _RE_GPSTRESS_HDR = re.compile(
        r'S\s+T\s+R\s+E\s+S\s+S\s+E\s+S\s+A\s+T\s+G\s+R\s+I\s+D\s+P\s+O\s+I\s+N\s+T\s+S', re.I)
    _RE_GPFORCE_HDR = re.compile(
        r'F\s+O\s+R\s+C\s+E\s+S\s+A\s+T\s+G\s+R\s+I\s+D\s+P\s+O\s+I\s+N\s+T\s+S', re.I)
    _RE_ENG_FORCE_HDR = re.compile(
        r'E\s+L\s+E\s+M\s+E\s+N\s+T\s+E\s+N\s+G\s+I\s+N\s+E\s+E\s+R\s+I\s+N\s+G\s+F\s+O\s+R\s+C\s+E\s+S', re.I)
    # MYSTRAN element stress header — E L E M E N T [SPC] S T R E S S E S (no parentheses)
    # Pattern: E\s+L\s+E\s+M\s+E\s+N\s+T\s+S\s+T\s+R\s+E\s+S\s+S\s+E\s+S
    _RE_MYSTRAN_STRESS_HDR = re.compile(
        r'E\s+L\s+E\s+M\s+E\s+N\s+T\s+S\s+T\s+R\s+E\s+S\s+S\s+E\s+S', re.I)
    _emap = {
        'QUADR':'CQUAD4','QUAD4':'CQUAD4','QUAD8':'CQUAD8',
        'TRIAR':'CTRIA3','TRIA3':'CTRIA3','TRIA6':'CTRIA6',
        'CQUAD4':'CQUAD4','CTRIA3':'CTRIA3',
        'PYRAM':'CPYRAM','CPYRA5':'CPYRAM','HEXA':'CHEXA',
        'PENTA':'CPENTA','TETRA':'CTETRA'}

    def _detect_format(self, lines):
        """Detect F06 format: MYSTRAN or NASTRAN."""
        for line in lines[:20]:
            if 'MYSTRAN BEGIN' in line.upper():
                return 'MYSTRAN'
        return 'NASTRAN'

    # ----------------------------------------------------------------
    # Subcase markers: NASTRAN uses OUTPUT FOR SUBCASE; MYSTRAN embeds
    # "0 ... SUBCASE N" (with leading zeros) within SC1's data stream.
    # ----------------------------------------------------------------
    _RE_MYSTRAN_SC_MARKER = re.compile(r'^0\s*SUBCASE\s+(\d+)', re.I)
    _RE_NASTRAN_SC_MARKER = re.compile(r'OUTPUT FOR SUBCASE\s+(\d+)', re.I)

    def parse(self, filepath: str) -> F06Results:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()

        is_mystran = self._detect_format(lines) == 'MYSTRAN'
        print(f"[F06] Format detected: {'MYSTRAN' if is_mystran else 'NASTRAN'}")

        results = F06Results()
        current_sc = 1
        results.subcases = [1]
        results.displacements[1] = {}
        results.stresses[1] = {}

        # Persistent element type carrier: valid header sets it, sub-blocks use it
        current_etype = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Subcase tracking ---
            # NASTRAN: OUTPUT FOR SUBCASE N  (full-width header line)
            # MYSTRAN: '0 ... SUBCASE N' embedded within other output (with leading zeros)
            m = (self._RE_SUBCASE.search(line)
                  or self._RE_SUBCASE2.search(line)
                  or self._RE_SUBCASE3.search(line))
            if m:
                sc_new = int(m.group(1))
                current_sc = sc_new
                if current_sc not in results.subcases:
                    results.subcases.append(current_sc)
                    results.displacements[current_sc] = {}
                    results.stresses[current_sc] = {}
                    if current_sc not in results.forces: results.forces[current_sc] = {}
                current_etype = None  # reset element type on subcase change

            # MYSTRAN embedded SC marker: '0SUBCASE N' within other sections
            # Resets element type since we're entering a new subcase's data block
            if is_mystran:
                m_sc = self._RE_MYSTRAN_SC_MARKER.match(line)
                if m_sc:
                    sc_num = int(m_sc.group(1))
                    if sc_num != current_sc:
                        current_sc = sc_num
                        current_etype = None
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

            # GP stress block (STRESSES AT GRID POINTS)
            if self._RE_GPSTRESS_HDR.search(line):
                i = self._parse_gp_stress(lines, i+1, current_sc, results)
                continue

            # GP force block (FORCES AT GRID POINTS)
            if self._RE_GPFORCE_HDR.search(line):
                i = self._parse_gp_force(lines, i+1, current_sc, results)
                continue

            # Element engineering forces (ELEMENT ENGINEERING FORCES)
            if self._RE_ENG_FORCE_HDR.search(line):
                i = self._parse_eng_forces(lines, i+1, current_sc, results)
                continue

            # MYSTRAN: 'E L E M E N T   S T R E S S E S ...' (no parentheses)
            # Type info is on the NEXT line: 'F O R   E L E M E N T   T Y P E   Q U A D 4'
            m_mystran = self._RE_MYSTRAN_STRESS_HDR.search(line)
            if m_mystran:
                # Look for TYPE on the next line
                etype = None
                if i+1 < len(lines):
                    dspace = re.sub(r'\s+', '', lines[i+1].strip()).upper()
                    mtype = re.search(r'TYPE([A-Z0-9]+)', dspace)
                    if mtype:
                        etype = self._emap.get(mtype.group(1), mtype.group(1))
                if etype not in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                                  'CPYRAM','CHEXA','CPENTA','CTETRA'):
                    # Try backward (older format)
                    for j in range(i-1, max(i-5, -1), -1):
                        dspace = re.sub(r'\s+', '', lines[j].strip()).upper()
                        mtype = re.search(r'TYPE([A-Z0-9]+)', dspace)
                        if mtype:
                            etype = self._emap.get(mtype.group(1), mtype.group(1))
                            break
                if etype in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                             'CPYRAM','CHEXA','CPENTA','CTETRA'):
                    current_etype = etype
                    i = self._parse_stress_2d(lines, i+1, current_sc, etype, results)
                    continue

            # Stress table — NASTRAN format (has parentheses with element type):
            m2 = self._RE_STRESS2D.search(line)
            if m2:
                raw = re.sub(r'\s+', '', m2.group(1)).upper()
                etype = self._emap.get(raw, raw)

                if etype not in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                                  'CPYRAM','CHEXA','CPENTA','CTETRA'):
                    # Try to extract element type from this same line (NASTRAN inline format)
                    # e.g. 'S T R E S S ... ( Q U A D 8 ) QUAD8'  → extract 'QUAD8' after ')'
                    inline = re.search(r'\)([^)]+)$', line.strip())
                    if inline:
                        candidate = re.sub(r'\s+', '', inline.group(1)).upper()
                        etype = self._emap.get(candidate, candidate)
                    # If still unknown, look for MYSTRAN 'TYPE QUAD4' on the NEXT line
                    # (MYSTRAN puts the type info AFTER the S T R E S S E S header line)
                    if etype not in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                                      'CPYRAM','CHEXA','CPENTA','CTETRA'):
                        if i+1 < len(lines):
                            dspace = re.sub(r'\s+', '', lines[i+1].strip()).upper()
                            mtype = re.search(r'TYPE([A-Z0-9]+)', dspace)
                            if mtype:
                                etype = self._emap.get(mtype.group(1), mtype.group(1))
                    # Fallback: look backward for MYSTRAN 'TYPE QUAD4' (older MYSTRAN format)
                    if etype not in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                                      'CPYRAM','CHEXA','CPENTA','CTETRA'):
                        for j in range(i-1, max(i-5, -1), -1):
                            dspace = re.sub(r'\s+', '', lines[j].strip()).upper()
                            mtype = re.search(r'TYPE([A-Z0-9]+)', dspace)
                            if mtype:
                                etype = self._emap.get(mtype.group(1), mtype.group(1))
                                break
                    # If STILL unknown, carry forward previous block's type
                    if etype not in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                                      'CPYRAM','CHEXA','CPENTA','CTETRA'):
                        if current_etype:
                            etype = current_etype

                # Valid type found — remember it for subsequent sub-blocks
                if etype in ('CQUAD4','CQUAD8','CTRIA3','CTRIA6',
                              'CPYRAM','CHEXA','CPENTA','CTETRA'):
                    current_etype = etype

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
            if re.search(r'S\s+T\s+R\s+E\s+S\s+S|F\s+O\s+R\s+C\s+E\s+S|SUBCASE', stripped):
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

    def _parse_gp_stress(self, lines, start, subcase, results):
        """Parse STRESSES AT GRID POINTS blocks.
        Format: GRID ID, ELEMENT ID, then Z1/Z2/MID rows with
        NORMAL-X, NORMAL-Y, SHEAR-XY, ANGLE, MAJOR, MINOR, MAX SHEAR, VON MISES.
        We store ElementGPStress keyed by (node_id, fiber) for each row.
        """
        if subcase not in results.gp_stresses:
            results.gp_stresses[subcase] = {}
        pending = {}
        last_nid = None
        i = start
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            # Check for markers BEFORE consuming the line.
            # This lets us put it back correctly if needed.
            if s:
                # MYSTRAN embedded SC marker: line starts with '0 ... SUBCASE N'
                if re.match(r'^0\s+SUBCASE\s+', s, re.I):
                    i -= 1; break
                # NASTRAN full-line OUTPUT FOR SUBCASE
                if re.search(r'O\s+U\s+T\s+P\s*U\s+T\s+F\s+O\s+R\s+S\s+U\s+B\s+C\s+A\s+S\s+E', s, re.I):
                    i -= 1; break
            i += 1
            if not s:
                continue
            if re.search(r'PAGE\s+\d+', s):
                continue
            # Break on next SURFACE block header
            if i > start + 1:
                if re.search(r'S\s+T\s+R\s+E\s+S\s+S\s+E\s+S\s+A\s+T\s+G\s+R\s+I\s+D\s+P\s+O\s+I\s+N\s+T\s+S', s, re.I):
                    i -= 1; break
            if s.startswith('GRID') or s.startswith('ELEMENT') or s.startswith('SURFACE'):
                continue

            # Determine fiber status from raw stripped line (before any lstrip)
            parts_raw = s.split()
            fiber0 = parts_raw[0].upper() if parts_raw else ''
            is_fiber_cont = fiber0 in ('Z1', 'Z2', 'MID')

            # Strip leading Fortran page-marker '0' only (before any non-space char)
            s_work = s.lstrip('0').strip()
            if not s_work and not is_fiber_cont:
                continue

            parts = s_work.split()

            if is_fiber_cont:
                if last_nid is None:
                    continue
                nid = last_nid
                fiber = fiber0
                vals = parts
            elif parts and parts[0].upper() in ('Z1', 'Z2', 'MID'):
                if last_nid is None:
                    continue
                nid = last_nid
                fiber = parts[0].upper()
                vals = parts
            elif parts and re.match(r'^[A-Z\s]+$', parts[0]) and not parts[0][0].isdigit():
                continue
            else:
                # New grid point: parts[0] must be integer GRID ID
                if len(parts) < 9:
                    continue
                try:
                    nid = int(parts[0])
                except ValueError:
                    continue
                last_nid = nid
                fiber = parts[2].upper() if len(parts) > 2 else 'Z1'
                if fiber not in ('Z1', 'Z2', 'MID'):
                    fiber = 'Z1'
                vals = parts[3:]

            nums = re.findall(r'[-+]?\d+\.\d+(?:E[+-]\d+)?', ' '.join(vals))
            if len(nums) < 6:
                continue
            nxx = float(nums[0]); nyy = float(nums[1])
            txy = float(nums[2])
            angle = float(nums[3])
            s1 = float(nums[4]); s2 = float(nums[5])
            vm = abs(float(nums[6])) if len(nums) > 6 else abs(s1)
            gp = ElementGPStress(
                subcase=subcase, node_id=nid,
                sxx=nxx, syy=nyy, txy=txy, angle=angle,
                s1=s1, s2=s2, s12=abs(float(nums[6])) if len(nums) > 6 else abs(txy), ovm=vm)
            key = (nid, fiber)
            pending.setdefault(nid, []).append(gp)
            results.gp_stresses[subcase][key] = gp

        if pending:
            print(f"  [GPSTRESS] SC{subcase}: {len(pending)} nodes parsed")
        return i

    def _parse_gp_force(self, lines, start, subcase, results):
        """Parse FORCES AT GRID POINTS blocks.
        Format: GRID ID, ELEMENT ID, then NXX/NYY/NXY/MXX/MYY/MXY/QX/QY.
        We store nodal force data under _gp_forces[subcase][node_id].
        """
        if subcase not in results._gp_forces:
            results._gp_forces[subcase] = {}
        nodal_acc = {}
        last_nid = None
        i = start
        while i < len(lines):
            line = lines[i]; i += 1
            s = line.strip()
            if not s:
                continue
            if re.search(r'PAGE\s+\d+', s):
                continue
            # Break on next GPFORCE header, MYSTRAN embedded '0 ... SUBCASE N',
            # or NASTRAN OUTPUT FOR SUBCASE.
            # Guard with i > start+1 so the first line (SURFACE X-AXIS info line)
            # which may coincidentally match the spaced header patterns does NOT
            # cause an early exit.
            if i > start + 1:
                if re.search(r'F\s+O\s+R\s+C\s+E\s+S\s+A\s+T\s+G\s+R\s+I\s+D\s+P\s+O\s+I\s+N\s+T\s+S', s, re.I):
                    i -= 1; break
                if re.search(r'^0\s+SUBCASE\s+', s, re.I):
                    i -= 1; break
                if re.search(r'O\s+U\s+T\s+P\s*U\s+T\s+F\s+O\s+R\s+S\s+U\s+B\s+C\s+A\s+S\s+E', s, re.I):
                    i -= 1; break
            # Skip header lines
            if s.startswith('GRID') or s.startswith('ELEMENT') or s.startswith('SURFACE'):
                continue
            if re.match(r'^[A-Z\s]+$', s) and not re.match(r'^[0-9]', s):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                nid = int(parts[0])
            except ValueError:
                continue
            last_nid = nid
            nums = re.findall(r'[-+]?\d+\.\d+(?:E[+-]\d+)?', s)
            if len(nums) < 8:
                continue
            nxx = float(nums[0]); nyy = float(nums[1]); nxy = float(nums[2])
            mxx = float(nums[3]); myy = float(nums[4]); mxy = float(nums[5])
            qx  = float(nums[6]); qy  = float(nums[7])
            nodal_acc[nid] = {'nxx': nxx, 'nyy': nyy, 'nxy': nxy,
                              'mxx': mxx, 'myy': myy, 'mxy': mxy,
                              'qx': qx, 'qy': qy}
            results._gp_forces[subcase][nid] = nodal_acc[nid]
        if nodal_acc:
            print(f"  [GPFORCE] SC{subcase}: {len(nodal_acc)} nodes parsed")
        return i

    def _parse_eng_forces(self, lines, start, subcase, results):
        """Parse ELEMENT ENGINEERING FORCES blocks.
        Format: eid + 8 floats (Nxx Nyy Nxy Mxx Myy Mxy Qx Qy).
        """
        if subcase not in results.forces:
            results.forces[subcase] = {}
        i = start
        count = 0
        while i < len(lines):
            line = lines[i]; i += 1
            s = line.strip()
            if not s:
                continue
            if re.search(r'PAGE\s+\d+', s):
                continue
            if re.search(r'S\s+T\s+R\s+E\s+S\s+S|F\s+O\s+R\s+C\s+E\s+S|'
                         r'E\s+L\s+E\s+M\s+E\s+N\s+T|SUBCASE|OUTPUT FOR', s):
                # Don't break on "FOR ELEMENT TYPE" sub-header
                if not re.match(r'F\s+O\s+R\s+E\s+L\s+E\s+M\s+E\s+N\s+T', s, re.I):
                    i -= 1; break
            if s.startswith('Element') or s.startswith('ID') or s.startswith('N o r m a l'):
                continue
            if re.match(r'F\s+O\s+R\s+E\s+L\s+E\s+M\s+E\s+N\s+T', s, re.I):
                continue
            if re.match(r'^[A-Z\s]+$', s) and not re.match(r'^[0-9]', s):
                continue
            if s.startswith('MAX') or s.startswith('MIN') or s.startswith('ABS') or s.startswith('*'):
                continue
            if s.startswith('---'):
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            try:
                eid = int(parts[0])
            except ValueError:
                continue
            try:
                nxx = _parse_float(parts[1]); nyy = _parse_float(parts[2]); nxy = _parse_float(parts[3])
                mxx = _parse_float(parts[4]); myy = _parse_float(parts[5]); mxy = _parse_float(parts[6])
                qx  = _parse_float(parts[7]); qy  = _parse_float(parts[8])
            except (IndexError, ValueError):
                continue
            results.forces[subcase][eid] = ElementForce(
                subcase=subcase, elem_id=eid, elem_type='CQUAD4',
                fx=nxx, fy=nyy, fxy=nxy, mx=mxx, my=myy, mxy=mxy, qx=qx, qy=qy)
            count += 1
        if count:
            print(f"  [EngForce] SC{subcase}: {count} elements parsed")
        return i




def _adapt_f06_rows(rows: 'F06Result', has_gp_forces: bool = True) -> F06Results:
    """Adapt F06Result (list-of-rows) to F06Results (Viewer dict format).

    MYSTRAN F06 stores one row per fiber location (CENTER-z1, CENTER-z2, GRD corner).
    We accumulate all rows per element and store only the max-von_mises row.
    """
    results = F06Results()
    results.subcases = sorted(set(d.subcase for d in rows.displacements))
    if not results.subcases:
        results.subcases = [1]

    for sc in results.subcases:
        results.displacements[sc] = {}
        results.stresses[sc] = {}
        results.forces[sc] = {}

    # ── Displacements ──────────────────────────────────────────────
    for d in rows.displacements:
        sc = d.subcase
        if sc not in results.displacements:
            results.displacements[sc] = {}
        results.displacements[sc][d.grid_id] = DisplacementResult(
            subcase=sc, node_id=d.grid_id,
            t1=d.t1, t2=d.t2, t3=d.t3,
            r1=d.r1, r2=d.r2, r3=d.r3)

    # ── Element stresses — accumulate per element, store max-vm row ─
    # rows.elem_stresses has one row per fiber location per element
    _acc = {}  # (sc, eid) -> list of ElemStressRow
    for s in rows.elem_stresses:
        key = (s.subcase, s.elem_id)
        _acc.setdefault(key, []).append(s)

    for (sc, eid), rows_list in _acc.items():
        if sc not in results.stresses:
            results.stresses[sc] = {}
        # Store the row with highest von_mises
        best = max(rows_list, key=lambda r: abs(r.von_mises))
        vm = abs(best.von_mises)
        results.stresses[sc][eid] = ElementStress(
            subcase=sc, elem_id=eid, elem_type=best.elem_type,
            values={'von_mises': vm,
                    'oxx': best.nx, 'oyy': best.ny, 'txy': best.sxy,
                    'omax': best.major, 'omin': best.minor},
            von_mises=vm)

    # ── GP stresses ────────────────────────────────────────────────
    for s in rows.grid_stresses:
        sc = s.subcase
        if sc not in results.gp_stresses:
            results.gp_stresses[sc] = {}
        key = (s.grid_id, s.fiber)
        results.gp_stresses[sc][key] = ElementGPStress(
            subcase=sc, node_id=s.grid_id,
            sxx=s.nx, syy=s.ny, txy=s.sxy,
            angle=s.angle, s1=s.major, s2=s.minor,
            s12=s.shear, ovm=abs(s.von_mises))

    # ── Element forces (from NASTRAN/NX reader, CENTER-only rows) ─
    # NASTRAN elem_forces has fx/fy/fxy; MYSTRAN has nxx/nyy/nxy.
    # Check which naming convention the reader uses.
    if hasattr(rows, 'elem_forces') and rows.elem_forces:
        sample = rows.elem_forces[0]
        use_naming = not hasattr(sample, 'fx')   # MYSTRAN uses nxx/nyy/nxy
        _ef_acc = {}  # (sc, eid) -> list
        for ef in rows.elem_forces:
            key = (ef.subcase, ef.elem_id)
            _ef_acc.setdefault(key, []).append(ef)
        for (sc, eid), ef_list in _ef_acc.items():
            if sc not in results.forces:
                results.forces[sc] = {}
            ef = ef_list[0]
            if use_naming:
                results.forces[sc][eid] = ElementForce(
                    subcase=sc, elem_id=eid, elem_type=ef.elem_type,
                    fx=ef.nxx, fy=ef.nyy, fxy=ef.nxy,
                    mx=ef.mxx, my=ef.myy, mxy=ef.mxy,
                    qx=ef.qx, qy=ef.qy)
            else:
                # Both readers use my_field (not my) for bending moment My
                my_val = getattr(ef, 'my', None) or getattr(ef, 'my_field', 0.0)
                results.forces[sc][eid] = ElementForce(
                    subcase=sc, elem_id=eid, elem_type=ef.elem_type,
                    fx=ef.fx, fy=ef.fy, fxy=ef.fxy,
                    mx=ef.mx, my=my_val, mxy=ef.mxy,
                    qx=ef.qx, qy=ef.qy)

    # ── GP forces (FORCES AT GRID POINTS) ─────────────────────────
    # Only MYSTRAN has this; NASTRAN does not.
    if has_gp_forces:
        for g in rows.grid_forces:
            sc = g.subcase
            if sc not in results._gp_forces:
                results._gp_forces[sc] = {}
            results._gp_forces[sc][g.grid_id] = {
                'nxx': g.nxx, 'nyy': g.nyy, 'nxy': g.nxy,
                'mxx': g.mxx, 'myy': g.myy, 'mxy': g.mxy,
                'qx': g.qx, 'qy': g.qy}

    ngp = sum(len(v) for v in results.gp_stresses.values())
    ngpf = sum(len(v) for v in results._gp_forces.values()) if has_gp_forces else 0
    extra = f"  gp_force_nodes={ngpf}" if has_gp_forces else ""
    print(f"[F06] subcases={results.subcases}  "
          f"disp_nodes={sum(len(v) for v in results.displacements.values())}  "
          f"elem_stresses={sum(len(v) for v in results.stresses.values())}  "
          f"gp_stress_entries={ngp}{extra}")
    return results


def _adapt_nx_f06_rows(rows: 'F06Result') -> F06Results:
    """Map validated NX rows to Viewer results, retaining every shell fiber.

    Element contours use the two fiber means. Corner rows are also retained
    for the Viewer's existing nodal averaging path. GPSTRESS comes from F06.
    """
    from collections import defaultdict

    results = F06Results()
    results.subcases = sorted({r.subcase for group in
        (rows.displacements, rows.grid_stresses, rows.elem_stresses, rows.elem_forces)
        for r in group})
    for sc in results.subcases:
        results.displacements[sc] = {}
        results.stresses[sc] = {}
        results.forces[sc] = {}

    for d in rows.displacements:
        results.displacements[d.subcase][d.grid_id] = DisplacementResult(
            d.subcase, d.grid_id, d.t1, d.t2, d.t3, d.r1, d.r2, d.r3)

    for s in rows.grid_stresses:
        results.gp_stresses.setdefault(s.subcase, {})[(s.grid_id, s.fiber)] = ElementGPStress(
            subcase=s.subcase, node_id=s.grid_id, sxx=s.nx, syy=s.ny,
            txy=s.sxy, angle=s.angle, s1=s.major, s2=s.minor,
            s12=s.shear, ovm=s.von_mises)

    groups = defaultdict(list)
    corners = defaultdict(lambda: defaultdict(list))
    components = {'oxx':'nx', 'oyy':'ny', 'txy':'sxy',
                  'omax':'major', 'omin':'minor', 'von_mises':'von_mises'}
    for s in rows.elem_stresses:
        groups[(s.subcase, s.elem_id)].append(s)
        if s.location == 'CORNER' and s.grid_id:
            corners[s.subcase][s.grid_id].append({
                'eid':s.elem_id, 'elem_type':s.elem_type,
                'fiber':'top' if s.fiber_dist >= 0 else 'bottom',
                'fiber_dist':s.fiber_dist,
                **{name:float(getattr(s, attr)) for name,attr in components.items()}})

    for (sc,eid), samples in groups.items():
        top=[s for s in samples if s.fiber_dist >= 0]
        bottom=[s for s in samples if s.fiber_dist < 0]
        if not top or not bottom:
            raise ValueError(f'NX shell {eid} SC{sc} lacks one fiber side')
        def means(layer):
            return {name:float(np.mean([getattr(s,attr) for s in layer]))
                    for name,attr in components.items()}
        t=means(top);b=means(bottom)
        mid={name:0.5*(t[name]+b[name]) for name in components}
        values=dict(mid)
        for name in components:
            values[name+'_top']=t[name]
            values[name+'_bottom']=b[name]
        results.stresses[sc][eid]=ElementStress(
            subcase=sc, elem_id=eid, elem_type=samples[0].elem_type,
            values=values, von_mises=mid['von_mises'])
    for sc,by_node in corners.items():
        results.stresses[sc]['_shell_corner_contribs']=dict(by_node)

    # The viewer's element force contours use center values. NX corner
    # force rows remain in the dedicated reader for per-location analysis.
    for f in rows.elem_forces:
        if f.location != 'CENTER':
            continue
        results.forces[f.subcase][f.elem_id]=ElementForce(
            subcase=f.subcase, elem_id=f.elem_id, elem_type=f.elem_type,
            fx=f.fx, fy=f.fy, fxy=f.fxy,
            mx=f.mx, my=f.my_field, mxy=f.mxy, qx=f.qx, qy=f.qy)

    print(f"[F06 NX] subcases={results.subcases}  "
          f"disp_nodes={sum(len(v) for v in results.displacements.values())}  "
          f"elem_stresses={len(groups)}  "
          f"gp_stress_entries={sum(len(v) for v in results.gp_stresses.values())}  "
          f"elem_forces={sum(len(v) for v in results.forces.values())}")
    return results


def _detect_f06_format(filepath: str) -> str:
    """Detect F06 format: MYSTRAN, NX_NASTRAN, or NASTRAN from file header."""
    with open(filepath, 'r', errors='replace') as f:
        for line in f:
            ul = line.upper()
            if 'MYSTRAN BEGIN' in ul:
                return 'MYSTRAN'
            if 'NASTRAN' in ul or 'SIMCENTER NASTRAN' in ul:
                return 'NX_NASTRAN'
    return 'NASTRAN'


def load_f06(filepath: str) -> F06Results:
    """Parse a NASTRAN/MYSTRAN F06 output file.

    Detects format from file header and routes to the appropriate reader:
      MYSTRAN     → mystran_f06_reader.py  (has GP forces)
      NASTRAN     → nxnastran_f06_codex.py (CENTER, CORNER, GPSTRESS)
    """
    fmt = _detect_f06_format(filepath)
    if fmt == 'MYSTRAN':
        from parser.mystran_f06_reader import parse_f06
        rows = parse_f06(filepath)
        return _adapt_f06_rows(rows)
    else:
        # NX/NASTRAN-like route uses the verified SC1/SC2 reader and keeps
        # its local center/corner, top/bottom and F06-only GPSTRESS data.
        from parser.nxnastran_f06_codex import parse_f06
        rows = parse_f06(filepath)
        return _adapt_nx_f06_rows(rows)
