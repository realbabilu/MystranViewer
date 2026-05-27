"""
MYSTRAN .dat file parser
Supports: GRID, CBAR, CBEAM, CROD, CQUAD4, CTRIA3, CHEXA, CPENTA, CTETRA
          SPC, FORCE, MOMENT, MAT1, PBAR, PSHELL, PSOLID, etc.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Node:
    id: int
    xyz: np.ndarray  # shape (3,)
    cp: int = 0      # coord system


@dataclass
class Element:
    id: int
    type: str        # 'CBAR','CBEAM','CROD','CQUAD4','CTRIA3','CHEXA','CPENTA','CTETRA'
    nodes: List[int] = field(default_factory=list)
    pid: int = 0     # property id
    # CBAR/CBEAM: orientation vector (v-vector, global coords) or None
    v_orient: Optional[np.ndarray] = None
    g0_ref: int = 0   # optional orientation reference GRID for CBAR/CBEAM
    pa: str = ""     # released DOFs at end A, e.g. '56'
    pb: str = ""     # released DOFs at end B


@dataclass
class RigidElement:
    id: int
    type: str        # 'RBE2' or 'RBE3'
    ref_grid: int = 0
    ref_comp: str = ""
    dep_grids: List[int] = field(default_factory=list)


@dataclass
class SPC:
    id: int
    node_id: int
    dofs: str        # e.g. '123456'
    values: List[float] = field(default_factory=list)


@dataclass
class Force:
    sid: int
    node_id: int
    cid: int
    magnitude: float
    direction: np.ndarray  # shape (3,)


@dataclass
class Moment:
    sid: int
    node_id: int
    cid: int
    magnitude: float
    direction: np.ndarray


@dataclass
class Material:
    id: int
    type: str  # 'MAT1'
    E: float = 0.0
    G: float = 0.0
    nu: float = 0.0
    rho: float = 0.0


@dataclass
class Property:
    id: int
    type: str  # 'PBAR','PBEAM','PROD','PSHELL','PSOLID','PBARL','PBEAML'
    mid: int = 0
    params: Dict = field(default_factory=dict)
    # Parsed section (populated after parse)
    section: 'Optional[object]' = None   # BeamSection or None


@dataclass
class MystranModel:
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    spcs: List[SPC] = field(default_factory=list)
    forces: List[Force] = field(default_factory=list)
    moments: List[Moment] = field(default_factory=list)
    materials: Dict[int, Material] = field(default_factory=dict)
    properties: Dict[int, Property] = field(default_factory=dict)
    rigids: List[RigidElement] = field(default_factory=list)
    title: str = ""
    pload4s: "List[object]" = field(default_factory=list)
    pload2s: "List[object]" = field(default_factory=list)
    pload1s: "List[object]" = field(default_factory=list)
    subcase_loads: Dict[int, int] = field(default_factory=dict)
    subcase_spcs: Dict[int, int] = field(default_factory=dict)

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box."""
        if not self.nodes:
            return np.zeros(3), np.ones(3)
        pts = np.array([n.xyz for n in self.nodes.values()])
        return pts.min(axis=0), pts.max(axis=0)

    def center(self) -> np.ndarray:
        mn, mx = self.bbox()
        return (mn + mx) * 0.5

    def scale(self) -> float:
        mn, mx = self.bbox()
        d = np.linalg.norm(mx - mn)
        return d if d > 1e-12 else 1.0

    def elements_by_dim(self):
        """Return dicts split by dimensionality."""
        e1d = {k: v for k, v in self.elements.items()
               if v.type in ('CBAR', 'CBEAM', 'CROD')}
        e2d = {k: v for k, v in self.elements.items()
               if v.type in ('CQUAD4', 'CTRIA3')}
        e3d = {k: v for k, v in self.elements.items()
               if v.type in ('CHEXA', 'CPENTA', 'CTETRA', 'CPYRAM')}
        return e1d, e2d, e3d


# ---------------------------------------------------------------------------
# Bulk-data field parsing helpers
# ---------------------------------------------------------------------------

def _split_free(line: str) -> List[str]:
    """Free-field: comma-separated."""
    return [f.strip() for f in line.split(',')]


def _split_small(line: str) -> List[str]:
    """Small-field: 8-char columns. Read full line (may exceed 80 chars for CHEXA etc)."""
    fields = []
    # Read up to 11 fields (88 chars) to handle 8-node elements on one line
    end = max(88, len(line) + 8)
    padded = line.ljust(end)
    for i in range(0, end, 8):
        s = padded[i:i+8].strip()
        if i < len(line) or s:   # include fields within line length
            fields.append(s)
        if i >= 80 and not s:    # stop at first empty field past col 80
            break
    return fields


def _split_large(line: str) -> List[str]:
    """Large-field: col0=card*(8), col1..4=fields(16), col5=cont*(8)."""
    fields = [line[0:8].strip()]
    for i in range(8, 72, 16):
        fields.append(line[i:i+16].strip())
    fields.append(line[72:80].strip())
    return fields


def _flt(s: str) -> float:
    """Parse NASTRAN-style float (handles 1.5E+3, 1.5+3, 1.5-3)."""
    if not s:
        return 0.0
    s = s.strip()
    if not s:
        return 0.0
    # Handle NASTRAN shorthand: '1.5+3' → '1.5e+3'
    import re
    s = re.sub(r'([0-9])([+-])([0-9])', r'\1e\2\3', s)
    try:
        return float(s)
    except ValueError:
        return 0.0


def _int(s: str) -> int:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return 0


def _is_plain_int_token(s: str) -> bool:
    tok = str(s).strip()
    if not tok:
        return False
    if any(ch in tok for ch in '.EeDd'):
        return False
    return tok.lstrip('+-').isdigit()


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class DatParser:
    def __init__(self):
        self.model = MystranModel()

    def parse(self, filepath: str) -> MystranModel:
        with open(filepath, 'r', errors='replace') as f:
            raw_lines = f.readlines()

        # Normalize + strip comments
        lines = []
        for line in raw_lines:
            line = line.rstrip('\n').rstrip('\r')
            # Strip $ comments
            if '$' in line:
                line = line[:line.index('$')]
            lines.append(line)

        # Collect title from TITLE= or first non-blank line before BEGIN BULK
        in_bulk = False
        bulk_lines = []
        case_lines = []

        for i, line in enumerate(lines):
            up = line.upper().strip()
            if up.startswith('TITLE'):
                self.model.title = line.split('=', 1)[-1].strip() if '=' in line else up
            if up.startswith('BEGIN BULK') or up == 'BEGIN':
                in_bulk = True
                continue
            if up.startswith('ENDDATA') or up.startswith('END DATA'):
                break
            if in_bulk:
                bulk_lines.append(line)
            else:
                case_lines.append(line)

        self._parse_case_control(case_lines)

        # If no BEGIN BULK found, treat whole file as bulk
        if not bulk_lines:
            bulk_lines = lines

        # Continuation joining
        cards = self._join_continuations(bulk_lines)

        # Dispatch
        for card_fields in cards:
            if not card_fields:
                continue
            name = card_fields[0].upper().rstrip('*').strip()
            self._dispatch(name, card_fields)

        self._resolve_beam_g0_orientations()

        return self.model

    def _parse_case_control(self, lines: List[str]):
        current_subcase = 1
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('SUBCASE'):
                parts = up.split()
                if len(parts) >= 2:
                    try:
                        current_subcase = int(parts[1])
                        self.model.subcase_loads.setdefault(current_subcase, 0)
                        self.model.subcase_spcs.setdefault(current_subcase, 0)
                    except ValueError:
                        pass
                continue
            if up.startswith('LOAD') and '=' in up:
                rhs = line.split('=', 1)[1].strip()
                try:
                    self.model.subcase_loads[current_subcase] = int(rhs.split()[0])
                except ValueError:
                    pass
            if up.startswith('SPC') and '=' in up:
                rhs = line.split('=', 1)[1].strip()
                try:
                    self.model.subcase_spcs[current_subcase] = int(rhs.split()[0])
                except ValueError:
                    pass

    def _join_continuations(self, lines: List[str]) -> List[List[str]]:
        """
        Join continuation lines and split into field lists.
        Handles small-field (8-char), large-field (*), and free-field (,).
        """
        cards = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue

            is_free = ',' in line
            is_large = line.startswith('*') or (len(line) > 0 and line[0] == '*')

            if is_free:
                fields = _split_free(line)
            elif is_large or (len(line) >= 1 and line[0] == '*'):
                fields = _split_large(line)
            else:
                fields = _split_small(line)

            # Look ahead for continuations
            while i + 1 < len(lines):
                nxt = lines[i + 1]
                if not nxt.strip():
                    break
                # Continuation marker: starts with + or * or has +/blank in col 0
                is_cont_free = ',' in nxt and nxt.strip().startswith('+')
                is_cont_large = nxt.startswith('*')
                is_cont_small = (len(nxt) >= 1 and nxt[0] in ('+', ' ', '\t')) and nxt.strip()

                if is_cont_free or is_cont_large or is_cont_small:
                    # Check it's not a new card
                    if not nxt.strip()[0].isalpha() and nxt.strip()[0] not in ('*', '+'):
                        break
                    if nxt.strip()[0].isalpha() and not nxt.strip().startswith('+'):
                        # Could be new card - check col 0
                        if nxt[0] != ' ' and nxt[0] != '+' and nxt[0] != '*':
                            break

                    i += 1
                    if ',' in nxt:
                        ext = _split_free(nxt)[1:]  # skip continuation marker
                    elif nxt.startswith('*'):
                        ext = _split_large(nxt)[1:]
                    else:
                        ext = _split_small(nxt)[1:]
                    fields.extend(ext)
                else:
                    break

            cards.append(fields)
            i += 1

        return cards

    def _dispatch(self, name: str, f: List[str]):
        """Route card to appropriate handler."""
        # Pad fields to avoid index errors
        while len(f) < 20:
            f.append('')

        if name == 'GRID':
            self._parse_grid(f)
        elif name in ('CBAR', 'CBEAM', 'CROD'):
            self._parse_1d(name, f)
        elif name in ('RBE2', 'RBE3'):
            self._parse_rigid(name, f)
        elif name in ('CQUAD4','CQUADR'):
            self._parse_cquad4(f)
        elif name in ('CTRIA3','CTRIAR'):
            self._parse_ctria3(f)
        elif name in ('CHEXA','CHEXA8'):
            self._parse_chexa(f)
        elif name in ('CPYRAM','CPYRA5','CPYRA'):
            self._parse_cpyram(f)
        elif name == 'CPENTA':
            self._parse_cpenta(f)
        elif name == 'CTETRA':
            self._parse_ctetra(f)
        elif name == 'SPC':
            self._parse_spc(f)
        elif name == 'SPC1':
            self._parse_spc1(f)
        elif name == 'FORCE':
            self._parse_force(f)
        elif name == 'MOMENT':
            self._parse_moment(f)
        elif name == 'MAT1':
            self._parse_mat1(f)
        elif name in ('PBAR', 'PBEAM', 'PROD', 'PSHELL', 'PSOLID', 'PBARL', 'PBEAML'):
            self._parse_property(name, f)
        elif name == 'PLOAD4':
            self._parse_pload4(f)
        elif name == 'PLOAD1':
            self._parse_pload1(f)
        elif name in ('PLOAD2', 'PLOAD'):
            self._parse_pload2(name, f)

    def _parse_grid(self, f):
        nid = _int(f[1])
        cp  = _int(f[2])
        x   = _flt(f[3])
        y   = _flt(f[4])
        z   = _flt(f[5])
        self.model.nodes[nid] = Node(id=nid, xyz=np.array([x, y, z]), cp=cp)

    def _parse_1d(self, etype, f):
        eid  = _int(f[1])
        pid  = _int(f[2])
        n1   = _int(f[3])
        n2   = _int(f[4])
        # CBAR/CBEAM: fields 5,6,7 = x1,x2,x3 of v-vector (or g0 node ref)
        v_orient = None
        g0_ref = 0
        if etype in ('CBAR', 'CBEAM') and len(f) > 7:
            t5 = str(f[5]).strip()
            t6 = str(f[6]).strip() if len(f) > 6 else ""
            t7 = str(f[7]).strip() if len(f) > 7 else ""
            if _is_plain_int_token(t5) and not t6 and not t7:
                g0_ref = _int(t5)
            else:
                x1 = _flt(f[5]); x2 = _flt(f[6]); x3 = _flt(f[7])
                if abs(x1)+abs(x2)+abs(x3) > 1e-12:
                    v_orient = np.array([x1, x2, x3], dtype=np.float32)
        pa = ""
        pb = ""
        if etype in ('CBAR', 'CBEAM'):
            tail = [str(x).strip() for x in f[8:] if str(x).strip()]
            digit_fields = [tok for tok in tail if tok.isdigit() and len(tok) <= 6]
            if digit_fields:
                pa = digit_fields[0]
                if len(digit_fields) > 1:
                    pb = digit_fields[1]
        self.model.elements[eid] = Element(id=eid, type=etype, nodes=[n1, n2],
                                           pid=pid, v_orient=v_orient, g0_ref=g0_ref,
                                           pa=pa, pb=pb)

    def _resolve_beam_g0_orientations(self):
        for elem in self.model.elements.values():
            if elem.type not in ('CBAR', 'CBEAM'):
                continue
            if elem.v_orient is not None or getattr(elem, 'g0_ref', 0) <= 0 or not elem.nodes:
                continue
            ga = self.model.nodes.get(elem.nodes[0])
            g0 = self.model.nodes.get(elem.g0_ref)
            if ga is None or g0 is None:
                continue
            vec = np.asarray(g0.xyz - ga.xyz, dtype=np.float32)
            nrm = float(np.linalg.norm(vec))
            if nrm <= 1e-12:
                continue
            elem.v_orient = vec / nrm

    def _parse_rigid(self, rtype, f):
        eid = _int(f[1])
        if rtype == 'RBE2':
            ref_grid = _int(f[2])
            ref_comp = str(f[3]).strip()
            dep = []
            for tok in f[4:]:
                if str(tok).strip():
                    nid = _int(tok)
                    if nid > 0:
                        dep.append(nid)
            self.model.rigids.append(RigidElement(
                id=eid, type='RBE2', ref_grid=ref_grid, ref_comp=ref_comp, dep_grids=dep))
        elif rtype == 'RBE3':
            ref_grid = _int(f[3]) if len(f) > 3 else 0
            ref_comp = str(f[4]).strip() if len(f) > 4 else ""
            dep = []
            for tok in f[5:]:
                if str(tok).strip():
                    nid = _int(tok)
                    if nid > 0:
                        dep.append(nid)
            self.model.rigids.append(RigidElement(
                id=eid, type='RBE3', ref_grid=ref_grid, ref_comp=ref_comp, dep_grids=dep))

    def _parse_cquad4(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[3]), _int(f[4]), _int(f[5]), _int(f[6])]
        self.model.elements[eid] = Element(id=eid, type='CQUAD4', nodes=ns, pid=pid)

    def _parse_ctria3(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[3]), _int(f[4]), _int(f[5])]
        self.model.elements[eid] = Element(id=eid, type='CTRIA3', nodes=ns, pid=pid)

    def _parse_cpyram(self, f):
        """CPYRAM: 5-node pyramid. Nodes 1-4=base quad, node 5=apex."""
        try:
            eid=_int(f[1]); pid=_int(f[2])
            ns=[_int(f[i]) for i in range(3,8) if i<len(f) and f[i].strip()]
            self.model.elements[eid]=Element(id=eid,type='CPYRAM',nodes=ns,pid=pid)
        except Exception: pass

    def _parse_chexa(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[i]) for i in range(3, 11)]
        self.model.elements[eid] = Element(id=eid, type='CHEXA', nodes=ns, pid=pid)

    def _parse_cpenta(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[i]) for i in range(3, 9)]
        self.model.elements[eid] = Element(id=eid, type='CPENTA', nodes=ns, pid=pid)

    def _parse_ctetra(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[i]) for i in range(3, 7)]
        self.model.elements[eid] = Element(id=eid, type='CTETRA', nodes=ns, pid=pid)

    def _parse_spc(self, f):
        sid  = _int(f[1])
        nid  = _int(f[2])
        dofs = f[3].strip()
        val  = _flt(f[4])
        self.model.spcs.append(SPC(id=sid, node_id=nid, dofs=dofs, values=[val]))

    def _parse_spc1(self, f):
        sid  = _int(f[1])
        dofs = f[2].strip()
        # remaining fields are node ids
        for i in range(3, len(f)):
            if f[i].strip():
                nid = _int(f[i])
                if nid > 0:
                    self.model.spcs.append(SPC(id=sid, node_id=nid, dofs=dofs))

    def _parse_force(self, f):
        sid = _int(f[1]); nid = _int(f[2]); cid = _int(f[3])
        mag = _flt(f[4])
        d   = np.array([_flt(f[5]), _flt(f[6]), _flt(f[7])])
        self.model.forces.append(Force(sid=sid, node_id=nid, cid=cid, magnitude=mag, direction=d))

    def _parse_moment(self, f):
        sid = _int(f[1]); nid = _int(f[2]); cid = _int(f[3])
        mag = _flt(f[4])
        d   = np.array([_flt(f[5]), _flt(f[6]), _flt(f[7])])
        self.model.moments.append(Moment(sid=sid, node_id=nid, cid=cid, magnitude=mag, direction=d))

    def _parse_pload1(self, f):
        # PLOAD1: sid, eid, type, scale, x1, p1, [x2, p2]
        # type: FX/FY/FZ/FXE/FYE/FZE = force; MX/MY/MZ = moment
        try:
            sid=int(f[1].strip() or 0); eid=int(f[2].strip() or 0)
            ltype=f[3].strip(); scale=f[4].strip()
            x1=float(f[5].strip() or 0); p1=float(f[6].strip() or 0)
            x2=float(f[7].strip() or 0) if len(f)>7 and f[7].strip() else x1
            p2=float(f[8].strip() or p1) if len(f)>8 and f[8].strip() else p1
            self.model.pload1s.append({'sid':sid,'eid':eid,'type':ltype,
                                        'scale':scale,'x1':x1,'p1':p1,'x2':x2,'p2':p2})
        except: pass

    def _parse_pload4(self, f):
        from dataclasses import dataclass, field as dc_field
        sid=int(f[1].strip() or 0); eid=int(f[2].strip() or 0)
        try: p=float(f[3].strip() or 0)
        except: p=0.0
        self.model.pload4s.append({'sid':sid,'eid':eid,'pressure':p})

    def _parse_pload2(self, name, f):
        sid=int(f[1].strip() or 0)
        try: p=float(f[2].strip() or 0)
        except: p=0.0
        eids=[int(f[i].strip()) for i in range(3,len(f)) 
              if f[i].strip() and f[i].strip().lstrip('-').isdigit() and int(f[i].strip())>0]
        self.model.pload2s.append({'sid':sid,'pressure':p,'eids':eids})

    def _parse_mat1(self, f):
        mid = _int(f[1])
        E   = _flt(f[2]); G = _flt(f[3]); nu = _flt(f[4]); rho = _flt(f[5])
        self.model.materials[mid] = Material(id=mid, type='MAT1', E=E, G=G, nu=nu, rho=rho)

    def _parse_property(self, ptype, f):
        pid = _int(f[1])
        mid = _int(f[2]) if len(f) > 2 and f[2].strip() else 0
        params = {f'f{i}': f[i] for i in range(3, len(f)) if f[i].strip()}
        prop = Property(id=pid, type=ptype, mid=mid, params=params)
        # Resolve section immediately
        try:
            from parser.beam_section import extract_section
            if ptype in ('PBAR','PBEAM','PROD','PBARL','PBEAML'):
                prop.section = extract_section(ptype, mid, params)
        except Exception:
            pass
        self.model.properties[pid] = prop


def load_dat(filepath: str) -> MystranModel:
    return DatParser().parse(filepath)


# ---------------------------------------------------------------------------
# Distributed / pressure loads
# ---------------------------------------------------------------------------

@dataclass
class PLoad4:
    """Pressure load on shell face (PLOAD4)."""
    sid:      int
    eid:      int         # element id
    pressure: float
    # optional face nodes
    nodes: List[int] = field(default_factory=list)

@dataclass  
class PLoad2:
    """Pressure on shell element (PLOAD2)."""
    sid:      int
    pressure: float
    eids:     List[int] = field(default_factory=list)
