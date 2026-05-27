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
    type: str  # 'PBAR','PBEAM','PROD','PSHELL','PSOLID'
    mid: int = 0
    params: Dict = field(default_factory=dict)


@dataclass
class MystranModel:
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    spcs: List[SPC] = field(default_factory=list)
    forces: List[Force] = field(default_factory=list)
    moments: List[Moment] = field(default_factory=list)
    materials: Dict[int, Material] = field(default_factory=dict)
    properties: Dict[int, Property] = field(default_factory=dict)
    title: str = ""

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
               if v.type in ('CHEXA', 'CPENTA', 'CTETRA')}
        return e1d, e2d, e3d


# ---------------------------------------------------------------------------
# Bulk-data field parsing helpers
# ---------------------------------------------------------------------------

def _split_free(line: str) -> List[str]:
    """Free-field: comma-separated."""
    return [f.strip() for f in line.split(',')]


def _split_small(line: str) -> List[str]:
    """Small-field: 8-char columns. col0=card(8), col1..8=fields(8), col9=cont(8)."""
    fields = []
    for i in range(0, 80, 8):
        fields.append(line[i:i+8].strip())
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

        return self.model

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
        elif name == 'CQUAD4':
            self._parse_cquad4(f)
        elif name == 'CTRIA3':
            self._parse_ctria3(f)
        elif name == 'CHEXA':
            self._parse_chexa(f)
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
        elif name in ('PBAR', 'PBEAM', 'PROD', 'PSHELL', 'PSOLID', 'PBARL'):
            self._parse_property(name, f)

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
        self.model.elements[eid] = Element(id=eid, type=etype, nodes=[n1, n2], pid=pid)

    def _parse_cquad4(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[3]), _int(f[4]), _int(f[5]), _int(f[6])]
        self.model.elements[eid] = Element(id=eid, type='CQUAD4', nodes=ns, pid=pid)

    def _parse_ctria3(self, f):
        eid = _int(f[1]); pid = _int(f[2])
        ns  = [_int(f[3]), _int(f[4]), _int(f[5])]
        self.model.elements[eid] = Element(id=eid, type='CTRIA3', nodes=ns, pid=pid)

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

    def _parse_mat1(self, f):
        mid = _int(f[1])
        E   = _flt(f[2]); G = _flt(f[3]); nu = _flt(f[4]); rho = _flt(f[5])
        self.model.materials[mid] = Material(id=mid, type='MAT1', E=E, G=G, nu=nu, rho=rho)

    def _parse_property(self, ptype, f):
        pid = _int(f[1]); mid = _int(f[2]) if len(f) > 2 else 0
        params = {f'f{i}': f[i] for i in range(3, len(f)) if f[i].strip()}
        self.model.properties[pid] = Property(id=pid, type=ptype, mid=mid, params=params)


def load_dat(filepath: str) -> MystranModel:
    return DatParser().parse(filepath)
