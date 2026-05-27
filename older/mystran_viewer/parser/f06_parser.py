"""
MYSTRAN .f06 output parser
Extracts: displacements, SPC reactions, element stresses/forces
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DisplacementResult:
    subcase: int
    node_id: int
    t1: float; t2: float; t3: float   # translational
    r1: float; r2: float; r3: float   # rotational

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.t1, self.t2, self.t3])

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.translation))


@dataclass
class ElementStress:
    subcase: int
    elem_id: int
    elem_type: str
    values: Dict[str, float] = field(default_factory=dict)   # e.g. {'sx':..., 'sy':..., 'txy':...}
    von_mises: float = 0.0


@dataclass
class F06Results:
    subcases: List[int] = field(default_factory=list)
    # keyed by subcase → node_id → DisplacementResult
    displacements: Dict[int, Dict[int, DisplacementResult]] = field(default_factory=dict)
    # keyed by subcase → elem_id → ElementStress
    stresses: Dict[int, Dict[int, ElementStress]] = field(default_factory=dict)

    def max_displacement(self, subcase: int) -> float:
        if subcase not in self.displacements:
            return 0.0
        vals = [d.magnitude for d in self.displacements[subcase].values()]
        return max(vals) if vals else 0.0

    def displacement_array(self, subcase: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (node_ids, magnitudes) arrays sorted by node_id."""
        if subcase not in self.displacements:
            return np.array([]), np.array([])
        items = sorted(self.displacements[subcase].items())
        ids  = np.array([i for i, _ in items])
        mags = np.array([d.magnitude for _, d in items])
        return ids, mags

    def von_mises_array(self, subcase: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (elem_ids, von_mises) arrays."""
        if subcase not in self.stresses:
            return np.array([]), np.array([])
        items = sorted(self.stresses[subcase].items())
        ids = np.array([i for i, _ in items])
        vm  = np.array([s.von_mises for _, s in items])
        return ids, vm


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class F06Parser:
    # Section header patterns
    _RE_SUBCASE   = re.compile(r'SUBCASE\s+(\d+)', re.IGNORECASE)
    _RE_DISP_HDR  = re.compile(r'D\s*I\s*S\s*P\s*L\s*A\s*C\s*E\s*M\s*E\s*N\s*T', re.IGNORECASE)
    _RE_STRESS_2D = re.compile(r'S\s*T\s*R\s*E\s*S\s*S\s*E\s*S\s+IN\s+(QUAD4|TRIA3)', re.IGNORECASE)
    _RE_STRESS_1D = re.compile(r'S\s*T\s*R\s*E\s*S\s*S\s*E\s*S\s+IN\s+(BAR|BEAM|ROD)', re.IGNORECASE)
    _RE_FLOAT     = re.compile(r'[+-]?\d+\.?\d*[EeDd]?[+-]?\d*')

    def parse(self, filepath: str) -> F06Results:
        results = F06Results()

        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()

        current_subcase = 1
        i = 0

        while i < len(lines):
            line = lines[i]

            # Track subcase
            m = self._RE_SUBCASE.search(line)
            if m:
                current_subcase = int(m.group(1))
                if current_subcase not in results.subcases:
                    results.subcases.append(current_subcase)
                    results.displacements[current_subcase] = {}
                    results.stresses[current_subcase] = {}

            # Displacement table
            if self._RE_DISP_HDR.search(line):
                i = self._parse_displacements(lines, i, current_subcase, results)
                continue

            # 2D stress table
            m2 = self._RE_STRESS_2D.search(line)
            if m2:
                etype = m2.group(1).upper()
                i = self._parse_stress_2d(lines, i, current_subcase, etype, results)
                continue

            i += 1

        # Ensure subcase 1 always exists
        if not results.subcases:
            results.subcases = [1]
            results.displacements[1] = {}
            results.stresses[1] = {}

        return results

    def _parse_displacements(self, lines, start, subcase, results):
        """Parse displacement block. Returns next line index."""
        i = start + 1
        # Skip header lines until data (lines with integer node IDs)
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            # End of block: page break or new section header
            if '---' in line or stripped.startswith('1 ') or stripped.startswith('0 '):
                break
            # Try parse data line: node_id  type  t1  t2  t3  r1  r2  r3
            parts = stripped.split()
            if len(parts) >= 7:
                try:
                    nid = int(parts[0])
                    # parts[1] is typically 'G' or type code - skip
                    offset = 1 if not self._is_float(parts[1]) else 0
                    vals = [self._parse_float(parts[j]) for j in range(offset+1, offset+7)]
                    if len(vals) == 6:
                        d = DisplacementResult(
                            subcase=subcase, node_id=nid,
                            t1=vals[0], t2=vals[1], t3=vals[2],
                            r1=vals[3], r2=vals[4], r3=vals[5]
                        )
                        results.displacements[subcase][nid] = d
                except (ValueError, IndexError):
                    pass
            i += 1
        return i

    def _parse_stress_2d(self, lines, start, subcase, etype, results):
        """Parse 2D stress block."""
        i = start + 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            if '---' in line or stripped.startswith('1 '):
                break
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    eid = int(parts[0])
                    floats = []
                    for p in parts[1:]:
                        try:
                            floats.append(self._parse_float(p))
                        except ValueError:
                            pass
                    if floats:
                        # Try to extract von Mises: typically last meaningful value
                        vm = floats[-1] if floats else 0.0
                        # For QUAD4/TRIA3: sx sy txy angle vm max_shear
                        stress_dict = {}
                        keys = ['fiber_dist', 'sx', 'sy', 'txy', 'angle', 'major', 'minor', 'max_shear']
                        for ki, kv in zip(keys, floats):
                            stress_dict[ki] = kv
                        # Von Mises from principal stresses if available
                        s1 = stress_dict.get('major', 0.0)
                        s2 = stress_dict.get('minor', 0.0)
                        vm_calc = np.sqrt(s1**2 - s1*s2 + s2**2)
                        results.stresses[subcase][eid] = ElementStress(
                            subcase=subcase, elem_id=eid, elem_type=etype,
                            values=stress_dict, von_mises=vm_calc
                        )
                except (ValueError, IndexError):
                    pass
            i += 1
        return i

    def _is_float(self, s: str) -> bool:
        try:
            float(s.replace('D', 'E').replace('d', 'e'))
            return True
        except ValueError:
            return False

    def _parse_float(self, s: str) -> float:
        s = s.strip().replace('D', 'E').replace('d', 'e')
        # Handle NASTRAN shorthand: 1.5+3 → 1.5e+3
        s = re.sub(r'([0-9])([+-])([0-9])', r'\1e\2\3', s)
        return float(s)


def load_f06(filepath: str) -> F06Results:
    return F06Parser().parse(filepath)
