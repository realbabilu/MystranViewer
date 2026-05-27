"""
OP2 results parser using pyNastran.
Reads displacements, element stresses (von Mises), SPC forces.
Falls back gracefully if pyNastran not installed.

Also includes NEU (Femap neutral file) reader as alternative.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Reuse F06Results dataclass structure for compatibility
from parser.f06_parser import F06Results, DisplacementResult, ElementStress


# ---------------------------------------------------------------------------
# OP2 Parser via pyNastran
# ---------------------------------------------------------------------------

class OP2Parser:
    def parse(self, filepath: str) -> F06Results:
        try:
            from pyNastran.op2.op2 import OP2
        except ImportError:
            raise ImportError(
                "pyNastran not installed. Run: pip install pyNastran")

        op2 = OP2(debug=False)
        op2.read_op2(filepath, combine=True)

        results = F06Results()

        # ── Subcases ─────────────────────────────────────────────────
        subcases = set()
        if op2.displacements:
            subcases.update(op2.displacements.keys())
        if op2.eigenvectors:
            subcases.update(op2.eigenvectors.keys())
        subcases = sorted(subcases) or [1]
        results.subcases = list(subcases)
        for sc in results.subcases:
            results.displacements[sc] = {}
            results.stresses[sc]      = {}

        # ── Displacements ────────────────────────────────────────────
        self._read_displacements(op2.displacements, results)

        # Eigenvectors (modal) — treat as displacement subcases
        self._read_displacements(op2.eigenvectors, results)

        # ── Element stresses ─────────────────────────────────────────
        # 2D shells
        for attr in ('cquad4_stress','ctria3_stress',
                     'cquad8_stress','ctria6_stress','cquadr_stress'):
            table = getattr(op2, attr, None)
            if table:
                self._read_plate_stress(table, results, attr.split('_')[0].upper())

        # 1D bars/beams/rods
        for attr in ('crod_stress','conrod_stress'):
            table = getattr(op2, attr, None)
            if table:
                self._read_1d_stress(table, results, attr.split('_')[0].upper())

        # 3D solids
        for attr in ('chexa_stress','cpenta_stress','ctetra_stress'):
            table = getattr(op2, attr, None)
            if table:
                self._read_solid_stress(table, results, attr.split('_')[0].upper())

        print(f"[OP2] subcases={results.subcases}")
        for sc in results.subcases:
            nd = len(results.displacements.get(sc, {}))
            ns = len(results.stresses.get(sc, {}))
            print(f"  SC {sc}: {nd} node displacements, {ns} element stresses")

        return results

    # ----------------------------------------------------------------
    def _read_displacements(self, table_dict, results: F06Results):
        if not table_dict:
            return
        for isubcase, res in table_dict.items():
            if isubcase not in results.displacements:
                results.displacements[isubcase] = {}
                if isubcase not in results.subcases:
                    results.subcases.append(isubcase)
                    results.stresses[isubcase] = {}

            # node_gridtype: (nnodes, 2)  col0=node_id
            # data:          (ntimes, nnodes, 6)  tx ty tz rx ry rz
            node_ids = res.node_gridtype[:, 0]
            # Use last time step (or only one for static)
            data = res.data[-1]  # (nnodes, 6)

            for i, nid in enumerate(node_ids):
                t1, t2, t3 = float(data[i,0]), float(data[i,1]), float(data[i,2])
                r1, r2, r3 = float(data[i,3]), float(data[i,4]), float(data[i,5])
                results.displacements[isubcase][int(nid)] = DisplacementResult(
                    subcase=isubcase, node_id=int(nid),
                    t1=t1, t2=t2, t3=t3,
                    r1=r1, r2=r2, r3=r3
                )

    def _read_plate_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            if isubcase not in results.stresses:
                results.stresses[isubcase] = {}
            # element_node: (nlayers, 2) col0=eid
            # data: (ntimes, nlayers, 8) fiber_dist oxx oyy txy angle omax omin ovm
            eids = res.element_node[:, 0]
            data = res.data[-1]  # last time step

            # Get von_mises — use built-in if available
            try:
                ovm = res.von_mises[-1]  # (nlayers,)
            except Exception:
                # compute manually: sqrt(sx^2 + sy^2 - sx*sy + 3*txy^2)
                sx  = data[:, 1]; sy  = data[:, 2]; txy = data[:, 3]
                ovm = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)

            # Average layers per element (centroid + corner nodes)
            seen = {}
            for i, eid in enumerate(eids):
                eid = int(eid)
                if eid not in seen:
                    seen[eid] = []
                seen[eid].append(float(ovm[i]))

            for eid, vals in seen.items():
                vm = float(np.mean(vals))
                results.stresses[isubcase][eid] = ElementStress(
                    subcase=isubcase, elem_id=eid, elem_type=etype,
                    values={'von_mises': vm,
                            'oxx': float(np.mean([data[i,1] for i,e in enumerate(eids) if int(e)==eid])),
                            'oyy': float(np.mean([data[i,2] for i,e in enumerate(eids) if int(e)==eid])),
                            'txy': float(np.mean([data[i,3] for i,e in enumerate(eids) if int(e)==eid]))},
                    von_mises=vm
                )

    def _read_1d_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            if isubcase not in results.stresses:
                results.stresses[isubcase] = {}
            # data layout varies by element type
            # For CBAR: data (ntimes, nelems, nresults)
            # element: (nelems,) element ids
            try:
                eids = res.element if hasattr(res, 'element') else res.element_node[:,0]
                data = res.data[-1]
                for i, eid in enumerate(eids):
                    eid = int(eid)
                    # Use max abs stress as "von mises equivalent"
                    row = data[i] if data.ndim == 2 else data[0, i]
                    vm = float(np.max(np.abs(row)))
                    results.stresses[isubcase][eid] = ElementStress(
                        subcase=isubcase, elem_id=eid, elem_type=etype,
                        values={'max_stress': vm}, von_mises=vm
                    )
            except Exception as e:
                print(f"  [OP2] skip {etype} stress: {e}")

    def _read_solid_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            if isubcase not in results.stresses:
                results.stresses[isubcase] = {}
            try:
                # element_node: (npts, 2) or element: (nelems,)
                if hasattr(res, 'element_node'):
                    eids = res.element_node[:, 0]
                else:
                    eids = res.element
                data = res.data[-1]

                # Von Mises from principal stresses or direct
                try:
                    ovm = res.von_mises[-1]
                except Exception:
                    # oxx oyy ozz txy tyz txz -> von mises
                    sx = data[:,0]; sy = data[:,1]; sz = data[:,2]
                    txy= data[:,3]; tyz= data[:,4]; txz= data[:,5]
                    ovm = np.sqrt(0.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2
                                       + 6*(txy**2+tyz**2+txz**2)))

                seen = {}
                for i, eid in enumerate(eids):
                    eid = int(eid)
                    if eid not in seen:
                        seen[eid] = []
                    seen[eid].append(float(ovm[i]))

                for eid, vals in seen.items():
                    vm = float(np.mean(vals))
                    results.stresses[isubcase][eid] = ElementStress(
                        subcase=isubcase, elem_id=eid, elem_type=etype,
                        values={'von_mises': vm}, von_mises=vm
                    )
            except Exception as e:
                print(f"  [OP2] skip {etype} stress: {e}")


# ---------------------------------------------------------------------------
# NEU (Femap Neutral File) Parser
# ---------------------------------------------------------------------------

class NEUParser:
    """
    Reads Femap .neu neutral file output results.
    Supports: displacement vectors, element centroid stresses.
    """
    def parse(self, filepath: str) -> F06Results:
        results = F06Results()
        results.subcases = [1]
        results.displacements[1] = {}
        results.stresses[1]      = {}

        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Output set header: "   OUTPUT SET   1  ..."
            if line.startswith('OUTPUT SET'):
                parts = line.split()
                try:
                    sc = int(parts[2])
                    if sc not in results.subcases:
                        results.subcases.append(sc)
                        results.displacements[sc] = {}
                        results.stresses[sc]      = {}
                except (IndexError, ValueError):
                    sc = 1
                current_sc = sc
                i += 1
                continue

            # Displacement block: starts with record type 1
            # Format: node_id  tx ty tz rx ry rz
            if line.startswith('1,') or (len(line) > 2 and line[0] == '1' and ',' in line[:5]):
                # Try parse displacement records
                try:
                    parts = line.split(',')
                    if len(parts) >= 7:
                        nid = int(parts[0].strip()) if parts[0].strip().isdigit() else -1
                        if nid > 0:
                            vals = [float(p.strip()) for p in parts[1:7]]
                            sc = getattr(self, '_current_sc', 1)
                            results.displacements.setdefault(sc, {})[nid] = \
                                DisplacementResult(subcase=sc, node_id=nid,
                                    t1=vals[0], t2=vals[1], t3=vals[2],
                                    r1=vals[3], r2=vals[4], r3=vals[5])
                except Exception:
                    pass
                i += 1
                continue

            i += 1

        if not results.subcases:
            results.subcases = [1]

        nd = len(results.displacements.get(1, {}))
        print(f"[NEU] {nd} displacements read")
        return results


# ---------------------------------------------------------------------------
# Auto-detect and load
# ---------------------------------------------------------------------------

def load_results(filepath: str) -> F06Results:
    """
    Auto-detect result file type and parse.
    Supports: .op2, .f06, .neu
    """
    ext = filepath.lower().split('.')[-1]

    if ext == 'op2':
        return OP2Parser().parse(filepath)
    elif ext in ('f06', 'pch'):
        from parser.f06_parser import load_f06
        return load_f06(filepath)
    elif ext == 'neu':
        try:
            from parser.neu_parser import load_neu
            return load_neu(filepath)
        except ImportError:
            return NEUParser().parse(filepath)   # fallback to simple parser
    else:
        # Try OP2 first, then F06
        try:
            return OP2Parser().parse(filepath)
        except Exception:
            from parser.f06_parser import load_f06
            return load_f06(filepath)
