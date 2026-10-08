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
from parser.f06_parser import F06Results, DisplacementResult, ElementStress, ElementGPStress


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
        # Normalize all subcase keys to int
        subcases_int = sorted({int(sc[0] if isinstance(sc, tuple) else sc) for sc in subcases}) or [1]
        results.subcases = subcases_int
        for sc in results.subcases:
            results.displacements[sc] = {}
            results.stresses[sc]      = {}

        # ── Displacements ────────────────────────────────────────────
        self._read_displacements(op2.displacements, results)

        # Eigenvectors (modal) — treat as displacement subcases
        self._read_displacements(op2.eigenvectors, results)

        # ── Element stresses ─────────────────────────────────────────
        # pyNastran 1.4+ uses op2_results.stress / .force
        stress = getattr(getattr(op2, 'op2_results', None), 'stress', None) or op2
        force = getattr(getattr(op2, 'op2_results', None), 'force', None) or op2

        # 2D shells
        for etype, attr in [('CQUAD4','cquad4_stress'),('CQUAD8','cquad8_stress'),
                              ('CTRIA3','ctria3_stress'),('CTRIA6','ctria6_stress'),
                              ('CQUADR','cquadr_stress')]:
            table = getattr(stress, attr, None)
            if table:
                self._read_plate_stress(table, results, etype)

        # 1D bars/beams/rods
        for etype, attr in [('CROD','crod_stress'),('CONROD','conrod_stress')]:
            table = getattr(stress, attr, None)
            if table:
                self._read_1d_stress(table, results, etype)

        # 3D solids
        for etype, attr in [('CHEXA','chexa_stress'),('CPENTA','cpenta_stress'),
                              ('CTETRA','ctetra_stress')]:
            table = getattr(stress, attr, None)
            if table:
                self._read_solid_stress(table, results, etype)

        # ── Grid-point surface stresses (GPSTRESS) ───────────────────────
        self._read_gp_surface_stress(op2, results)

        print(f"[OP2] subcases={results.subcases}")
        for sc in results.subcases:
            nd = len(results.displacements.get(sc, {}))
            ns = len(results.stresses.get(sc, {}))
            ngp = len(results.gp_stresses.get(sc, {}))
            print(f"  SC {sc}: {nd} node displacements, {ns} element stresses, {ngp} GP stress entries")

        return results

    # ----------------------------------------------------------------
    def _read_displacements(self, table_dict, results: F06Results):
        if not table_dict:
            return
        for isubcase, res in table_dict.items():
            # Normalize to integer subcase key
            sc = int(isubcase[0]) if isinstance(isubcase, tuple) else int(isubcase)
            if sc not in results.displacements:
                results.displacements[sc] = {}
                if sc not in results.subcases:
                    results.subcases.append(sc)
                    results.stresses[sc] = {}

            # node_gridtype: (nnodes, 2)  col0=node_id
            # data:          (ntimes, nnodes, 6)  tx ty tz rx ry rz
            node_ids = res.node_gridtype[:, 0]
            data = res.data[-1]  # (nnodes, 6)

            for i, nid in enumerate(node_ids):
                t1, t2, t3 = float(data[i,0]), float(data[i,1]), float(data[i,2])
                r1, r2, r3 = float(data[i,3]), float(data[i,4]), float(data[i,5])
                results.displacements[sc][int(nid)] = DisplacementResult(
                    subcase=sc, node_id=int(nid),
                    t1=t1, t2=t2, t3=t3,
                    r1=r1, r2=r2, r3=r3
                )

    def _read_plate_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            sc = int(isubcase[0]) if isinstance(isubcase, tuple) else int(isubcase)
            if sc not in results.stresses:
                results.stresses[sc] = {}
            # element_node: (nlayers, 2) col0=eid
            # data: (ntimes, nlayers, 8) fiber_dist oxx oyy txy angle omax omin ovm
            eids = res.element_node[:, 0]
            data = res.data[-1]  # last time step

            # Get von_mises — use built-in if available
            try:
                ovm = res.von_mises[-1]  # (nlayers,)
            except Exception:
                sx  = data[:, 1]; sy  = data[:, 2]; txy = data[:, 3]
                ovm = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)

            # Separate Top (fiber>0) and Bottom (fiber<0) layers per element
            # Columns: 0=fiber_dist, 1=oxx, 2=oyy, 3=txy, 4=angle, 5=omax, 6=omin, 7=ovm
            seen_top = {}
            seen_bot = {}
            for i, eid in enumerate(eids):
                eid = int(eid)
                fiber = float(data[i, 0])
                if fiber >= 0:
                    if eid not in seen_top: seen_top[eid] = []
                    seen_top[eid].append({
                        'oxx': float(data[i, 1]), 'oyy': float(data[i, 2]),
                        'txy': float(data[i, 3]), 'omax': float(data[i, 5]),
                        'omin': float(data[i, 6]), 'ovm': float(ovm[i]),
                    })
                else:
                    if eid not in seen_bot: seen_bot[eid] = []
                    seen_bot[eid].append({
                        'oxx': float(data[i, 1]), 'oyy': float(data[i, 2]),
                        'txy': float(data[i, 3]), 'omax': float(data[i, 5]),
                        'omin': float(data[i, 6]), 'ovm': float(ovm[i]),
                    })

            def _avg(layers):
                if not layers:
                    return {'oxx': 0.0, 'oyy': 0.0, 'txy': 0.0,
                            'omax': 0.0, 'omin': 0.0, 'ovm': 0.0}
                n = len(layers)
                return {k: sum(l[k] for l in layers) / n for k in layers[0]}

            all_eids = set(list(seen_top.keys()) + list(seen_bot.keys()))
            for eid in all_eids:
                top = _avg(seen_top.get(eid, []))
                bot = _avg(seen_bot.get(eid, []))
                # Mid = average of top and bottom
                mid = {k: 0.5 * (top[k] + bot[k]) for k in top}
                # Default (centroid) = mid values
                vm_mid = mid['ovm']
                results.stresses[sc][eid] = ElementStress(
                    subcase=sc, elem_id=eid, elem_type=etype,
                    values={'von_mises': vm_mid,
                            'oxx': mid['oxx'], 'oyy': mid['oyy'], 'txy': mid['txy'],
                            'omax': mid['omax'], 'omin': mid['omin'],
                            'von_mises_top': top['ovm'], 'von_mises_bottom': bot['ovm'],
                            'oxx_top': top['oxx'], 'oxx_bottom': bot['oxx'],
                            'oyy_top': top['oyy'], 'oyy_bottom': bot['oyy'],
                            'txy_top': top['txy'], 'txy_bottom': bot['txy'],
                            'omax_top': top['omax'], 'omax_bottom': bot['omax'],
                            'omin_top': top['omin'], 'omin_bottom': bot['omin'],
                            },
                    von_mises=vm_mid
                )

    def _read_1d_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            sc = int(isubcase[0]) if isinstance(isubcase, tuple) else int(isubcase)
            if sc not in results.stresses:
                results.stresses[sc] = {}
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
                    results.stresses[sc][eid] = ElementStress(
                        subcase=sc, elem_id=eid, elem_type=etype,
                        values={'max_stress': vm}, von_mises=vm
                    )
            except Exception as e:
                print(f"  [OP2] skip {etype} stress: {e}")

    def _read_solid_stress(self, table_dict, results, etype):
        for isubcase, res in table_dict.items():
            sc = int(isubcase[0]) if isinstance(isubcase, tuple) else int(isubcase)
            if sc not in results.stresses:
                results.stresses[sc] = {}
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
                    results.stresses[sc][eid] = ElementStress(
                        subcase=sc, elem_id=eid, elem_type=etype,
                        values={'von_mises': vm}, von_mises=vm
                    )
            except Exception as e:
                print(f"  [OP2] skip {etype} stress: {e}")

    # ----------------------------------------------------------------
    def _read_gp_surface_stress(self, op2, results: F06Results):
        """Read OP2 grid_point_surface_stresses into results.gp_stresses.

        Data layout per entry:
            node_element: (nrows, 2)  col0=node_id, col1=zero
            location:     (nrows,)    'Z1  ' or 'Z2  ' (fiber location)
            data:        (ntimes, nrows, 8)
                         [sxx, syy, txy, angle, s1, s2, s12, ovm]

        Each grid node appears twice (Z1=bot, Z2=top fiber).
        We store both as separate GP entries keyed by (node_id, location).
        """
        gpss = getattr(op2, 'grid_point_surface_stresses', None)
        if not gpss:
            return

        for isubcase, res in gpss.items():
            # Map subcase key to user-facing subcase
            # isubcase is (approach, analysis, ...); use first int as subcase id
            sc = isubcase[0] if isinstance(isubcase, tuple) else isubcase
            if sc not in results.gp_stresses:
                results.gp_stresses[sc] = {}

            # Ensure subcase is tracked
            if sc not in results.subcases:
                results.subcases.append(sc)

            try:
                ne    = res.node_element   # (nrows, 2)
                locs  = res.location       # (nrows,)  'Z1  ' or 'Z2  '
                data  = res.data[-1]       # (nrows, 8) last time step

                for i in range(data.shape[0]):
                    nid = int(ne[i, 0])
                    loc = locs[i].strip()  # 'Z1' or 'Z2'
                    row = data[i]
                    gp = ElementGPStress(
                        subcase=sc,
                        node_id=nid,
                        sxx=float(row[0]),
                        syy=float(row[1]),
                        txy=float(row[2]),
                        angle=float(row[3]),
                        s1=float(row[4]),
                        s2=float(row[5]),
                        s12=float(row[6]),
                        ovm=float(row[7]),
                    )
                    # Key: (node_id, location) so Z1 and Z2 are distinct
                    results.gp_stresses[sc][(nid, loc)] = gp

            except Exception as e:
                print(f"  [OP2] skip gp_surface_stress subcase {isubcase}: {e}")




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
