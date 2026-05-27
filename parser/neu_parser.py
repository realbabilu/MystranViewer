import re
"""
Femap Neutral File (.neu) parser using femap-neutral-parser library.

Reads via femap_neutral_parser.parser.Parser:
  - Displacements (T1/T2/T3 translation, R1/R2/R3 rotation)
  - Plate stresses (von Mises, principal, membrane/bending)
  - Solid stresses (von Mises)
  - Bar/beam stresses

Falls back to raw title scanning if specific vectors not found.
"""

import numpy as np
from typing import Dict, List, Optional

from parser.f06_parser import F06Results, DisplacementResult, ElementStress


# ---------------------------------------------------------------------------
# Known Femap output vector title patterns for stress
# These are the actual strings Femap writes in .neu files
# ---------------------------------------------------------------------------

# Node-based displacement vector titles
# femap-neutral-parser uses 'category::field' format
_DISP_T1 = ['displacements::t1', 'T1 translation', 'T1 Translation']
_DISP_T2 = ['displacements::t2', 'T2 translation', 'T2 Translation']
_DISP_T3 = ['displacements::t3', 'T3 translation', 'T3 Translation']
_DISP_R1 = ['displacements::r1', 'R1 rotation',    'R1 Rotation']
_DISP_R2 = ['displacements::r2', 'R2 rotation',    'R2 Rotation']
_DISP_R3 = ['displacements::r3', 'R3 rotation',    'R3 Rotation']

# Element stress vector title patterns (partial match)
_STRESS_VONMISES_PATTERNS = [
    'von mises', 'Von Mises', 'VON MISES',
    'plate top von mises', 'plate bot von mises', 'solid von mises',
    'Plate Top VonMises', 'Plate Bot VonMises',
    'VonMises', 'vonmises',
]
_STRESS_NORMAL_PATTERNS = [
    'x normal', 'y normal', 'xy shear',
    'X Normal', 'Y Normal', 'XY Shear',
    'Plate Top X Normal', 'Plate Bot X Normal',
]
_STRESS_PLATE_TOP = ['Plate Top', 'plate top', 'PLATE TOP']
_STRESS_PLATE_BOT = ['Plate Bot', 'plate bot', 'PLATE BOT']


def _match_any(title: str, patterns: list) -> bool:
    tl = title.lower()
    return any(p.lower() in tl for p in patterns)


class NEUFemapParser:
    """Parse Femap .neu neutral file using femap-neutral-parser library."""

    def parse(self, filepath: str) -> F06Results:
        try:
            from femap_neutral_parser.parser import Parser
        except ImportError:
            raise ImportError(
                "femap-neutral-parser not installed.\n"
                "Run: pip install femap-neutral-parser")

        p = Parser(filepath)
        results = F06Results()

        # ── Subcases ─────────────────────────────────────────────────
        subcases = sorted(p.output_sets.keys())
        results.subcases = subcases
        for sc in subcases:
            results.displacements[sc] = {}
            results.stresses[sc]      = {}

        print(f"[NEU] {len(subcases)} subcase(s): {subcases}")
        # Print available vectors
        avail = list(p._output_vectors.keys())
        print(f"[NEU] {len(avail)} output vectors available")

        # ── Displacements ────────────────────────────────────────────
        self._read_displacements(p, subcases, results)

        # ── Stresses ─────────────────────────────────────────────────
        self._read_stresses(p, subcases, results)

        for sc in subcases:
            nd = len(results.displacements.get(sc, {}))
            ns = len(results.stresses.get(sc, {}))
            print(f"  SC {sc}: {nd} node disp, {ns} elem stress")

        return results

    # ----------------------------------------------------------------
    def _read_displacements(self, p, subcases, results):
        """Read T1/T2/T3 translations and R1/R2/R3 rotations."""
        avail = {k.lower(): k for k in p._output_vectors.keys()}

        def find_title(candidates):
            for c in candidates:
                if c.lower() in avail:
                    return avail[c.lower()]
            return None

        t1_key = find_title(_DISP_T1)
        t2_key = find_title(_DISP_T2)
        t3_key = find_title(_DISP_T3)
        r1_key = find_title(_DISP_R1)
        r2_key = find_title(_DISP_R2)
        r3_key = find_title(_DISP_R3)

        if not any([t1_key, t2_key, t3_key]):
            print("[NEU] No displacement vectors found")
            return

        print(f"[NEU] Displacement vectors: T1={t1_key} T2={t2_key} T3={t3_key}")

        for sc in subcases:
            # Build per-node dict: {nid: [t1,t2,t3,r1,r2,r3]}
            node_data: Dict[int, List[float]] = {}

            for idx, key in enumerate([t1_key, t2_key, t3_key,
                                        r1_key, r2_key, r3_key]):
                if key is None:
                    continue
                try:
                    ov = p._output_vectors[key]
                    if sc not in ov:
                        continue
                    rec = ov[sc]['record']
                    # femap-neutral-parser: fields are (NodeID, <vector_name>)
                    id_field = 'NodeID' if 'NodeID' in rec.dtype.names else 'entityID'
                    # value field = second field name (the vector key itself)
                    val_field = [f for f in rec.dtype.names if f != id_field][0]
                    for row in rec:
                        nid = int(row[id_field])
                        val = float(row[val_field])
                        if nid not in node_data:
                            node_data[nid] = [0.0] * 6
                        node_data[nid][idx] = val
                except Exception as e:
                    print(f"[NEU]   skip {key} SC{sc}: {e}")

            for nid, vals in node_data.items():
                results.displacements[sc][nid] = DisplacementResult(
                    subcase=sc, node_id=nid,
                    t1=vals[0], t2=vals[1], t3=vals[2],
                    r1=vals[3], r2=vals[4], r3=vals[5]
                )

    # ----------------------------------------------------------------
    def _read_stresses(self, p, subcases, results):
        """Read element stresses — find von Mises or principal stress vectors."""
        avail = list(p._output_vectors.keys())

        # Find von Mises vectors
        all_vm_keys = [k for k in avail if _match_any(k, _STRESS_VONMISES_PATTERNS)]

        # Priority: centroid stress > centroid strain > corner stress > corner strain
        def _vm_priority(k):
            kl = k.lower()
            if 'strain' in kl: return 10  # strain = lowest
            if re.search(r'solidc[0-9]', kl): return 5  # corner = lower
            if 'plate' in kl and ('top' in kl or 'bot' in kl): return 2
            return 1  # centroid stress = best

        # Sort and keep only the best group (lowest priority number)
        if all_vm_keys:
            all_vm_keys.sort(key=_vm_priority)
            best_prio = _vm_priority(all_vm_keys[0])
            vm_keys = [k for k in all_vm_keys if _vm_priority(k) == best_prio]
        else:
            vm_keys = []

        if not vm_keys:
            # Try normal stress as proxy for von mises
            normal_keys = [k for k in avail if _match_any(k, _STRESS_NORMAL_PATTERNS)]
            if normal_keys:
                print(f"[NEU] Computing VM proxy from: {normal_keys}")
                vm_keys = normal_keys  # will average these
            elif any('stress' in k.lower() or 'strain' in k.lower() for k in avail):
                vm_keys = [k for k in avail if 'stress' in k.lower() or 'strain' in k.lower()][:3]
                print(f"[NEU] Using stress vectors: {vm_keys}")
            else:
                print("[NEU] No stress vectors found")
                return

        print(f"[NEU] Von Mises vectors: {vm_keys}")

        for sc in subcases:
            # Collect per-element von Mises from all matching vectors
            elem_vm: Dict[int, List[float]] = {}

            for key in vm_keys:
                try:
                    ov = p._output_vectors[key]
                    if sc not in ov:
                        continue
                    rec = ov[sc]['record']
                    # Field names: (EntityID/ElementID, <vector_name>)
                    id_field  = ('ElementID' if 'ElementID' in rec.dtype.names
                                 else 'entityID' if 'entityID' in rec.dtype.names
                                 else 'NodeID' if 'NodeID' in rec.dtype.names
                                 else rec.dtype.names[0])
                    val_field = [f for f in rec.dtype.names if f != id_field][0]
                    for row in rec:
                        eid = int(row[id_field])
                        val = float(row[val_field])
                        if eid not in elem_vm:
                            elem_vm[eid] = []
                        elem_vm[eid].append(abs(val))
                except Exception as e:
                    print(f"[NEU]   skip {key} SC{sc}: {e}")

            for eid, vals in elem_vm.items():
                vm = float(np.max(np.abs(vals)))  # worst case across layers
                results.stresses[sc][eid] = ElementStress(
                    subcase=sc, elem_id=eid, elem_type='NEU',
                    values={'von_mises': vm},
                    von_mises=vm
                )

    # ----------------------------------------------------------------
    def info(self, filepath: str):
        """Print available vectors in a .neu file."""
        from femap_neutral_parser.parser import Parser
        p = Parser(filepath)
        p.info()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_neu(filepath: str) -> F06Results:
    return NEUFemapParser().parse(filepath)
