"""OGS1 direct volume stresses, kept separate by volume and grid ID."""
import numpy as np
from parser.f06_parser import ElementGPStress

GP_COMPONENTS = {'gp_vm': 'ovm', 'gp_sxx': 'sxx', 'gp_syy': 'syy',
                 'gp_szz': 'szz', 'gp_txy': 'txy', 'gp_tyz': 'tyz',
                 'gp_txz': 'txz', 'gp_s1': 's1', 'gp_s2': 's2',
                 'gp_s3': 's3', 'gp_pressure': 'pressure'}
GP_LABELS = {'gp_vm': 'Von Mises', 'gp_sxx': 'Sxx', 'gp_syy': 'Syy',
             'gp_szz': 'Szz', 'gp_txy': 'Txy', 'gp_tyz': 'Tyz',
             'gp_txz': 'Txz', 'gp_s1': 'S1', 'gp_s2': 'S2',
             'gp_s3': 'S3', 'gp_pressure': 'Pressure'}


def gp_location(result_type):
    return result_type.rsplit('_', 1)[-1].upper()


def read_gp_volume_stress(op2, results):
    count = 0
    for key, res in getattr(op2, 'grid_point_stresses_volume_direct', {}).items():
        sc = int(key[0] if isinstance(key, tuple) else key)
        volume = int(res.ogs_id)
        location = f'V{volume}'
        cid = int(getattr(res, 'refid', res.data_code.get('refid', 0)))
        store = results.gp_stresses.setdefault(sc, {})
        results.gp_volume_cids.setdefault(sc, {})[volume] = cid
        if sc not in results.subcases:
            results.subcases.append(sc)
        # Decode by headers, not undocumented positional principal-table slots.
        columns = {name: i for i, name in enumerate(res.get_headers())}
        for nid, row in zip(res.node, res.data[-1]):
            sx, sy, sz = (float(row[columns[c]]) for c in ('ox', 'oy', 'oz'))
            xy, yz, xz = (float(row[columns[c]]) for c in ('txy', 'tyz', 'txz'))
            principals = np.linalg.eigvalsh([[sx, xy, xz], [xy, sy, yz], [xz, yz, sz]])
            nid = int(nid)
            store[(nid, location)] = ElementGPStress(
                subcase=sc, node_id=nid, sxx=sx, syy=sy, szz=sz,
                txy=xy, tyz=yz, txz=xz, s1=float(principals[2]),
                s2=float(principals[1]), s3=float(principals[0]),
                pressure=float(row[columns['pressure']]), ovm=float(row[columns['ovm']]))
            count += 1
    return count


def gp_location_label(results, subcase, location):
    if location.startswith('V') and location[1:].isdigit():
        volume = int(location[1:])
        cid = results.gp_volume_cids.get(subcase, {}).get(volume, 0)
        return f"Volume {volume} (CID {cid})"
    return {'Z1': 'Top', 'Z2': 'Bot', 'MID': 'Mid'}.get(location, location)


def volume_node_groups(model, results=None, subcase=1):
    """Volume notation from DAT element sets, with OP2 grid IDs as a fallback."""
    groups = {}
    if model is not None:
        for vid, (eids, _) in model.volumes.items():
            groups[vid] = {nid for eid in eids if eid in model.elements
                           for nid in model.elements[eid].nodes}
    if results is not None:
        for (nid, location) in results.gp_stresses.get(subcase, {}):
            if location.startswith('V') and location[1:].isdigit():
                groups.setdefault(int(location[1:]), set()).add(nid)
    return groups
