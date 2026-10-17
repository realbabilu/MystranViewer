"""Report absent shell/solid element stress without substituting GPSTRESS."""
from collections import defaultdict

SOLID_TYPES = {'CHEXA', 'CPENTA', 'CTETRA', 'CPYRAM'}
SHELL_TYPES = {'CQUAD4', 'CQUAD8', 'CTRIA3', 'CTRIA6', 'CQUADR', 'CTRIAR'}


def missing_element_stresses(model, results, subcase):
    stresses = results.stresses.get(subcase, {})
    if not any(isinstance(eid, int) and hasattr(value, 'von_mises')
               for eid, value in stresses.items()):
        return {}
    missing = defaultdict(list)
    for eid, element in model.elements.items():
        if element.type not in SOLID_TYPES | SHELL_TYPES:
            continue
        if eid not in stresses:
            family = (f'{element.type}{len(element.nodes)}'
                      if element.type in SOLID_TYPES else element.type)
            missing[family].append(eid)
    return {family: sorted(ids) for family, ids in sorted(missing.items())}


def element_id_ranges(ids, max_ranges=4):
    """Compact IDs for logs and the results summary."""
    ranges = []
    for eid in sorted(set(ids)):
        if ranges and eid == ranges[-1][1] + 1:
            ranges[-1][1] = eid
        else:
            ranges.append([eid, eid])
    text = ', '.join(str(a) if a == b else f'{a}-{b}' for a, b in ranges[:max_ranges])
    return text + (', ...' if len(ranges) > max_ranges else '')
