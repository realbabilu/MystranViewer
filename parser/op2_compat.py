"""Small compatibility fixes for pyNastran's existing OP2 decoders."""
from pyNastran.op2.op2 import OP2
from pyNastran.op2.op2_interface import op2_scalar


class ViewerOP2(OP2):
    def set_table_type(self):
        super().set_table_type()
        # The solid OES decoder already handles 255, but 1.4.1's name map omits it.
        self.element_mapper = dict(self.element_mapper)
        self.element_mapper.setdefault(255, 'CPYRAM')


def read_op2(filepath):
    op2_scalar.STR_PARAMS_1 |= {b'SOLIDTYP', b'PENTATYP', b'QUAD4TYP',
                              b'QUAD8TYP', b'CTRIA6TYP', b'T3D', b'CQUAD4TYP'}
    model = ViewerOP2(debug=False)
    model.read_op2(str(filepath), combine=True)
    # OGS volume IDs force pyNastran to retain tuple keys in other result tables.
    # Keep volume keys intact, but use integer subcases for the viewer's usual data.
    for name in model.get_table_types():
        if 'volume' in name:
            continue
        table = model.get_result(name)
        if not isinstance(table, dict) or not table:
            continue
        keys = [key[0] if isinstance(key, tuple) else key for key in table]
        if len(set(keys)) != len(keys):
            continue  # Do not collapse distinct analysis results.
        values = list(table.values())
        table.clear()
        table.update(zip(keys, values))
    return model
