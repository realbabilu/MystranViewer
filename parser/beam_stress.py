"""
Beam stress recovery helpers for PBEAM/PBEAML-style section points.
"""


def beam_stress_at_station(bm1, bm2, af, A, I1, I2, I12,
                           C1, C2, D1, D2, E1, E2, F1, F2):
    """
    Exact beam normal stress recovery with I12 coupling.

    sigma = AF/A + beta*C1 + gamma*C2
    beta  = -(bm1*I2 - bm2*I12) / (I1*I2 - I12^2)
    gamma = -(bm2*I1 - bm1*I12) / (I1*I2 - I12^2)
    """
    denom = float(I1) * float(I2) - float(I12) ** 2
    if abs(float(A)) < 1e-30 or abs(denom) < 1e-30:
        return {'C': 0.0, 'D': 0.0, 'E': 0.0, 'F': 0.0}
    beta = -(float(bm1) * float(I2) - float(bm2) * float(I12)) / denom
    gamma = -(float(bm2) * float(I1) - float(bm1) * float(I12)) / denom
    base = float(af) / float(A)
    return {
        'C': base + beta * float(C1) + gamma * float(C2),
        'D': base + beta * float(D1) + gamma * float(D2),
        'E': base + beta * float(E1) + gamma * float(E2),
        'F': base + beta * float(F1) + gamma * float(F2),
    }


def pbeam_cdef_points(params: dict):
    """Return C/D/E/F section points from PBEAM-style raw params when present."""
    def g(key, default=0.0):
        v = params.get(key, '')
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).strip()) if str(v).strip() else default
        except ValueError:
            return default

    return {
        'C': (g('f9'),  g('f10')),
        'D': (g('f11'), g('f12')),
        'E': (g('f13'), g('f14')),
        'F': (g('f15'), g('f16')),
    }
