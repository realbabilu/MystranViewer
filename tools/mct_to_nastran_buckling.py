from __future__ import annotations

import argparse
from pathlib import Path


def _split_csv(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def _fmt8(value) -> str:
    if value is None or value == "":
        return " " * 8
    if isinstance(value, str):
        return value[:8].ljust(8)
    if isinstance(value, int):
        return f"{value:d}"[:8].rjust(8)
    text = f"{float(value):.7g}"
    if len(text) > 8:
        text = f"{float(value):.5E}".replace("E+0", "+").replace("E-0", "-")
    return text[:8].rjust(8)


def _card8(name: str, *fields) -> str:
    return name[:8].ljust(8) + "".join(_fmt8(field) for field in fields)


def _iter_section_lines(lines: list[str], name: str) -> list[str]:
    out: list[str] = []
    active = False
    tag = f"*{name}"
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("*"):
            active = line.upper().startswith(tag.upper())
            continue
        if active:
            out.append(line)
    return out


def parse_mct(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    nodes: list[tuple[int, float, float, float]] = []
    for line in _iter_section_lines(lines, "NODE"):
        nid, x, y, z = _split_csv(line)[:4]
        nodes.append((int(nid), float(x), float(y), float(z)))

    elements: list[tuple[int, int, int, int, int, int]] = []
    for line in _iter_section_lines(lines, "ELEMENT"):
        parts = _split_csv(line)
        if len(parts) < 8 or parts[1].upper() != "PLATE":
            continue
        eid = int(parts[0])
        pid = int(parts[3])
        n1, n2, n3, n4 = map(int, parts[4:8])
        elements.append((eid, pid, n1, n2, n3, n4))

    mat = None
    for line in _iter_section_lines(lines, "MATERIAL"):
        parts = _split_csv(line)
        if len(parts) < 11:
            continue
        mid = int(parts[0])
        if parts[1].upper() != "USER":
            continue
        mode = parts[9]
        if mode != "2":
            continue
        e = float(parts[10])
        nu = float(parts[11])
        mat = (mid, e, nu)
        break
    if mat is None:
        raise ValueError("No USER material with elastic data found in MCT")

    thickness = None
    for line in _iter_section_lines(lines, "THICKNESS"):
        parts = _split_csv(line)
        if len(parts) < 5:
            continue
        thickness = (int(parts[0]), float(parts[4]))
        break
    if thickness is None:
        raise ValueError("No thickness definition found in MCT")

    constraints: list[tuple[int, str]] = []
    for line in _iter_section_lines(lines, "CONSTRAINT"):
        parts = _split_csv(line)
        if len(parts) < 2:
            continue
        nid = int(parts[0])
        dof_mask = parts[1]
        dofs = "".join(str(i + 1) for i, ch in enumerate(dof_mask[:6]) if ch == "1")
        if dofs:
            constraints.append((nid, dofs))

    loads: list[tuple[int, float, float, float, float, float, float]] = []
    for line in _iter_section_lines(lines, "CONLOAD"):
        parts = _split_csv(line)
        if len(parts) < 7:
            continue
        nid = int(parts[0])
        vals = tuple(float(v) for v in parts[1:7])
        loads.append((nid, *vals))

    buck = None
    buck_lines = _iter_section_lines(lines, "BUCK-CTRL")
    if buck_lines:
        parts = _split_csv(buck_lines[0])
        nmode = int(parts[0])
        positive = parts[1].upper() == "YES"
        buck = (nmode, positive)

    return {
        "nodes": nodes,
        "elements": elements,
        "material": mat,
        "thickness": thickness,
        "constraints": constraints,
        "loads": loads,
        "buckling": buck,
    }


def build_bdf(data: dict, title: str) -> str:
    mid, e, nu = data["material"]
    pid, t = data["thickness"]
    nmode = data["buckling"][0] if data["buckling"] else 10

    out: list[str] = [
        f"ID {title}",
        "SOL 105",
        "CEND",
        f"TITLE = {title}",
        "ECHO = NONE",
        "DISP(PRINT,POST) = ALL",
        "STRES(CENTER,PRINT,POST) = ALL",
        "FORCES(PRINT,POST) = ALL",
        "SPCFORCES(PRINT,POST) = ALL",
        "SUBCASE 1",
        "  SUBTITLE = PREBUCKLING",
        "  SPC = 101",
        "  LOAD = 201",
        "SUBCASE 2",
        "  SUBTITLE = BUCKLING",
        "  SPC = 101",
        "  METHOD = 20",
        "  STATSUB(PRELOAD) = 1",
        "  DISP(PRINT,POST) = ALL",
        "BEGIN BULK",
        "PARAM,POST,-1",
        "PARAM,K6ROT,1.0",
        "PARAM,AUTOSPC,NO",
        _card8("EIGRL", 20, None, None, nmode),
    ]

    for nid, x, y, z in data["nodes"]:
        out.append(_card8("GRID", nid, None, x, y, z))

    g = e / (2.0 * (1.0 + nu))
    out.append(_card8("MAT1", mid, e, g, nu))
    out.append(_card8("PSHELL", pid, mid, t))

    for eid, epid, n1, n2, n3, n4 in data["elements"]:
        out.append(_card8("CQUAD4", eid, epid, n1, n2, n3, n4))

    for i, (nid, dofs) in enumerate(data["constraints"], start=1):
        out.append(_card8("SPC1", 101, dofs, nid))

    for nid, fx, fy, fz, mx, my, mz in data["loads"]:
        if abs(fx) > 0.0:
            out.append(_card8("FORCE", 201, nid, 0, fx, 1.0, 0.0, 0.0))
        if abs(fy) > 0.0:
            out.append(_card8("FORCE", 201, nid, 0, fy, 0.0, 1.0, 0.0))
        if abs(fz) > 0.0:
            out.append(_card8("FORCE", 201, nid, 0, fz, 0.0, 0.0, 1.0))
        if abs(mx) > 0.0:
            out.append(_card8("MOMENT", 201, nid, 0, mx, 1.0, 0.0, 0.0))
        if abs(my) > 0.0:
            out.append(_card8("MOMENT", 201, nid, 0, my, 0.0, 1.0, 0.0))
        if abs(mz) > 0.0:
            out.append(_card8("MOMENT", 201, nid, 0, mz, 0.0, 0.0, 1.0))

    out.append("ENDDATA")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MIDAS MCT plate buckling model to NX Nastran BDF")
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ns = ap.parse_args()

    data = parse_mct(ns.input)
    title = ns.input.stem.replace("_", " ").upper()
    text = build_bdf(data, title)
    out = ns.output or ns.input.with_name(ns.input.stem + "_nxnastran.bdf")
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Nodes: {len(data['nodes'])}")
    print(f"Elements: {len(data['elements'])}")
    print(f"Constraints: {len(data['constraints'])}")
    print(f"Loads: {len(data['loads'])}")


if __name__ == "__main__":
    main()
