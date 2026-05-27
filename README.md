# MystranViewer
A simplified python Nastran-compatible Viewer

# MYSTRAN Viewer — FEM Post Processor (for now, later maybe can be as Pre too)

Python OpenGL FEM viewer for NASTRAN/MYSTRAN models.
<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/9fdbf94a-a4af-4fdc-aa4b-c3a2530be11c" />
<img width="1917" height="1031" alt="image" src="https://github.com/user-attachments/assets/8bc8cf06-6157-49a9-9517-cc812625ed55" />
<img width="1919" height="1032" alt="image" src="https://github.com/user-attachments/assets/6c6949e7-9bb0-4966-95eb-7514f1590047" />
<img width="1919" height="1030" alt="image" src="https://github.com/user-attachments/assets/224dd7ef-d144-45dc-8552-5eb09f2712d9" />
<img width="1919" height="1029" alt="image" src="https://github.com/user-attachments/assets/d88bcc5e-81b5-42bf-b082-f0bb6a6d9b36" />

## Requirements
```
Use python 312 since it need pyNastran
pip install moderngl glfw imgui-bundle pyrr numpy pyNastran imageio[ffmpeg] femap-neutral-parser

```

## Usage
```
python main.py model.bdf                          # geometry only
python main.py model.bdf model.f06                # + results (F06)
python main.py model.bdf model.op2                # + results (OP2)
python main.py model.bdf model.neu                # + results (NEU/Femap)
```

## Features
- Display modes: Wireframe / Hidden Line / Contour (Post)
- Results: Displacement (T1/T2/T3/Total), Stress (Vm/Sxx/Syy/Sxy/S1/S3), Forces (Fx/Fy/Mx/My/Mxy/Qx/Qy)
- Element types: CQUAD4, CQUADR, CTRIA3, CTRIAR, CHEXA, CPENTA, CTETRA, CPYRAM, CBAR, CBEAM, CROD
- Modal/Eigen: automatic deform scale, mode Hz in title
- Solid stress: centroid (element) + corner nodes (nodal average)
- Beam diagrams: exact Hermitian shape functions, exact V/M with load jumps
- Export: MP4 animation (requires ffmpeg)
- Notation: node numbers, element numbers, result value labels, SPC constraints
- Notation clipped to safe zone (hidden behind panels)
- View presets: X-Z / Y-Z / X-Y / ISO

## Key Files
- main.py                    — entry point
- gui/panels.py              — ImGui UI panels
- parser/dat_parser.py       — BDF/DAT geometry reader
- parser/f06_parser.py       — F06 results (displacement, stress, force, solid)
- parser/neu_parser.py       — Femap neutral file reader
- renderer/mesh_renderer.py  — OpenGL VAO/VBO rendering
- renderer/beam_diagram.py   — Beam force/moment diagrams (OP2)
- renderer/exact_beam.py     — Exact Hermitian beam computations
- renderer/camera.py         — Arcball camera
- renderer/contour.py        — Colormaps

