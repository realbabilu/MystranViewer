# MystranViewer
A simplified python Nastran-compatible Viewer

# MYSTRAN Viewer — FEM Post Processor 

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
- Results: Displacement (T1/T2/T3/Total), Stress (Vm/Sxx/Syy/Sxy/S1/S3), Forces (Fx/Fy/Fxy/Mx/My/Mxy/Qx/Qy)
- Element types: CQUAD4, CQUAD8, CQUADR, CTRIA3, CTRIA6, CTRIAR, CHEXA, CPENTA, CTETRA, CPYRAM, CBAR, CBEAM, CROD
- Modal/Eigen: automatic deform scale, mode Hz in title
- Solid stress: centroid (element) + corner nodes (nodal average)
- Beam diagrams: exact Hermitian shape functions, exact V/M with load jumps
- Export: MP4 animation (requires ffmpeg)
- Notation: node numbers, element numbers, result value labels, SPC constraints
- Notation clipped to safe zone (hidden behind panels)
- View presets: X-Z / Y-Z / X-Y / ISO
- Quadratic shells: CQUAD8 and CTRIA6 geometry, midside-node rendering, and interpolated GP stress contour colors
- F06 readers: bundled MYSTRAN and NX/Simcenter Nastran readers for shell center/corner results and grid-point stresses
- Shell stress fibers: Top / Bottom / Mid controls for element and averaged nodal stresses, with legends and result labels
- Nodal contours: solver and derived stress averages, including mixed shell models and centroid-only CTRIA3 results
- Grid-point surface stresses (GPSTRESS): VM, Sxx, Syy, Txy, S1, and S2 with Z1 / Z2 / MID fiber selection when present in the results
- MYSTRAN F06 grid-point forces: Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qx, and Qy contours
- Surfaces: case-control SET / SURFACE definitions, model-browser selection, magenta highlighting, surface IDs, and surface-axis arrows
- Beam sections: circular CROD/PROD profiles and area-based PBAR/PBEAM visualization
- Diagnostics: on-screen log window and OP2 summary of available result tables
- OP2 compatibility: integer/tuple subcase keys and newer pyNastran stress-table organization; GPSTRESS availability depends on pyNastran and the OP2 format
- Examples: bundled `examples/duel3a.dat` / `examples/duel3a.f06`, plus CQUAD8 and CTRIA6 F06/OP2 fixtures

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


## Examples

Run the included model and F06 results:

```
python main.py examples/duel3a.dat examples/duel3a.f06
```

The `test/duel3a_macq8d.*` and `test/duel3a_mht6.*` fixtures contain CQUAD8 and CTRIA6 geometry with F06 and OP2 results.
