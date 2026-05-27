# MYSTRAN Viewer

Pre/post processor viewer untuk MYSTRAN FEA solver.
Terinspirasi dari MIDAS Civil versi lama — antarmuka engineering yang clean dan efisien.

## Fitur

### Element types yang didukung
| Dimensi | Tipe |
|---------|------|
| 1D | CBAR, CBEAM, CROD |
| 2D | CQUAD4, CTRIA3 |
| 3D | CHEXA (8-node brick), CPENTA (6-node wedge), CTETRA (4-node tet) |

### Display modes
- **Wireframe** — edge rendering, hijau terang (default)
- **Solid** — flat-shaded + Phong lighting, edge overlay opsional
- **Contour** — color mapping hasil analisis

### Results yang ditampilkan
- Displacement magnitude (dari .f06)
- Von Mises stress (dari .f06)
- Deformed shape dengan skala adjustable

### Navigasi 3D
| Input | Aksi |
|-------|------|
| Left drag | Rotate (arcball) |
| Right drag / Middle drag | Pan |
| Scroll | Zoom |
| `F` | Fit model ke viewport |
| `R` | Reset camera |
| `W` | Wireframe mode |
| `S` | Solid mode |
| `C` | Contour mode |
| `Esc` | Quit |

## Instalasi

```bash
pip install moderngl imgui-bundle glfw pyrr numpy
```

## Cara pakai

```bash
# Buka viewer kosong
python main.py

# Langsung load .dat
python main.py model.dat

# Load .dat + .f06
python main.py model.dat model.f06
```

Atau via GUI: **File → Open .dat** / **File → Open .f06**

## Struktur project

```
mystran_viewer/
├── main.py                  ← Entry point, GLFW window + render loop
├── requirements.txt
├── parser/
│   ├── dat_parser.py        ← Parse MYSTRAN .dat (semua element types)
│   └── f06_parser.py        ← Parse .f06 output (displacement, stress)
├── renderer/
│   ├── camera.py            ← Arcball camera (rotate/pan/zoom)
│   ├── contour.py           ← Colormap utilities (rainbow/jet/coolwarm)
│   └── mesh_renderer.py     ← OpenGL VAO/VBO, shader, draw
├── gui/
│   └── panels.py            ← Dear ImGui panels (left/right sidebar, legend)
└── samples/
    ├── cook_membrane.dat    ← Cook's membrane benchmark
    └── mixed_demo.dat       ← Mixed 1D/2D/3D element demo
```

## Roadmap

- [ ] Node picking (click → highlight + info tooltip)
- [ ] Element picking
- [ ] Node number labels (bitmap font)
- [ ] Force/moment arrow rendering
- [ ] .f06 stress contour per element centroid
- [ ] Export screenshot (PNG)
- [ ] Multiple load case comparison
- [ ] Section cut plane
- [ ] CTETRA10, CHEXA20 (quadratic elements)
- [ ] Animate deformation

## Platform notes

- **Windows**: fully tested target
- **Mac**: moderngl pakai OpenGL core profile 3.3, masih didukung macOS (deprecated tapi fungsional).
  Jika ingin future-proof di Mac, upgrade ke Metal via `wgpu-py` nanti.
- **Linux**: harusnya jalan langsung

## Catatan teknis

Parser menggunakan fixed-field 8-char parsing sesuai standar NASTRAN bulk data.
Supports small-field, large-field (*), dan free-field (,) format.
Continuation lines (+) di-handle otomatis.
