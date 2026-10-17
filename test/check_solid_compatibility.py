"""Read-only DAT/OP2 compatibility probe. Run with --numpy-compat for old pyNastran + NumPy 2."""
import sys, pathlib, contextlib, io, json, ast, types
sys.path.insert(0,str(pathlib.Path.cwd()))
import numpy as np
if '--numpy-compat' in sys.argv:
 from numpy._core.defchararray import chararray
 np.chararray=chararray
 np.in1d=lambda a,b,**kw:np.isin(a,b,**kw)
from parser.dat_parser import load_dat
from parser.f06_parser import F06Results
from pyNastran.op2.op2 import OP2
import pyNastran
source=ast.parse(pathlib.Path('main.py').read_text(encoding='utf-8'))
method=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='_read_solid_stress')
ns={}; exec(compile(ast.Module(body=[method],type_ignores=[]),'main.py','exec'),ns)
viewer=types.SimpleNamespace(_log=lambda *a:None)
root=pathlib.Path('H:/Projects/MYSTRAN')
base=root/'codex_mod/2026-10-17/solid_sap_comparison/all_solids/runs/5-004/24x4x2'
paths=[f/'mystran_NEWSOLID/model.dat' for f in sorted(base.iterdir())]
paths+=list((root/'nastran_test/mystran/quadratic_patch_audit').glob('*PATCH.dat'))
paths+=[root/'CTETRA4_NEWSOLID_CODEX_HANDOFF_v3_20261017/CASE_5011/5011_NEWSOLID.dat']
rows=[]
for p in paths:
 if not p.exists(): continue
 row={'type':p.parent.parent.name if p.name=='model.dat' else p.stem,'dat':str(p)}
 try:
  with contextlib.redirect_stdout(io.StringIO()): model=load_dat(str(p))
  row.update(nodes=len(model.nodes),elements=len(model.elements),connectivity=sorted(set(len(e.nodes) for e in model.elements.values())))
  op=OP2(debug=False)
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): op.read_op2(str(p.with_suffix('.OP2')),combine=True)
  result=F06Results(); tables={}
  stress=getattr(getattr(op,'op2_results',None),'stress',op)
  for attr in ('chexa_stress','cpenta_stress','ctetra_stress','cpyram_stress'):
   table=getattr(stress,attr,{})
   if table:
    tables[attr]=[{'subcase':str(k),'shape':list(v.data.shape),'centroids':int((v.element_node[:,1]==0).sum())} for k,v in table.items()]
    ns['_read_solid_stress'](viewer,table,attr.split('_')[0].upper(),result)
  row.update(displacements={str(k):len(v.node_gridtype) for k,v in op.displacements.items()},tables=tables,viewer_stress={str(k):sum(isinstance(e,int) for e in v) for k,v in result.stresses.items()})
 except Exception as e: row['error']=type(e).__name__+': '+str(e)
 rows.append(row)
report={'numpy_compat_shim':'--numpy-compat' in sys.argv,'numpy':np.__version__,'pyNastran':pyNastran.__version__,'cases':rows}
pathlib.Path('test/solid_compatibility_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for r in rows: print(r['type'], {k:v for k,v in r.items() if k not in ('type','dat')})
