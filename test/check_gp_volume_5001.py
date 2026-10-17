"""Local integration probe for the audited MYSTRAN 5-001 files (read-only inputs)."""
import sys,contextlib,io,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from parser.dat_parser import load_dat
from parser.op2_compat import read_op2
from parser.gp_volume import GP_COMPONENTS
from gui.panels import ViewerState,draw_contour_toolbar,draw_legend
from main import MystranViewerApp
from imgui_bundle import imgui
root=Path('H:/Projects/MYSTRAN')
dat=root/'source/update/solid_benchmark/5-001_8_SOLIDS_GPSTRESS_PENTA_REMESH_FIXED_QUADRATIC_MYSTRAN_EAS9.dat'
actual=root/'nastran_test/mystran/quadratic_patch_audit/5-001_PENTATYP_EAS9.OP2'
reference=root/'nastran_test/5-001_GPSTRESS_BASIC/nastran/5-001_8_solids_gpstress_penta_remesh.op2'
app=MystranViewerApp.__new__(MystranViewerApp);app.state=ViewerState();app.state.model=load_dat(str(dat))
logs=[];app._log=lambda level,msg: logs.append((level,msg));app._rebuild=lambda:None;app._load_beam_diagrams=lambda *args:None
with contextlib.redirect_stdout(io.StringIO()):app._load_op2(str(actual));ref=read_op2(reference)
assert not [x for x in logs if x[0]=='err']
state=app.state;result=state.results;assert len(result.gp_stresses[1])==204
missing=result.missing_element_stresses[1]
assert missing == {'CHEXA8':list(range(500001,500008)), 'CPYRAM13':list(range(800001,800043))}
assert not state.has_shell_stress_fibers()
# Independent Fortran-record audit: make sure the reader did not discard the blank groups.
import struct,re,collections
binary=actual.read_bytes();position=0;table_name=b'';header=None;raw_counts=collections.Counter()
while position<len(binary):
 length=struct.unpack_from('<i',binary,position)[0]
 assert length>=0 and struct.unpack_from('<i',binary,position+length+4)[0]==length
 record=binary[position+4:position+length+4];position+=length+8
 if len(record)==8 and re.fullmatch(b'[A-Z0-9 ]{8}',record):
  table_name=record;header=None
 if table_name==b'OES1X1  ':
  if len(record)==584:
   header=struct.unpack('<50i',record[:200])
  elif header and len(record)>4:
   width=header[9]*4;assert len(record)%width==0
   for offset in range(0,len(record),width):
    eid=struct.unpack_from('<i',record,offset)[0]//10
    raw_counts[eid//100000]+=1
assert sum(raw_counts.values())==413
assert raw_counts[5]==raw_counts[8]==0
comparison=[]
for key,table in ref.grid_point_stresses_volume_direct.items():
 v=int(table.ogs_id);state.result_type=f'gp_vm_v{v}';gps=state.gp_stress_components()
 assert len(gps)==[47,47,16,16,16,16,23,23][v-1]
 assert set(gps)==set(map(int,table.node))
 values=np.array([[g.sxx,g.syy,g.szz,g.txy,g.tyz,g.txz,g.pressure,g.ovm] for nid in table.node for g in [gps[int(nid)]]])
 error=float(np.max(np.abs(values-table.data[-1])/np.maximum(np.abs(table.data[-1]),1)))
 comparison.append({'volume':v,'grids':len(gps),'reference_relative_difference':error})
ui=imgui.create_context();i=imgui.get_io();i.display_size=(1600,900);i.delta_time=1/60;i.backend_flags|=imgui.BackendFlags_.renderer_has_textures
i.set_ini_filename(None)
state.display_mode='contour'
for v in range(1,9):
 for comp in GP_COMPONENTS:
  key=f'{comp}_v{v}';state.result_type=key
  imgui.new_frame();draw_contour_toolbar(state,1600,900);draw_legend(state,1600,900);imgui.render()
  assert state.result_type==key,(key,state.result_type)
state.result_type='von_mises'
imgui.new_frame();draw_contour_toolbar(state,1600,900);draw_legend(state,1600,900);imgui.render()
assert state.result_type=='von_mises'
imgui.destroy_context(ui)
report={'dat':str(dat),'op2':str(actual),'reference_op2':str(reference),'nodes':len(state.model.nodes),'elements':len(state.model.elements),'gp_entries':204,'volumes':comparison,'app_loader_errors':[],'toolbar_and_legend_cases':89,'element_stress_records':sum(raw_counts.values()),'raw_oes_records_by_group':dict(raw_counts),'missing_element_stresses':missing}
if '--opengl' in sys.argv:
 import moderngl
 from renderer.mesh_renderer import MeshRenderer
 ctx=moderngl.create_standalone_context();renderer=MeshRenderer(ctx)
 for v in range(1,9):
  for comp in GP_COMPONENTS:
   renderer.upload(state.model,result,subcase=1,display_mode='contour',result_type=f'{comp}_v{v}')
   assert renderer._fill['3d'][2]>0
 ctx.release();report['opengl_upload_combinations']=88
Path('test/gpstress_volume_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
