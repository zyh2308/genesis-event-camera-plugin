"""Display only packets that had actually arrived when the command was issued."""
import argparse,csv,json
from pathlib import Path
import cv2,imageio.v2 as imageio,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
 trace=json.loads((a.run/'episode_000/trace.json').read_text());sensor=a.run/'migration_audit/sensor_episode_000';packets=json.loads((sensor/'packet_audit.json').read_text());raw=np.load(sensor/'raw_events.npz')['events'];counts=np.cumsum([0]+[x['events'] for x in packets]);assert counts[-1]==len(raw);available=np.array([x['available_s'] for x in packets]);rows=[];cv2.setNumThreads(2)
 movie=a.output/'arrived_events_and_estimate.mp4'
 with imageio.get_writer(movie,fps=25,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
  for row in trace:
   now=row['t_before'];j=int(np.searchsorted(available,now+1e-9,side='right'));state=row['event_state'];held=bool(state and state['source'].startswith('event_acquisition_then_'));image=np.zeros((280,320,3),np.uint8);last_end=packets[j-1]['time_s'] if j else None;shown=np.empty((0,4))
   if j and not held:
    first=j-1
    while first>0 and packets[first-1]['time_s']>=last_end-.04-1e-9:first-=1
    candidate=raw[counts[first]:counts[j]];shown=candidate[(candidate[:,0]>=last_end-.04-1e-9)&(candidate[:,0]<=last_end+1e-9)];assert all(x['available_s']<=now+1e-9 for x in packets[first:j])
    if len(shown):
     assert shown[:,0].max()<=last_end+1e-6 and last_end<=now+1e-9;x=np.clip((shown[:,1]/2).astype(int),0,319);y=np.clip((shown[:,2]/2).astype(int),0,239)+40;positive=shown[:,3]>0;image[y[positive],x[positive]]=[245,90,65];image[y[~positive],x[~positive]]=[60,170,250]
    if row['predicted_event_uv']:
     u,v=row['predicted_event_uv']
     if 0<=u<640 and 0<=v<480:cv2.circle(image,(round(u/2),round(v/2)+40),10,(55,240,105),1)
   if held:cv2.putText(image,'HAND STATE HOLD',(55,142),cv2.FONT_HERSHEY_SIMPLEX,.55,(150,210,190),1)
   cv2.putText(image,'CONTROL t=%.2fs'%now,(6,15),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1)
   label='No event packet arrived' if last_end is None else 'Packet end %.2fs | age %.0fms'%(last_end,(now-last_end)*1000)
   cv2.putText(image,label,(6,32),cv2.FONT_HERSHEY_SIMPLEX,.39,(230,230,230),1);writer.append_data(image);rows.append({'control_t_before':now,'scene_frame_t_after':row['t'],'arrived_packets':j,'arrived_raw_events':int(counts[j]),'latest_arrived_packet_end':last_end,'displayed_events':len(shown),'maximum_displayed_event_s':float(shown[:,0].max()) if len(shown) else None,'held_by_proprioception':held})
 with (a.output/'visualization_causal_audit.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'event_panel_provenance.json').write_text(json.dumps({'source_run':str(a.run),'raw_events':len(raw),'frames':len(trace),'window_s':.04,'selection':'only actual arrived packet prefixes; window ends at latest arrived packet, not an assumed time-minus-delay cutoff','green_circle':'logged current control estimate, not ground truth','alignment':'frame i contains the input for the action whose resulting scene state is trace[i].t','held_state':'no new event input displayed'},indent=2));print(json.dumps({'movie':str(movie),'frames':len(trace),'raw_events':len(raw)}))

if __name__=='__main__':main()
