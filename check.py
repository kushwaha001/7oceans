import json
import numpy as np

stats_path = 'data_genration_and_raw_data/raw_data/preprocessing/norm_stats/norm_stats.json'
mstats_path = 'data_genration_and_raw_data/raw_data/preprocessing/norm_stats/micro_norm_stats.json'                                                                                                      
                                                                                                                                                                                                           
print('=== norm_stats.json ===')                                                                                                                                                                         
with open(stats_path) as f:                                                                                                                                                                              
  ns = json.load(f)                                                                                                                                                                                    
  for k,v in ns.items():
      if 'lat' in k.lower() or 'lon' in k.lower() or 'mean' in k.lower() or 'std' in k.lower():                                                                                                        
          print(f'  {k} = {v}')                                                                                                                                                                        
   
print()                                                                                                                                                                                                  
print('=== micro_norm_stats.json ===')                                                                                                                                                                   
with open(mstats_path) as f:
  mns = json.load(f)                                                                                                                                                                                   
  for k,v in mns.items():
      if isinstance(v, (int, float)):
          print(f'  {k} = {v}')                                                                                                                                                                        
      elif isinstance(v, dict):
          print(f'  {k}:')                                                                                                                                                                             
          for k2,v2 in v.items():                                                                                                                                                                      
              print(f'      {k2} = {v2}')
                                                                                                                                                                                                           
print()         
print('=== correlation check (pred vs true) ===')
d = np.load('path_predictions/v2_1_sanity/path_predictions.npz', allow_pickle=True)                                                                                                                      
pl, pll = d['pred_lat'], d['pred_lon']                                                                                                                                                                   
tl, tll = d['true_lat'], d['true_lon']                                                                                                                                                                   
print(f'corr(pred_lat, true_lat) = {np.corrcoef(pl, tl)[0,1]:.4f}')                                                                                                                                      
print(f'corr(pred_lon, true_lon) = {np.corrcoef(pll, tll)[0,1]:.4f}')                                                                                                                                    
                                                                                                                                                                                                           
print()                                                                                                                                                                                                  
print('=== error by gap type ===')                                                                                                                                                                       
gt = d['gap_type']                                                                                                                                                                                       
si = d['sample_idx']
err = d['dist_err_km']                                                                                                                                                                                   
# map per-ping to gap_type via sample_idx                                                                                                                                                                
gt_per_ping = np.array([gt[i] for i in si])                                                                                                                                                              
for t in np.unique(gt_per_ping):                                                                                                                                                                         
    m = gt_per_ping == t                                                                                                                                                                                 
    print(f'  {t:15s}  n={m.sum():6d}  median_km={np.median(err[np.unique(si[m])]):.1f}  mean_km={err[np.unique(si[m])].mean():.1f}')
