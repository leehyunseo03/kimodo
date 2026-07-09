# Generate Trajectory (kimodo conda)
```
python3 kimodo_sonic/generate_to_target_motion_lib.py \
  --duration 10 \
  --root_waypoints 21 \
  --forward_meters 5 \
  --append_target_hold 30 
```

# Metrics (gear-sonic container)
```
LIVESTREAM=2 /workspace/isaaclab/isaaclab.sh -p kimodo_sonic/kimodo_metrics.py --livestream 2
```

# GearSonic Record
```
./kimodo_sonic/run_gearsonic_kimodo_eval.sh
```

# Kimodo-GearSonic Metrics (kimodo conda)
```
python3 kimodo_sonic/kimodo_metrics.py \
  --recording ../GR00T-WholeBodyControl/metrics/kimodosonic/motion \
  --marker-json ../GR00T-WholeBodyControl/kimodo_sonic/motion/visualization/kimodo_to_target_forward_5m_hand_raise_trajectory_markers.json \
  --qpos ../GR00T-WholeBodyControl/kimodo_sonic/motion/qpos/kimodo_to_target_forward_5m_hand_raise.npz \
  --target ../GR00T-WholeBodyControl/kimodo_sonic/motion/target_reference/forward_5m_hand_raise_target.npz \
  --out-dir /home/hslee/IsaacLab_ws/kimodo/kimodo_sonic/metrics \
  --unit mm
```
