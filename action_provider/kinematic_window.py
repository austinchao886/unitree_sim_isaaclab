"""Physics-rate kinematic evidence, independent of status/trace publish cadence.

Measurement only: no control filtering, admission thresholds, or safety bypass.
"""
from collections import deque
import math


class KinematicWindow:
    def __init__(self, session_id):
        self.session_id=session_id
        self.rows=deque(maxlen=512)
        self.error=None

    def add(self, simulation_s, velocity, height, tilt, torque_ratio):
        values=(simulation_s,*velocity,height,tilt,torque_ratio)
        if len(values)!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            self.rows.clear();self.error='invalid measurement';return
        if self.rows and simulation_s<=self.rows[-1][0]:
            self.rows.clear();self.error='nonmonotonic clock';return
        self.error=None
        self.rows.append(values)
        while self.rows and simulation_s-self.rows[0][0]>1.+1e-9:self.rows.popleft()

    def snapshot(self):
        result=dict(schema_version=1,session_id=self.session_id,ready=False,
                    error=self.error,sample_count=len(self.rows),scope='physics_kinematics_not_qualification')
        if len(self.rows)<2:return result
        rows=list(self.rows);end=rows[-1][0];span=end-rows[0][0]
        gaps=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
        result.update(start_simulation_s=rows[0][0],end_simulation_s=end,
                      span_s=span,max_sample_gap_s=max(gaps))
        if span<.995-1e-9 or max(gaps)>.006:return result
        # Time-weighted trapezoidal integration over the latest half second.
        # Keep mean speed AND norm of mean velocity: opposite motions must not
        # disappear from the path-speed diagnostic through vector cancellation.
        cutoff=end-.5;integral=[0.,0.,0.]
        for a,b in zip(rows,rows[1:]):
            lo=max(a[0],cutoff)
            if b[0]<=lo:continue
            u=(lo-a[0])/(b[0]-a[0]);dt=b[0]-lo
            va=(a[1],a[2],math.hypot(a[1],a[2]))
            vb=(b[1],b[2],math.hypot(b[1],b[2]))
            for i in range(3):integral[i]+=(va[i]+u*(vb[i]-va[i])+vb[i])*.5*dt
        mean=[x/.5 for x in integral]
        result.update(ready=True,mean_window_s=.5,mean_planar_velocity_m_s=mean[:2],
            mean_planar_velocity_norm_m_s=math.hypot(*mean[:2]),mean_planar_speed_m_s=mean[2],
            peak_planar_speed_m_s=max(math.hypot(r[1],r[2]) for r in rows),
            peak_abs_vertical_speed_m_s=max(abs(r[3]) for r in rows),
            min_height_m=min(r[4] for r in rows),max_height_m=max(r[4] for r in rows),
            peak_tilt_rad=max(r[5] for r in rows),peak_torque_ratio=max(r[6] for r in rows))
        return result
