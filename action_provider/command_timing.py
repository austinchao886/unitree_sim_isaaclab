"""Cumulative physics-consumer command timing, not a DDS packet capture."""
import math


class CommandTiming:
    thresholds_s=(.1,.25,.5,1.)

    def __init__(self):
        self.last_received=None
        self.accepted=0
        self.age_samples=0
        self.max_interval_s=0.
        self.max_age_s=0.
        self.intervals_over=[0]*len(self.thresholds_s)
        self.ages_over=[0]*len(self.thresholds_s)

    def accept(self, received_at):
        if not math.isfinite(received_at) or received_at<0:
            raise ValueError("invalid receive time")
        if self.last_received is not None:
            if received_at<=self.last_received:raise ValueError("receive time did not advance")
            interval=received_at-self.last_received
            self.max_interval_s=max(self.max_interval_s,interval)
            for i,threshold in enumerate(self.thresholds_s):
                self.intervals_over[i]+=int(interval>threshold)
        self.last_received=received_at
        self.accepted+=1

    def observe_age(self, age_s):
        if not math.isfinite(age_s) or age_s<0:raise ValueError("invalid command age")
        self.age_samples+=1
        self.max_age_s=max(self.max_age_s,age_s)
        for i,threshold in enumerate(self.thresholds_s):
            self.ages_over[i]+=int(age_s>threshold)

    def snapshot(self):
        return dict(scope="physics_consumer_observations_since_provider_start",
            accepted_count=self.accepted,age_observation_count=self.age_samples,
            max_accepted_interval_s=self.max_interval_s,max_observed_age_s=self.max_age_s,
            thresholds_s=list(self.thresholds_s),interval_counts_over=list(self.intervals_over),
            age_observation_counts_over=list(self.ages_over))
