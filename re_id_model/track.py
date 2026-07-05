import numpy as np
class Track:
    #field is 570px wide, and 16.54m wide
    px_to_m = 570 / 16.54

    #20px to account for errors in drawing the bounding boxes
    min_circle_radius = 30

    max_accel = 20 #in m/s^2
    max_vel = 7 #in m/s

    fps = 0

    def __init__(self, fps, color, id):
        Track.fps = fps
        self.color = color

        self.id = id

        self.prev_points = []
        self.prev_point_frame = 0
        self.prev_point_frame_2 = -1
        self.is_lost = False
        self.last_found_frame = 0

        self.pts_in_range = []
    
    def add_point(self, point, frame):
        self.prev_points.append(point)
        self.last_found_frame = frame
        self.prev_point_frame_2 = self.prev_point_frame
        self.prev_point_frame = frame
    
    def mark_as_lost(self):
        self.is_lost = True
    
    def mark_as_found(self, frame):
        self.is_lost = False
    
    def get_current_vel(self):
        if len(self.prev_points) <= 2 or self.prev_point_frame - self.prev_point_frame_2 == 0:
            return np.array([0, 0])
        return (self.prev_points[-1] - self.prev_points[-2]) / (self.prev_point_frame - self.prev_point_frame_2)
    
    def get_next_point(self, frame):
        if len(self.prev_points) == 0:
            print("Error: track.py get_next_point called before any points added")
            return np.array([0, 0])

        if len(self.prev_points) <= 2:
            return self.prev_points[-1]
        
        return self.prev_points[-1] + (self.get_current_vel() * (frame - self.last_found_frame))
    
    def get_max_dist(self, frame):
        dt = (frame - self.last_found_frame)

        if len(self.prev_points) == 0:
            print("Error: track.py get_max_dist called before any points added")
            return max(((self.px_to_m * self.max_vel) / self.fps) * dt, Track.min_circle_radius)

        #acceleration in px/frame^2
        a = (self.max_accel * self.px_to_m) / (self.fps**2)

        return max(min(
            np.linalg.norm(self.get_current_vel()*dt + 0.5*a*(dt**2)), #assuming max acceleration
            ((self.px_to_m * self.max_vel) / self.fps) * dt #assuming max velocity
        ), Track.min_circle_radius)