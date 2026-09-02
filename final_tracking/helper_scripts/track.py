import cv2
import numpy as np
import heapq
import inspect

class Track:
    # embeddings = []
    # embedding_times = []

    # points = []
    # point_times = []

    # color = (255, 0, 255)
    # id = -1

    #field is 570px wide, and 16.54m wide
    px_to_m = 570 / 16.54

    #account for errors in drawing the bounding boxes
    min_circle_radius = 30

    max_accel = 14 #in m/s^2
    max_vel = 6 #in m/s

    fps = 0

    num_images_to_compare_to = 5

    velocity_smoothing_factor = 0.5

    def __init__(self, color, embeddings, id, fps=None):
        self.color = color
        self.id = id
        self.embeddings = embeddings
        self.embedding_times = [0 for _ in embeddings]

        if fps is not None:
            Track.fps = fps

        self.points = []
        self.point_times = []
        self.images = []

        self.prev_vel = 0
    
    def add_point(self, point, time, include_embedding=False, embedding=None):
        self.points.append(point)
        self.point_times.append(time)

        if include_embedding:
            self.embeddings.append(embedding)
            self.embedding_times.append(time)
    
    def get_num_embeddings(self):
        return len(self.embeddings)
    
    def embedding_added_on_frame(self, count):
        return count in self.embedding_times
    
    def add_embedding(self, embedding, time):
        self.embeddings.append(embedding)
        self.embedding_times.append(time)

    def get_points(self): return self.points
    def get_point_times(self): return self.point_times
    def get_embeddings(self): return self.embeddings
    def get_images(self): return self.images
    
    def get_distance(self, base_embedding):            
        emb_dists = [np.linalg.norm(base_embedding - e) for e in self.embeddings]

        if self.num_images_to_compare_to >= len(self.embeddings):
            return np.mean(np.array(emb_dists))
        
        max_heap = [-x for x in emb_dists[:self.num_images_to_compare_to]]
        heapq.heapify(max_heap)
        
        for num in emb_dists[self.num_images_to_compare_to:]:
            if num < - max_heap[0]:
                heapq.heappushpop(max_heap, -num)
                
        return np.mean(np.array([-x for x in max_heap]))

    def get_current_vel(self, frame_num, use_for_dist=False):
        #if cannot find velocity (first point or has been lost for half second or more), 
        # assume max velocity for distance and 0 velocity for point
        if len(self.points)==0 or frame_num - self.point_times[-1] > Track.fps * 0.5:
            return np.array([Track.max_vel, 0]) if use_for_dist else np.array([0, 0])
        
        #if can't calculate velocity normally, return 0
        if len(self.points) < 2 or self.point_times[-1] - self.point_times[-2] == 0:
            return np.array([0.0, 0.0])
        
        #normally, calculate the velocity        
        current_vel = (self.points[-1] - self.points[-2]) / (self.point_times[-1] - self.point_times[-2])
        final_vel = current_vel * (1 - self.velocity_smoothing_factor) + self.prev_vel * self.velocity_smoothing_factor
        self.prev_vel = final_vel

        return final_vel
    
    def get_next_point(self, frame_num):
        if len(self.points) == 0:
            print("Error: track.py get_next_point called before any points added")

            return np.array([0, 0])

        if len(self.points) < 2:
            return self.points[-1]
        
        return self.points[-1] + (self.get_current_vel(frame_num) * (frame_num - self.point_times[-1]))
    
    def get_max_dist(self, frame_num):
        if len(self.points) == 0:
            print("Error: track.py get_next_point called before any points added")
            return Track.min_circle_radius
        
        vel = self.get_current_vel(frame_num, True)
        
        dt = (frame_num - self.point_times[-1])

        #acceleration in px/frame^2
        a = (self.max_accel * self.px_to_m) / (self.fps**2)

        return max(min(
            np.linalg.norm(vel)*dt + 0.5*a*(dt**2), #assuming max acceleration
            ((self.px_to_m * self.max_vel) / self.fps) * dt #assuming max velocity
        ), Track.min_circle_radius)