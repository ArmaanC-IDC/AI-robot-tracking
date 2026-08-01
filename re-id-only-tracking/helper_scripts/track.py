import cv2
import numpy as np
import heapq

class Track:
    # embeddings = []
    # embedding_times = []

    # points = []
    # point_times = []

    # color = (255, 0, 255)
    # id = -1

    num_images_to_compare_to = 5

    def __init__(self, color, embeddings, id):
        self.color = color
        self.id = id
        self.embeddings = embeddings
        self.embedding_times = [0 for _ in embeddings]

        self.points = []
        self.point_times = []
        self.images = []
    
    def add_point(self, point, time, include_embedding=False, embedding=None, image=None):
        self.points.append(point)
        self.point_times.append(time)

        if include_embedding:
            self.embeddings.append(embedding)
            self.embedding_times.append(time)

            # resize images to be 128px tall
            target_height = 128
            h, w = image.shape[:2]
            if h != target_height:
                new_width = int(w * (target_height / h))
                image = cv2.resize(image, (new_width, target_height))

            self.images.append(image)

    def get_points(self):
        return self.points
    
    def get_embeddings(self):
        return self.embeddings

    def get_images(self):
        return self.images
    
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