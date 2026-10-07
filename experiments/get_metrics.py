import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def normalize_detections(raw_detections):
    """
    Normalizes detections based on your specific JSON structure:
    - id: string or int
    - map_point: [x, y] coordinates
    - calculated_embedding: optional embedding vector
    """
    dets = {}
    for det in raw_detections:
        obj_id = det.get("id")
        map_pt = det.get("map_point")
        calculated_embedding = det.get("calculated_embedding")
        
        if obj_id is not None and map_pt is not None and len(map_pt) >= 2:
            clean_id = str(obj_id)
            dets[clean_id] = {
                "map_point": np.array(map_pt, dtype=float),
                "calculated_embedding": calculated_embedding
            }
            
    return dets

def load_json_annotations(json_path):
    """Loads JSON file and maps frame index to its normalized detections."""
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found.")
        return {}
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    frame_map = {}
    for entry in data:
        frame_idx = entry.get("frame")
        if frame_idx is not None:
            frame_map[frame_idx] = normalize_detections(entry.get("detections", []))
            
    return frame_map


class MOTEvaluator:
    """
    Modular Multi-Object Tracking Evaluator.
    Designed for clarity, correctness, and easy extension of new metrics.
    """
    def __init__(self, gt_json_path, pred_json_path, distance_threshold=30.0):
        self.distance_threshold = distance_threshold
        self.gt_frames = load_json_annotations(gt_json_path)
        self.pred_frames = load_json_annotations(pred_json_path)
        
        # Raw accumulation counters
        self.total_gt = 0
        self.total_pred = 0
        self.total_matches = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_ids = 0
        self.total_distance = 0.0
        self.total_predicted_points = 0
        self.embeddings_calculated = 0
        
        # Tracking structures for AssA & HOTA
        self.global_counts = {}   # gt_id -> {pred_id: count}
        self.gt_totals = {}       # gt_id -> total occurrences
        self.pred_totals = {}     # pred_id -> total occurrences
        
        # History for ID switches
        self.id_history = {}      # gt_id -> last matched pred_id

    def evaluate(self):
        """Runs the frame-by-frame evaluation loop and computes all metrics."""
        all_frames = sorted(list(set(self.gt_frames.keys()).union(set(self.pred_frames.keys()))))
        
        for frame_idx in all_frames:
            gt_dets = self.gt_frames.get(frame_idx, {})
            pred_dets = self.pred_frames.get(frame_idx, {})
            
            num_gt = len(gt_dets)
            num_pred = len(pred_dets)
            self.total_gt += num_gt
            self.total_pred += num_pred
            
            # Update totals for association stats
            for g_id in gt_dets:
                self.gt_totals[g_id] = self.gt_totals.get(g_id, 0) + 1
            for p_id in pred_dets:
                self.pred_totals[p_id] = self.pred_totals.get(p_id, 0) + 1

                self.total_predicted_points += 1
                if pred_dets[p_id]["calculated_embedding"]:
                    self.embeddings_calculated += 1

                
            # Handle empty frame edge cases
            if num_gt == 0 and num_pred == 0:
                continue
            if num_gt == 0:
                self.total_fp += num_pred
                continue
            if num_pred == 0:
                self.total_fn += num_gt
                self.id_history.clear()
                continue
                
            gt_ids = list(gt_dets.keys())
            pred_ids = list(pred_dets.keys())
            
            # 1. Build Cost Matrix using Euclidean distance on map_points
            cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, g_id in enumerate(gt_ids):
                for j, p_id in enumerate(pred_ids):
                    cost_matrix[i, j] = np.linalg.norm(
                        gt_dets[g_id]["map_point"] - pred_dets[p_id]["map_point"]
                    )
                    
            # 2. Hungarian Algorithm Matching for current frame
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_gt = set()
            matched_pred = set()
            current_frame_ids = {}
            
            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                if dist <= self.distance_threshold:
                    g_id = gt_ids[r]
                    p_id = pred_ids[c]
                    
                    matched_gt.add(g_id)
                    matched_pred.add(p_id)
                    
                    self.total_distance += dist
                    self.total_matches += 1
                    current_frame_ids[g_id] = p_id
                    
                    # Record global co-occurrence for AssA
                    if g_id not in self.global_counts:
                        self.global_counts[g_id] = {}
                    self.global_counts[g_id][p_id] = self.global_counts[g_id].get(p_id, 0) + 1
                    
                    # Check for Identity Switch (IDS)
                    if g_id in self.id_history and self.id_history[g_id] != p_id:
                        self.total_ids += 1
                        
            # Accounting for False Negatives and False Positives in current frame
            self.total_fn += (num_gt - len(matched_gt))
            self.total_fp += (num_pred - len(matched_pred))
            self.id_history = current_frame_ids

        return self.compute_metrics()

    def compute_metrics(self):
        """Computes final benchmark scores based on accumulated statistics."""
        # --- Association Accuracy (AssA) ---
        all_gt_ids = list(self.gt_totals.keys())
        all_pred_ids = list(self.pred_totals.keys())
        
        if all_gt_ids and all_pred_ids:
            ass_cost_matrix = np.zeros((len(all_gt_ids), len(all_pred_ids)))
            for i, g_id in enumerate(all_gt_ids):
                for j, p_id in enumerate(all_pred_ids):
                    count = self.global_counts.get(g_id, {}).get(p_id, 0)
                    ass_cost_matrix[i, j] = -count # Maximize counts -> minimize negative counts
                    
            g_inds, p_inds = linear_sum_assignment(ass_cost_matrix)
            global_matching = {all_gt_ids[g]: all_pred_ids[p] for g, p in zip(g_inds, p_inds)}
        else:
            global_matching = {}

        association_scores = []
        for g_id, p_matches in self.global_counts.items():
            matched_p_id = global_matching.get(g_id)
            for p_id, tpa in p_matches.items():
                if p_id == matched_p_id:
                    fna = self.gt_totals.get(g_id, 0) - tpa
                    fpa = self.pred_totals.get(p_id, 0) - tpa
                    assoc_score = tpa / max(1, tpa + fna + fpa)
                    for _ in range(tpa):
                        association_scores.append(assoc_score)

        assa = np.mean(association_scores) if association_scores else 0.0
        deta = self.total_matches / max(1, self.total_matches + self.total_fn + self.total_fp)
        hota = np.sqrt(deta * assa)

        # --- Standard Metrics ---
        mota = 1.0 - (float(self.total_fn + self.total_fp + self.total_ids) / max(1, self.total_gt))
        motp = (self.total_distance / self.total_matches) if self.total_matches > 0 else 0.0
        precision = self.total_matches / max(1, self.total_matches + self.total_fp)
        recall = self.total_matches / max(1, self.total_gt)
        f1 = 2 * (precision * recall) / max(1e-6, precision + recall)

        percent_embeddings = self.embeddings_calculated / self.total_predicted_points

        metrics = {
            "total_gt": self.total_gt,
            "total_matches": self.total_matches,
            "false_positives": self.total_fp,
            "false_negatives": self.total_fn,
            "identity_switches": self.total_ids,
            "mota": mota,
            "motp": motp,
            "assa": assa,
            "deta": deta,
            "hota": hota,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "percent_embeddings": percent_embeddings
        }
        return metrics

    def print_report(self):
        """Prints a clean summary report of the evaluation metrics."""
        metrics = self.evaluate()
        
        print("\n" + "="*45)
        print("        MULTI-OBJECT TRACKING BENCHMARKS")
        print("="*45)
        print(f"Total Ground Truth Detections : {metrics['total_gt']}")
        print(f"Total Matches                 : {metrics['total_matches']}")
        print(f"False Positives (FP)          : {metrics['false_positives']}")
        print(f"False Negatives (FN)          : {metrics['false_negatives']}")
        print(f"Identity Switches (IDS)       : {metrics['identity_switches']}")
        print("-" * 45)
        print(f"MOTA (Accuracy)               : {metrics['mota']:.4f} ({metrics['mota'] * 100:.2f}%)")
        print(f"MOTP (Precision / Distance)   : {metrics['motp']:.2f} pixels")
        print(f"AssA (Association Accuracy)   : {metrics['assa']:.4f} ({metrics['assa'] * 100:.2f}%)")
        print(f"DetA (Detection Accuracy)     : {metrics['deta']:.4f} ({metrics['deta'] * 100:.2f}%)")
        print(f"HOTA                          : {metrics['hota']:.4f} ({metrics['hota'] * 100:.2f}%)")
        print(f"Precision                     : {metrics['precision']:.4f}")
        print(f"Recall                        : {metrics['recall']:.4f}")
        print(f"F1 Score                      : {metrics['f1']:.4f}")
        print(f"% pred points emb calculated  : {metrics['percent_embeddings']:.4f} ({metrics['percent_embeddings'] * 100:.2f}%)")
        print("="*45)

if __name__ == "__main__":
    GT_JSON = "annotations/2026cancmp_f1m3/ground_truth.json"
    PRED_JSON = "annotations/2026cancmp_f1m3/test2/iom=0.4.json"
    THRESHOLD_PIXELS = 30.0
    
    evaluator = MOTEvaluator(GT_JSON, PRED_JSON, THRESHOLD_PIXELS)
    evaluator.print_report()