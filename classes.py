import cv2
import numpy as np
from ultralytics import YOLO
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
from dataclasses import dataclass
import json

import math
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import os
from frame_renderer.window_manager import WindowManager
from frame_renderer.drawer import Drawer
from frame_renderer.fonts import Font
import tools as tools
import helpers as helpers
import time

@dataclass
class StitchingAction:
    name: str
    video_path: str
    texture_path: str
    yaw: float
    tilt: float

#################################################################
class TrackerManager:

    def __init__(self, hilo_class_id, min_samples=20):
        self.hilo_class_id = hilo_class_id
        self.min_samples = min_samples
        self.history = {}

    def update(self,frame_id, boxes,contours, ids, classes, tela_bbox=None):

        for box, tid, cls in zip(boxes, ids, classes):

            if cls != self.hilo_class_id:
                continue

            if tid not in self.history:
                self.history[tid] = []

            self.history[tid].append(box)

        # ---- filtrar tracks válidos
        valid_tracks = []

        for tid, boxes in self.history.items():

            if len(boxes) < self.min_samples:
                continue

            if tela_bbox is not None:
                if not self._inside_tela(boxes, tela_bbox):
                    continue
            cn = boxes[-1][0:2] + (boxes[-1][2:4] - boxes[-1][0:2]) / 2

            valid_tracks.append({
                "track_id": tid, "history_lengh" : len(boxes),
                "boxes": boxes, "frame" : frame_id,
                  "last_box": boxes[-1],
                  "center" : cn,
                  "length": 100,
                   "distance_to_origin": 50
            })

        return valid_tracks

    @staticmethod
    def _inside_tela(boxes, tela_bbox):
        x1t, y1t, x2t, y2t = tela_bbox
        for b in boxes:
            x1,y1,x2,y2 = b
            if not (x1>=x1t and y1>=y1t and x2<=x2t and y2<=y2t):
                return False
        return True
        """
        Verifica que TODAS las boxes estén dentro de la tela
        """
        x1_t, y1_t, x2_t, y2_t = tela_bbox

        for b in boxes:
            x1, y1, x2, y2 = b
            if not (
                x1 >= x1_t and y1 >= y1_t and
                x2 <= x2_t and y2 <= y2_t
            ):
                return False

        return True
    # --------------------------------------------------
class ActionManager:
    def __init__(self, actions):
        self.actions = actions
        self.current = None
        self.texture = None

    def set_action(self, name):
        action = self.actions[name]

        # Cargar textura solo si cambia
        if self.current is None or action.texture_path != self.current.texture_path:
            self.texture = cv2.imread(action.texture_path)

        self.current = action
        return action

class CylinderRenderer:
    def __init__(self, eps=1e-4):
        self.last_yaw = None
        self.last_tilt = None
        self.last_img = None
        self.eps = eps

    def render(self, texture, out_size, yaw, tilt):
        if (
            self.last_img is not None and
            abs(yaw - self.last_yaw) < self.eps and
            abs(tilt - self.last_tilt) < self.eps
        ):
            return self.last_img

        img = render_cylinder(
            texture=texture,
            out_size=out_size,
            yaw=yaw,
            tilt=tilt
        )

        self.last_yaw = yaw
        self.last_tilt = tilt
        self.last_img = img
        return img

#############################
## class representing objects segmented (cloth, threads, gloves, ). Could have several contours
class SegmentedObject:
    def __init__(self,box, contour, name, color):
        self.contour = contour
        self.name = name
        self.color = color
        self.rect = None
        self.intersections = []
        self.valid = True
        self.track_id = -1
        self.length = 0

        all_points = np.vstack([contour])
        if len(contour) > 0:
            self.rect = cv2.minAreaRect(all_points)  # ((cx, cy), (w, h), angle)
            _, (w, h), angle = self.rect
            self.length = max(w, h)
        
        self.box = box
        self.center, self.direction = tools.contour_center_line(contour)

    def get_center_line(self):
        _, (w, h), angle = self.rect
        self.length = max(w, h)
        
        cX, cY = int(self.center[0]), int(self.center[1])
        vx, vy = self.direction

    # Draw line
        pt1 = (int(cX - vx * self.length/2), int(cY - vy * self.length/2))
        pt2 = (int(cX + vx * self.length/2), int(cY + vy * self.length/2))
    
        return pt1, pt2

    def smooth(self, epsilon = 5.0):
           # higher → fewer points
        self.contour = cv2.approxPolyDP(self.contour, epsilon, True)

    def compute_intersection_contour(self, contour):
        if contour is None:
            self.intersections = []
        else:
            vx, vy = self.direction
            perp_dir = (-vy, vx)

            self.intersections = intersect_line_contour(self.center, perp_dir, contour)
        return self.intersections

    def angle(self):
        dx, dy = self.direction
        angle = math.degrees(math.atan2(dy, dx))

        return angle

    
    def draw_contours(self, frame, color=(0, 255, 0), width=2):
        cv2.drawContours(frame, [self.contour], 0, color, width)

    def compute_features(self):
        pass

    def draw(self, frame, color=(0, 0, 255), width=2):
        self.get_center_line()

        tools.draw_center_line(frame, self.center, self.direction, max(self.rect[1]), color)

        for pt in self.intersections:
            cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
    
         # Draw rectangle on original image
        #cv2.drawContours(frame, [self.contour], 0, color, width)

     
###########################################
class Stitch(SegmentedObject):
    def __init__(self, contour, id):
        self.contour = contour
        self.id = id
        self.name = "stitch"
        all_points = np.vstack([contour])
        self.rect = cv2.minAreaRect(all_points)  # ((cx, cy), (w, h), angle)
        self.box = cv2.boxPoints(self.rect)
        self.center, self.direction = tools.contour_center_line(contour)
        self.color = (255, 0, 255)
        self.aux_contours = []
        self.distances = [] # distance to centerline. Close to 0 is long
        self.std = 0
        self.avg = 0
        self.length = self.rect
        self.normalized_length = 0
        self.distance_to_prev_stitch = 0
        self.skeleton_points = []
        self.get_center_line()
       
       
    def angle(self):
        dx, dy = self.direction
        angle = math.degrees(math.atan2(dy, dx))

        return angle
    
    def compute_features(self):
        pass

class StitchEvent:
    def __init__(self, event_id, start_frame, frame, segment):
        self.event_id = event_id
        self.start_frame = start_frame
        self.frames = [start_frame]
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.frame = frame
        self.segment =segment
        self.px_to_cm = 0
        self.cm_to_px = 0
        self.dist_px_mean = 0
    #########################################
    ## segment has estimated pos + Needle detection
    def get_hide_stitch(self, frame):
        p1,p2, group = self.segment

        for s in group:
            p1,p2 = s.get_center_line()
            cv2.line(frame, p1, p2, (0,200,110), 1)

        
        if len(group)>=2:
            p01,p02 = group[0].get_center_line()
            p11,p12 = group[1].get_center_line()
           
            cv2.line(frame, p02, p11, (0,255,110), 2)
            contour = np.array([p02,p11], dtype=np.int32).reshape((-1, 1, 2))
            self.hide_stitch = Stitch(contour , self.event_id)

            return [p02, p11]

        
        return []

    def add_frame(self, frame_number):
        self.frames.append(frame_number)
        self.last_time = time.perf_counter()

    def is_expired(self, timeout_sec):
        return (time.perf_counter() - self.last_time) > timeout_sec
