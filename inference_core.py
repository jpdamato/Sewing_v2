import cv2
import numpy as np
from ultralytics import YOLO
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
from dataclasses import dataclass
import json

import math
import platform
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import os
from frame_renderer.window_manager import WindowManager
from frame_renderer.drawer import Drawer
from frame_renderer.fonts import Font
import tools as tools
import helpers as helpers
import time
from classes import ConsecutiveEventFSM,StitchEvent, StitchingAction, SegmentedObject, CylinderRenderer, ActionManager , TrackerManager

# ==================================================
# Configuración
# ==================================================

RESIZE_CAM_WIDTH = 1280
RESIZE_CAM_HEIGHT = 720

CLASE_HILO = 6
CLASE_TELA = 0

MINIMAL_THREAD_LENGTH = 100

#######
###RULES
## SOP 10 : 16. if a needle is detected in parts, it is doing an stitch, and we save it
##          we save these stitches and compute measurement
#           17. after 7/8 and if we detect a thread, we have a 360 around            
### SOP 30 :
###  Each time we detect scissor for several frames, we are preparing cloth
### after scissor dissapear and needle is detected, we start are running tasks 8,9,10
###  Task 8 if cloth is centered and circle
### Task 9 if cloth is lateral 

ACTIONS = {
    "sop30_1": StitchingAction(
        name="sop30_1",
        texture_path="SOP30-Lateral_B.png",
        video_path = "E:/Resources/Novathena/INSIPIRIS/INSPIRIS Stent ID30.mp4",
        yaw=0.0,
        tilt=-np.pi/2 ,
    ),
    "sop30_2": StitchingAction(
        name="sop30_2",
        video_path = "E:/Resources/Novathena/INSIPIRIS/INSPIRIS Stent ID30.mp4",
        texture_path="SOP30-Lateral_B.png",
        yaw=np.pi / 2,   # distinta orientación
        tilt=0.0
    ),
    "sop10_1": StitchingAction(
        name="sop10_1",
        texture_path="SOP10_1.png",
        video_path = "E:/Resources/Novathena/INSIPIRIS/INSPIRIS Stent ID10.mp4",
        yaw=0.0,
        tilt=-0.3
    )
}

STATES = [ "Preparing" , "Sewing" , "Checking"  ]

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def select_best_pair(detections, frame_shape,
                     w_dist=1.0, w_center=0.5,
                     max_pair_dist=None):
    """
    Elige el par de objetos de distinta clase:
    - más cercanos entre sí
    - mejor centrados en el frame

    frame_shape: (H, W)
    """

    H, W = frame_shape[:2]
    frame_center = np.array([W / 2, H / 2])

    best_score = np.inf
    best_pair = None

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):

            d1, d2 = detections[i], detections[j]

            # --- clases distintas
            if (d1.name ==  d2.name) :
                continue
            
            if not ((d1.name == "cloth" or d2.name== "cloth") and (d1.name == "framework" or d2.name  == "framework") ) :
                continue
            
            c1 = bbox_center(d1.box)
            c2 = bbox_center(d2.box)

            # --- distancia entre objetos
            dist_pair = np.linalg.norm(c1 - c2)

            if max_pair_dist is not None and dist_pair > max_pair_dist:
                continue

            # --- centrado respecto al frame (promedio)
            dist_center = (
                np.linalg.norm(c1 - frame_center) +
                np.linalg.norm(c2 - frame_center)
            ) / 2

            # --- score combinado
            score = w_dist * dist_pair + w_center * dist_center

            if score < best_score:
                best_score = score
                best_pair = (d1, d2)

    return best_pair, best_score

def render_cylinder(texture, out_size=(400, 400),
                         yaw=0.0, tilt=0.0,
                         step_u=2, step_v=2):
    Ht, Wt = texture.shape[:2]
    Hr, Wr = out_size

    render = np.zeros((Hr, Wr, 3), dtype=np.uint8)

    cx = Wr // 2
    R = int(Wr * 0.35)

    for v in range(0, Ht, step_v):
        z_norm = v / Ht
        y_base = int(z_norm * Hr)

        for u in range(0, Wt, step_u):
            theta = 2 * np.pi * (u / Wt)
            theta_rot = theta + yaw

            x = int(cx + R * np.sin(theta_rot))
            y = y_base + int(np.sin(tilt) * np.cos(theta_rot) * R)

            if 0 <= x < Wr and 0 <= y < Hr:
                render[y, x] = texture[v, u]

    return render


#######################################################
def intersect_line_segment(p, d, a, b, eps=1e-6):
    """
    Intersección entre recta infinita P + t*d y segmento AB
    Retorna punto o None
    """
    p = np.array(p, dtype=np.float32)
    d = np.array(d, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    r = d
    s = b - a

    rxs = np.cross(r, s)
    q_p = a - p

    if abs(rxs) < eps:
        return None  # paralelos

    t = np.cross(q_p, s) / rxs
    u = np.cross(q_p, r) / rxs

    if 0.0 <= u <= 1.0:
        return p + t * r

    return None

def intersect_line_contour(center, direction, contour):
    """
    Retorna lista de puntos donde la recta intersecta el contorno
    """
    intersections = []

    p = np.array(center, dtype=np.float32)
    d = np.array(direction, dtype=np.float32)

    n = len(contour)
    for i in range(n):
        a = contour[i]
        b = contour[(i + 1) % n]

        pt = intersect_line_segment(p, d, a, b)
        if pt is not None:
            intersections.append(pt)

    return intersections


#
def draw_segmentation(frame, result, model, alpha=0.4):
    overlay = frame.copy()

    if result.masks is None:
        return frame

    for i, poly in enumerate(result.masks.xy):
        cls_id = int(result.boxes.cls[i])
        cls_name = model.names[cls_id]

        # Convertir a int
        poly = poly.astype(np.int32)

        # Color por clase (determinístico)
        color = (
            (cls_id * 53) % 255,
            (cls_id * 97) % 255,
            (cls_id * 193) % 255
        )

        # Relleno semitransparente
        cv2.fillPoly(overlay, [poly], color)

        # Contorno
        cv2.polylines(frame, [poly], True, color, 2)

        # Texto (clase)
        x, y = poly[0]
        cv2.putText(frame, cls_name, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # Mezcla alpha
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


### BUG JUAN : current data is not fully implemented
#  for testing purposes
def get_data_for_unity_sop30(sop_Manager,frame):
    return [], []


### BUG JUAN : current data is not fully implemented
#  for testing purposes
def get_data_for_unity_sop10(sop_Manager,frame):
    
    data = {}
    dthreas = []
    point_ids = []
    idx = 0
    for stitch_event in sop_Manager.stitches_events:
        thread =stitch_event.hide_stitch

        dthreas.append({"id": idx,
                    "center": [float(thread.center[0]), float(thread.center[1])],
                    "length": float(thread.length),
                    "has_devitation" : False,
                    "deviationDistance": 0.0122350436, # distance to next thread
                    "deviationLength": 0.149589762, # expected length
                    "deviationAngle": 2.6627202 }   ) # deviation angle
        point_ids.append(idx)
        idx += 1

    data["threads"] = dthreas
    
    return data, point_ids
 

def get_data_for_unity_sop2(frame):
    # Placeholder function to simulate data extraction for Unity
    # In a real scenario, this would involve image processing to find points of interest
    height, width = frame.shape[:2]
    points = [
        [width // 4, height // 4],
        [width // 2, height // 2],
        [3 * width // 4, 3 * height // 4]
    ]
    point_ids = [1, 2, 3]

    data = {}
    threads = [{ "id": 4,
                    "deviationDistance": 0.0122350436,
                    "deviationLength": 0.149589762,
                    "deviationAngle": 2.6627202,
                    "deviations": [] }    ]


    data["threads"] = threads
    
    return data, point_ids

def get_data_for_unity_sop3(frame):
    return {}, []
def render_guideline(frame, hilos_contours, contorno_tela,
                     color=(0, 255, 0), thickness=3,
                            smooth=1.0,
                            n_samples=200):
   
    """
    Renderiza una spline suave que pasa por el centro de los hilos
    y se extiende hasta el contorno de la tela.

    smooth: 0 → pasa por los puntos
            >0 → más suavizado visual
    """

    out = frame.copy()

    # --------------------------------------------------
    # 1) Centros de hilos
    centers = []
    for cnt in hilos_contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            centers.append([
                M["m10"] / M["m00"],
                M["m01"] / M["m00"]
            ])

    if len(centers) < 3:
        return out

    centers = np.array(centers, dtype=np.float32)

    # --------------------------------------------------
    # 2) Recta principal (solo para ordenar, no se dibuja)
    vx, vy, x0, y0 = cv2.fitLine(
        centers,
        cv2.DIST_L2,
        0, 0.01, 0.01
    )

    v = np.array([vx, vy]).reshape(2)
    p0 = np.array([x0, y0]).reshape(2)

    # ordenar sobre la recta
    t = (centers - p0) @ v
    order = np.argsort(t)
    centers = centers[order]

    # --------------------------------------------------
    # 3) Spline paramétrica
    x = centers[:, 0]
    y = centers[:, 1]

    # k=3 → cúbica, pero requiere >=4 puntos
    k = 3 if len(centers) >= 4 else 2

    tck, _ = splprep([x, y], s=smooth, k=k)

    u = np.linspace(0, 1, n_samples)
    xs, ys = splev(u, tck)
    spline = np.stack([xs, ys], axis=1)

    # --------------------------------------------------
    # 4) Extender spline hasta contorno (usando tangente)
    dir_start = spline[1] - spline[0]
    dir_start /= np.linalg.norm(dir_start)

    dir_end = spline[-1] - spline[-2]
    dir_end /= np.linalg.norm(dir_end)

    def intersect_ray(p, d, contour, L=5000):
        p1 = p + d * L
        best = None
        min_proj = np.inf

        for i in range(len(contour)):
            a = contour[i][0]
            b = contour[(i + 1) % len(contour)][0]

            hit, q1, q2 = cv2.clipLine(
                (int(a[0]), int(a[1]), int(b[0]), int(b[1])),
                (int(p[0]), int(p[1])),
                (int(p1[0]), int(p1[1]))
            )

            if hit:
                q = np.array(q1, dtype=np.float32)
                proj = np.dot(q - p, d)
                if proj > 0 and proj < min_proj:
                    min_proj = proj
                    best = q

        return best

    p_start = intersect_ray(spline[0], -dir_start, contorno_tela)
    p_end   = intersect_ray(spline[-1], dir_end, contorno_tela)

    if p_start is not None:
        spline = np.vstack([p_start, spline])
    if p_end is not None:
        spline = np.vstack([spline, p_end])

    # --------------------------------------------------
    # 5) Render spline
    cv2.polylines(
        frame,
        [spline.astype(np.int32)],
        False,
        color,
        thickness
    )

    return frame



def contour_closest_to_screen_center(detections,  frame_shape):
    """
    contours: lista de contornos OpenCV (cada uno Nx1x2 o Nx2)
    frame_shape: shape del frame (H,W) o (H,W,C)

    return:
        best_contour: contorno elegido (o None)
        best_point: (x,y) punto del contorno más cercano al centro (o None)
        best_dist: distancia euclídea al centro (float, o inf)
    """
    if detections is None or len(detections) == 0:
        return None, None, float("inf")

    h, w = frame_shape[:2]
    center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    best_contour = None
    best_point = None
    best_dist = float("inf")

    for det in detections:
        if det.name != "needle":
            continue
        cnt = det.contour
        if cnt is None or len(cnt) == 0:
            continue

        cnt = [det.start , det.center, det.end]
        # Asegurar forma Nx2
        pts = np.squeeze(cnt)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        pts = pts.astype(np.float32)

        # Distancia de todos los puntos al centro
        dists = np.linalg.norm(pts - center, axis=1)

        # Punto más cercano de este contorno al centro
        idx = int(np.argmin(dists))
        dmin = float(dists[idx])
        pmin = tuple(map(int, pts[idx]))

        # Comparar contra el mejor global
        if dmin < best_dist:
            best_dist = dmin
            best_point = pmin
            best_contour = cnt

    return det, best_point, best_dist
##############################################################333
def get_front_end_data(nFrame,  frame, SOP):
    try:
        sFrame = tools.serializeFrame(None, frame, resizeFactor=0.5)
        Data = {}
        #### 
        Data["id"] = nFrame                    
        Data["frame"] = sFrame
        Data["sop"] = SOP["name"]
        Data["step_number"] = SOP["index"]
        Data["step_order"] = SOP["step_order"]

        unity_data1 , unity_points_ids = get_data_for_unity_sop1(frame)
        unity_data2, _ = get_data_for_unity_sop2(frame)
        unity_data3 , _ = get_data_for_unity_sop3(frame)
        ## bug still hardcoded points
        Data["sop_data"] = {"sop_1": unity_data1,
                            "sop_2": unity_data2,
                            "sop_3": unity_data3}
                            
        Data["unity_points_ids"] = unity_points_ids
        Data["grading_table"] = ""
     
        return Data

    except Exception as e:
        print (f"Exception at exporting : {e}")

class SOP_Manager:
    def __init__(self, model_path):
        self.current_sop = None
        
        #self.yaw = 0.20
        #self.tilt = 0.20
        self.sop_index = 10
        
        self.current_state = STATES[0]
        self.event_detect_thread= ConsecutiveEventFSM(target_class="thread")
        self.event_detect_scissor= ConsecutiveEventFSM(target_class="scissor")
        
        self.prev_cloth_frame = None
        self.detections = []
        self.px_to_cm, self.cm_to_px, self.dist_px_mean = 0,0,0
        self.renderer = CylinderRenderer(eps=1e-3)
        self.action_mgr=ActionManager(ACTIONS)
        self.tracking=100
        self.render_ribs = False

        self.distance_estimator = tools.CLAHEThread()
        ### handle stitches events
        self.stitches_events = []
        self.active_event =None
        self.original_frame = None
        self.timeout_sec = 10.0
        self.status = False # user is working or idle
        self.tracked_stitches = {}
        self.tracker_mgr = TrackerManager(    hilo_class_id = 6,    min_samples=10)
        # Acción inicial
        self.current_action = self.action_mgr.set_action("sop10_1")
        self.rendered_frame = None
        self.model = YOLO(model_path)
        self.cloth = None
        self.metal_framework = None

        self.guideline = helpers.VisualGuideline(color=(39,127,255),offset=100, curvature = 0.001)

   
    def estimate_state(self, frame, detections):
        global STATES

        self.event_detect_thread.update(detections)
        self.event_detect_scissor.update(detections)

        for det in self.detections:
            if self.current_state == STATES[0]  : ### preparing:
                if self.event_detect_thread.changed :
                    self.current_state = STATES[1] 
                    break
                    
            if self.current_state == STATES[1] : ### sewing:
                
                if self.event_detect_scissor.changed:
                    self.current_state = STATES[2] ### review work 
                    break

        return self.current_state

    def estimate_SOP(self, frame, frame_number, detections,  sop_index):
        best_conf = 0
        ### Hardcoded SOP 10
        self.sop_index = sop_index
        results = self.detections

        if self.sop_index == 10:
            self.step_order = 16
        
            self.tracking -= 1
            ##use previous detections
            for det in self.detections:
                if det.name == "needle" or det.name == "scissor":
                    self.tracking += 1
                    break
            #############################
            if self.tracking <= 0 : 
                self.step_order = 17
            else:
                self.step_order = 16
        else:
            self.step_order =  8
        
        self.SOP = {"name": f"SOP{self.sop_index}", "index": 0, "end": frame_number ,"step_order" :self.step_order,"orientation" : 0,"rank":best_conf, "index" : self.sop_index}
     
        return self.SOP 
    ################################################
    def estimate_orientation_sop30(self,frame, detections, index):
        
        return index / 1000
    
    ################################################
    def estimate_orientation_sop10(self,frame, detections, index):

        return index / 1000
    
    def run_frame(self, SOP,frame,frame_number,  review_mode, maximized = False):
        
        prev_time = time.perf_counter()
        tools.startProcess("inference")

        ### run detections
        if SOP["name"] == "SOP30":
           
            frame_render, _, _, detections = self.run_frame30(frame,frame_number, review_mode)
            framework_yaw = self.estimate_orientation_sop30(frame, detections, frame_number)
        else:
            frame_render, _, _, detections = self.run_frame10(frame,frame_number, review_mode)
       
            framework_yaw = self.estimate_orientation_sop10(frame, detections, frame_number)
        tools.endProcess("inference")

        tools.startProcess("orientation")
            
        
        SOP["idle"] = self.status
        # ----------------------------------------------
        # Composición final
        # ----------------------------------------------
        #combined = np.hstack([frame_resized, right_stack])
        combined =  frame_render.copy()
        
        if len(self.distance_estimator.ribs) > 10 and self.render_ribs:
            self.distance_estimator.draw(combined, color = tools.material_colors["ribs"] )

        ### SOP 10 Helpers
        if SOP["name"] == "SOP10":   
            if self.cloth is not None:
                self.guideline.update(self.cloth, self.metal_framework)
                #render_perpendicular_curved_guideline(combined, self.cloth_contours,)

            #pts_in = np.array(sop_Manager.ribs, dtype=np.float32)  #helpers.filter_points_inside_contour(sop_Manager.ribs,sop_Manager.cloth_contours)
            #ellipse = helpers.fit_robust_ellipse(pts_in, keep_ratio=0.8)

            #if ellipse is not None:
            #    if helpers.ellipse_inside_contour(ellipse, sop_Manager.cloth_contours):
            #        helpers.draw_dashed_ellipse(  combined,  ellipse,   color=(255, 255, 0),
            #                thickness=2,   dash_length=12,    gap_length=8)
        tools.endProcess("orientation")

        # ---- timing ----
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time

        fps = 1.0 / dt if dt > 0 else 0.0

        # ---- texto ----
        text_fps = f"FPS: {fps:.2f}"
        text_dt = f"Infer time: {dt*1000:.1f} ms"
        text_dt = f"Infer time: {dt*1000:.1f} ms"

        cv2.putText(            combined, text_fps,            (10, 30),            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,            (0, 255, 0),            2,            cv2.LINE_AA        )

        cv2.putText(            combined, text_dt,            (10, 60),            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,            (0, 255, 255),            2,            cv2.LINE_AA        )
        cv2.putText(            combined, f"frame :{frame_number}",            (10, 80),            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,            (0, 255, 255),            2,            cv2.LINE_AA        )

        
        if  platform.system() == "Windows":
            cv2.imshow("Frame | Imagen 2D + Cilindro 3D", combined)
        
        self.rendered_frame = combined.copy()

        if (len (self.stitches_events) > 0):
            strip = helpers.render_event_grid(self.stitches_events,cols=4, scale = 0.325)
            if strip is not None and platform.system() == "Windows":
                cv2.imshow("Eventos", strip)

        return combined

    def run_frame30(self, frame,frame_number, tracker_mgr, review_mode, maximized = False):
        tools.startProcess("Yolo")
      #  
        results = self.model.predict(frame, conf=0.3, verbose=False,imgsz=320)
        tools.endProcess("Yolo")
        

        tools.startProcess("Segmentation")
                    # bbox de la tela
        self.detections = []
        self.scissors = False
        #### compute other contours        
        for result in results:
            if result.masks is None:
                continue
                
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]
                
                cls_name = result.names[int(box.cls[0])],
                if cls_name == "scissor":
                    self.scissor = True
                
                if mask is not None:
                    mask = mask.data[0].cpu().numpy()
                    mask_np = (mask* 255).astype(np.uint8)
                
                #frame = draw_segmentation(frame, results, model)
                if mask_np is not None:
                    segment =  tools.segment_from_mask(mask_np, frame.shape)
                    cnt =segment["contour"]
                    if len(cnt) == 0:
                        continue
                    s = SegmentedObject(  box=box,  contour=cnt,name=cls_name,  color=(0, 255, 0)   )
                    s.mask = mask_np
                    s.track_id = 0
                    s.smooth(epsilon=5.0)
                    self.detections.append(s)


        #cv2.imshow("tracking", annotated)
        self.ribs =[]
        tools.endProcess("Segmentation")

        annotations = results[0].plot()
        if platform.system() == "Windows":
            cv2.imshow("annotations", annotations)
        frame_render = frame.copy()
        if self.scissors:
            cv2.circle(frame_render,(1000,50), 20, (200,100,100), -1)

        best_pair, best_score = select_best_pair(self.detections, frame.shape)
        ## choose 
        if best_pair is not None:
            d1, d2 = best_pair
            d1.draw_contours(frame_render)
            d2.draw_contours(frame_render)

        return frame_render, None, None, self.detections
    

    def render_needle_next_position(self, frame_render):
        best_cnt, best_pt, best_dist = contour_closest_to_screen_center(self.detections, frame.shape)
        if best_cnt is not None and best_dist < 150:
            d = 1000
            if self.metal_framework is not None:
                d = tools.dist(best_cnt.center, self.metal_framework.center)

            if d < 200:
                cv2.circle(frame_render, best_pt, 15, (0, 200, 55), -1)  # punto más cercano al centro
       
            else:    
                cv2.circle(frame_render, best_pt, 15, (0, 55, 200), -1)  # punto más cercano al centro
       

    ###########################################################################################
    def run_frame10(self, frame,frame_number,  review_mode, maximized = False):
        # ----------------------------------------------
        # ----------------------------------------------
        mask_cloth = None
        annotated = None

        tools.startProcess("Yolo")      #  
        results = self.model.predict(frame, conf=0.1, verbose=False)[0]
        
        #results = self.model.track(    frame,    persist=True, 
        #            conf=0.3,     classes=None,     # detecta todo
        #            tracker="bytetrack.yaml",
        #            verbose=False)[0]

                    # bbox de la tela
        try:
            annotated = results.plot()
        #cv2.imshow("tracking", annotated)
        except:
            annotated = None
        tools.endProcess("Yolo")
        x1, y1, x2, y2 = 0, 0, 0, 0
        
        needle_masks = []
        needle_contours = []
        cloth_contour = None
        self.detections = []
        self.cloth_visible = False
        self.metal_framework = None
        
        #### compute cloth 
        tools.startProcess("merge_cloth_masks")
        mask_cloth = tools.merge_cloth_masks(results, cloth_class_id=0, frame_shape=frame.shape)
        cloth_contour, cloth_box = tools.extract_cloth_contour_and_bbox(mask_cloth)
        tools.endProcess("merge_cloth_masks")
        
        
        tools.startProcess("convert_to_objects")
        
        if cloth_contour is not None:
            cloth = SegmentedObject(  box=cloth_box,  contour=cloth_contour,name="cloth",  color=(0, 255, 0)   )
            cloth.mask = mask_cloth
            cloth.smooth(epsilon=5.0)
            self.cloth = cloth
            self.detections.append(cloth)
        else:
            self.cloth = None
            self.cloth_visible = False
        
        if mask_cloth is not None:
            self.distance_estimator.update(frame, mask_cloth)

        #### compute other contours        
        for i, cls_id in enumerate(results.boxes.cls):
            cls_name = self.model.names[int(cls_id)]
            mask_np = None
            box = results.boxes.xyxy[i].cpu().numpy()
            conf = results.boxes.conf[i].cpu().numpy()

            if cls_name == "thread" and conf > 0.3:
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            if cls_name == "metal_framework":
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
            if cls_name == "framework":
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
            if cls_name == 'needle' and review_mode and conf > 0.3:
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
                #frame = draw_segmentation(frame, results, model)
            if mask_np is not None :
                segment =  tools.segment_from_mask(mask_np, frame.shape)
                cnt =segment["contour"]
                if len(cnt) == 0:
                    continue
                s = SegmentedObject(  box=box,  contour=cnt,name=cls_name,  color=(0, 255, 0)   )
                s.mask = mask_np
                s.track_id = results.boxes.id[i].cpu().numpy().astype(int) if results.boxes.id is not None else -1
                s.smooth(epsilon=5.0)

                if cls_name == "metal_framework" :
                    self.metal_framework = s

                if cls_name == "framework" and self.metal_framework is None:
                    self.metal_framework = s
                
                if cls_name == "needle":
                    needle_masks.append(s)
                    s.segment = segment
                ### check if thread is valid
                #if cls_name == "thread":
                   # straight = tools.is_straight_segment(cnt)
                   # valids = tools.filtrar_hilos_validos([cnt], cloth_contour)
               
                self.detections.append(s)  
        
        if self.cloth is not None:
            for det in self.detections:
                if det.name == "thread":
                    if tools.box_intersection_area(det.box , self.cloth.box)>0.75 and det.length < MINIMAL_THREAD_LENGTH:
                        det.valid = True
                    else:
                        det.valid = False

        tools.endProcess("convert_to_objects")
        
        ### calculate tracking
        tools.startProcess("tracking")
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
        
            ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            contours = results.masks.data.cpu().numpy() if results.masks is not None else None

            tela_boxes = boxes[classes == 0]
            tela_bbox = tela_boxes[0] if len(tela_boxes) > 0 else None
            if self.tracker_mgr is not None:
                self.valid_tracks = self.tracker_mgr.update(frame_number, boxes, contours, ids, classes, tela_bbox)
            
            for v in self.valid_tracks:
                tid = v["track_id"]

                if tid in self.tracked_stitches:
                    continue
                selected = [n for n in self.detections if n.track_id == tid]
                if len(selected) == 0:
                    continue
                self.tracked_stitches[tid] = selected[0]
                self.tracked_stitches[tid].track_id = tid
                self.tracked_stitches[tid].compute_features()

        tools.endProcess("tracking")
        
        
        frame_render = frame.copy() ##results.plot()
        self.ribs = self.distance_estimator.ribs
            #### extract cloth
            #helpers.show_oriented_cloth(mask_cloth, frame, is_maximized =maximized)
            
            #if  self.prev_cloth_frame is not None:
            #    frame_render, _, _, _, = flujo_denso(  self.prev_cloth_frame,   frame_cloth,   draw=True     )

        if self.distance_estimator is not None:
            try:
                self.cm_to_px = self.distance_estimator.cm_to_px
                self.dist_px_mean = self.distance_estimator.dist_px_mean
                self.px_to_cm  = self.distance_estimator.dist_px_mean
                #print(f"Distancia media: {dist_px_mean:.2f} px")
            except Exception as e:
                print(f"Exception computing grid: {e}")
        
        map_2d =  self.action_mgr.texture
        ################ SOP 10 AND SEWING
        tools.startProcess("post_process1")
            
        if self.SOP["name"] == "SOP10" and self.current_state == STATES[1] :
            
            ### stitch event
            if len(needle_masks)>1:
                possible_stitches = tools.process_needle(needle_masks, frame_render)
                #############################################################################
                if len(possible_stitches)>0:
                    if self.active_event is None:
                        # crear nuevo evento
                        self.active_event = StitchEvent(len(self.stitches_events), frame_number,frame,possible_stitches[0])
                        self.active_event.get_hide_stitch()
                        ## save current measurements
                        self.active_event.px_to_cm =  self.px_to_cm
                        self.active_event.cm_to_px = self.active_event.cm_to_px
                        self.active_event.dist_px_mean= self.dist_px_mean

                        for det in self.detections:
                            if det.name == "thread":
                                self.active_event.threads.append(det)
                        
                        if platform.system() == "Windows":
                            self.active_event.draw(frame_render)
                            self.active_event.frame = frame_render.copy()
                           # cv2.imshow(f"new stitch{self.active_event.event_id}", frame_render)
                    else:
                        # continuar evento
                        self.active_event.add_frame(frame_number,possible_stitches[0],self.detections)

            # ---- chequear expiración ----
            if self.active_event is not None:
                if self.active_event.is_expired(self.timeout_sec):
                # self.closed_events.append(self.active_event)
                    self.stitches_events.append(self.active_event)
                    self.active_event = None
        tools.endProcess("post_process1")

        tools.startProcess("post_process2")
        #################################################################
        thread_contours = [n.contour for n in self.detections if n.name == "thread"]
        self.cloth_contours = cloth_contour
        
        if len(thread_contours) >= 2 and cloth_contour is not None:            
            self.hilos_ok = tools.filtrar_hilos_validos(
                thread_contours, cloth_contour,
                inside_ratio_min=0.85, length_factor=2.5)
        else:
            self.hilos_ok = []
        tools.endProcess("post_process2")
        
        # ----------------------------------------------
        # Rendering
        # ----------------------------------------------
        tools.startProcess("rendering")

        if self.active_event is not None:
            cv2.circle(frame, (50,1000), 3, (0,255,0))
            self.active_event.draw(frame)
        ### render helpers
        #if self.cloth_visible:
        #    if self.step_order ==16:
        #        helpers.draw_helper_SOP10_Task16(frame_render, self.detections ,self)
        
        if annotated is not None and platform.system() == "Windows":
            cv2.imshow("tracking", annotated)

        ## check if needle is close to a good position
        self.render_needle_next_position(frame_render)
      
        self.distance_estimator.draw(frame_render, tools.material_colors["ribs"])

        if self.metal_framework is not None:
            self.metal_framework.draw(frame_render, color = tools.material_colors["metal_framework"])
        
        ### Render straight threads
        for det in self.detections :
            if det.name == "thread":
             ## BUG JUAN
             ##   det.compute_intersection_contour(self.cloth_contours)
                if det.valid :
                    det.draw(frame_render, color = (232,155,30), width=2)
                else:
                    det.draw(frame_render, color = (0,0,255), width = 1)
            elif det.name in tools.material_colors:            
                det.draw_contours(frame_render, color = tools.material_colors[det.name])
      
        tools.endProcess("rendering")

        
 
        return frame_render, map_2d, None, self.detections 
        
