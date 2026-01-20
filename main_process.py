import cv2
import numpy as np
from ultralytics import YOLO
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

from dataclasses import dataclass
import argparse
import json
import base64
import glfw
import math
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import os
from frame_renderer.window_manager import WindowManager
from frame_renderer.drawer import Drawer
from frame_renderer.fonts import Font
# ==================================================
# Configuración
# ==================================================

IMAGE_2D_PATH = "SOP30-Lateral_B.png"
MODEL_PATH = "edwards_insipiris_best_14jan.pt"

RESIZE_CAM_WIDTH = 1920
RESIZE_CAM_HEIGHT = 1080

CLASE_HILO = 6
CLASE_TELA = 0


################################################
### Render helpers
def draw_SOP10_1(canvas_size=(400, 400), needle_theta=0.0, needle_visible=False):
    img = np.ones((*canvas_size, 3), dtype=np.uint8) * 255

    cv2.putText(img, "SOP 10_1 ",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    return img

def draw_SOP30_1(canvas_size=(400, 400), needle_theta=0.0, needle_visible=False):
    img = np.ones((*canvas_size, 3), dtype=np.uint8) * 255

    h, w = canvas_size
    cx, cy = w // 2, h // 2
    R = int(min(w, h) * 0.35)

    # --- Círculo principal
    cv2.circle(img, (cx, cy), R, (0, 0, 0), 2)

    # --- Rectángulos (3 zonas)
    rect_w, rect_h = 40, 20
    angles = [0, 2*np.pi/3, 4*np.pi/3]

    for i, a in enumerate(angles):
        rx = int(cx + (R + 25) * np.cos(a) - rect_w / 2)
        ry = int(cy + (R + 25) * np.sin(a) - rect_h / 2)
        if i == 0:
            color = (0, 0, 255)  # rojo para la zona central
        else:
            color = (0, 0, 0)

        cv2.rectangle(img,
                      (rx, ry),
                      (rx + rect_w, ry + rect_h),
                      color, 2)

    # --- Aguja (línea)
    x_end = int(cx + R * np.cos(needle_theta))
    y_end = int(cy + R * np.sin(needle_theta))

    if needle_visible:
        cv2.line(img, (cx, cy), (x_end, y_end), (0, 0, 255), 3)

    cv2.putText(img, "TopCutting:SOP 30_1 - Frontal view",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    return img

def draw_SOP30_2(canvas_size=(400, 400),needle_visible=False, needle_x_norm=0.5):
    img = np.ones((*canvas_size, 3), dtype=np.uint8) * 255

    h, w = canvas_size

    # --- Cilindro (vista lateral)
    margin = 50
    cyl_rect = (margin, h//3, w - 2*margin, h//3)

    cv2.rectangle(img,
                  (cyl_rect[0], cyl_rect[1]),
                  (cyl_rect[0] + cyl_rect[2], cyl_rect[1] + cyl_rect[3]),
                  (0, 0, 0), 2)

    # --- Rectángulos internos (3 zonas)
    zone_w = cyl_rect[2] // 5
    zone_h = cyl_rect[3] // 2
    offsets = [0.2, 0.5, 0.8]

    for idx,o in enumerate(offsets):
        zx = int(cyl_rect[0] + o * cyl_rect[2] - zone_w / 2)
        zy = int(cyl_rect[1] + cyl_rect[3] / 2 - zone_h / 2)
        if idx == 0:
            color = (0, 0, 255)  # rojo para la zona central
        else:
            color = (0, 0, 0)
            
        cv2.rectangle(img,
                      (zx, zy),
                      (zx + zone_w, zy + zone_h),
                      color, 2)

    # --- Aguja (línea)
    nx = int(cyl_rect[0] + needle_x_norm * cyl_rect[2])
    if needle_visible:
        cv2.line(img,
                (nx, cyl_rect[1] - 30),
                (nx, cyl_rect[1] + cyl_rect[3] + 30),
                (0, 0, 255), 3)

    cv2.putText(img, "TopCutting:SOP 30_2 - Lateral view",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)

    return img


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=10, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2

    length = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(0, length, dash_len + gap):
        start = i
        end = min(i + dash_len, length)

        xs = int(x1 + (x2 - x1) * start / length)
        ys = int(y1 + (y2 - y1) * start / length)
        xe = int(x1 + (x2 - x1) * end / length)
        ye = int(y1 + (y2 - y1) * end / length)

        cv2.line(img, (xs, ys), (xe, ye), color, thickness)


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


def contour_center_line(contour):
    """
    Compute the center line (main axis) of a contour using PCA.

    Returns:
        center: (x, y)
        direction: (vx, vy) - unit vector
    """
    data_pts = contour.reshape(-1, 2).astype(np.float32)

    # Perform PCA
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean=np.array([]))

    # Extract center and main direction
    center = tuple(mean[0])
    direction = tuple(eigenvectors[0])  # main axis direction

    return center, direction


def draw_center_line(img,  center, direction, length, color = (0,0,255) ):
    
    cX, cY = int(center[0]), int(center[1])
    vx, vy = direction

    # Draw line
    pt1 = (int(cX - vx * length/2), int(cY - vy * length/2))
    pt2 = (int(cX + vx * length/2), int(cY + vy * length/2))
    cv2.line(img, pt1, pt2, color, 2)
   # cv2.circle(img, (cX, cY), 4, (255, 0, 0), -1)

    return img


################################################
### Classes
def estimate_pixel_to_cm(points,
                          real_dist_mm=0.01,
                          k=4):
    """
    Estima relación pixel → cm usando distancia promedio entre puntos

    Parámetros:
    - points: Nx2 array en pixeles
    - real_dist_mm: distancia real promedio entre puntos (mm)
    - k: vecinos más cercanos

    Retorna:
    - px_to_cm
    - cm_to_px
    - dist_px_mean
    """

    points = np.asarray(points, np.float32)

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)

    # eliminar distancia a sí mismo (col 0)
    neighbor_dists = dists[:, 1:]

    # promedio robusto
    dist_px_mean = np.median(neighbor_dists)

    # conversión real
    real_dist_cm = real_dist_mm * 0.1   # mm → cm

    px_to_cm = real_dist_cm / dist_px_mean
    cm_to_px = dist_px_mean / real_dist_cm

    return px_to_cm, cm_to_px, dist_px_mean

@dataclass
class StitchingAction:
    name: str
    video_path: str
    texture_path: str
    yaw: float
    tilt: float

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

        all_points = np.vstack([contour])
        if len(contour) > 0:
            self.rect = cv2.minAreaRect(all_points)  # ((cx, cy), (w, h), angle)
        
        self.box = cv2.boxPoints(self.rect)
        self.center, self.direction = contour_center_line(contour)

    def get_center_line(self, frame):
        _, (w, h), angle = self.rect
        length = max(w, h)
        
        cX, cY = int(self.center[0]), int(self.center[1])
        vx, vy = self.direction

    # Draw line
        pt1 = (int(cX - vx * length/2), int(cY - vy * length/2))
        pt2 = (int(cX + vx * length/2), int(cY + vy * length/2))
    
        return pt1, pt2

    def smooth(self, epsilon = 5.0):
           # higher → fewer points
        self.contour = cv2.approxPolyDP(self.contour, epsilon, True)

    def angle(self):
        dx, dy = self.direction
        angle = math.degrees(math.atan2(dy, dx))

        return angle

    def draw(self, frame, color = (0,0,255), width = 2):
         # Draw rectangle on original image
        cv2.drawContours(frame, [self.contour], 0, color, width)

     
###########################################
class Stitch(SegmentedObject):
    def __init__(self, contour, id):
        self.contour = contour
        self.id = id
        self.name = "stitch"
        all_points = np.vstack([contour])
        self.rect = cv2.minAreaRect(all_points)  # ((cx, cy), (w, h), angle)
        self.box = cv2.boxPoints(self.rect)
        self.center, self.direction = contour_center_line(contour)
        self.color = (255, 0, 255)
        self.aux_contours = []
        self.distances = [] # distance to centerline. Close to 0 is long
        self.std = 0
        self.avg = 0
        self.normalized_length = 0
        self.distance_to_prev_stitch = 0
        self.skeleton_points = []
       
       
    def angle(self):
        dx, dy = self.direction
        angle = math.degrees(math.atan2(dy, dx))

        return angle
    
###########################################

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

GLOBAL_KEY_BOARD =   0
RUNNING_APP = True

def on_key(window, key, scancode, action, mods):
    global RUNNING_APP, GLOBAL_KEY_BOARD
    if action == glfw.PRESS:
        print(f"Tecla presionada: {key}")
    elif action == glfw.RELEASE:
        print(f"Tecla liberada: {key}")

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        print("Escape pressed, closing window.")
        RUNNING_APP = False

def load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT,monitor_id = 0,maximized=False):
    print("=" * 60)
    print("Test: Renderizado de frames de diferentes tamaños")
    print("=" * 60)
    print(f"\nTamaño de ventana: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print("\nEste test verificará que el sistema adapta correctamente")
    print("frames de diferentes tamaños al tamaño de la ventana.\n")

    wm = WindowManager()

    monitors = glfw.get_monitors()
    print("Monitores detectados:")
    for i, monitor in enumerate(monitors):
        mode = glfw.get_video_mode(monitor)
        print(
            f"  Monitor {i}: {glfw.get_monitor_name(monitor)} - "
            f"{mode.size.width}x{mode.size.height} @ {mode.refresh_rate}Hz"
        )

    print(f"\nCreando ventana de {WINDOW_WIDTH}x{WINDOW_HEIGHT}...")
    window = wm.create_window(
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        "test_resize",
        position=(monitor_id*WINDOW_WIDTH, 100),
        maximized=maximized,
        decorators=False,
    )

    glfw.make_context_current(window.glfw_window)
    glfw.set_key_callback( window.glfw_window , on_key)

    return wm

### current data is hardcoded for testing purposes
def get_data_for_unity_sop1(frame):
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

def rotate_full(image, angle, border=(0,0,0), interp=cv2.INTER_CUBIC):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0,0])
    sin = abs(M[0,1])

    new_w = int(h*sin + w*cos)
    new_h = int(h*cos + w*sin)

    M[0,2] += (new_w / 2) - center[0]
    M[1,2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border)
        
#################################################################
def segment_from_mask(mask, shape):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 0):
        return {      "contour" : [] }
    h_orig, w_orig = shape[:2]
    sx = w_orig / mask.shape[1]
    sy = h_orig / mask.shape[0]
    contours_scaled = []
    for cnt in cnts:
        cnt = cnt.astype(np.float32)
        cnt[:, 0, 0] *= sx   # x
        cnt[:, 0, 1] *= sy   # y
        contours_scaled.append(cnt.astype(np.int32))
    mx_cnt = max(contours_scaled, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    if w < h:
        angle += 90

    return {
        "contour" : mx_cnt,
        "center": np.array([cx, cy]),
        "angle": angle,
        "length": max(w, h),
        "rect": rect
    }

def are_colinear(seg1, seg2, angle_tol=5, dist_tol=15):
    # 1. Orientación similar
    if seg1 is None or seg2 is None:
        return False

    if abs(seg1["angle"] - seg2["angle"]) > angle_tol:
        return False

    # 2. Distancia del centro de seg2 a la recta de seg1
    theta = np.deg2rad(seg1["angle"])
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-direction[1], direction[0]])

    dist = abs(np.dot(seg2["center"] - seg1["center"], normal))
    return dist < dist_tol

def merge_segments(segments):
    pts = []

    for seg in segments:
        rect = seg["rect"]
        box = cv2.boxPoints(rect)
        pts.extend(box)

    pts = np.array(pts)

    # Ajuste de línea (mínimos cuadrados)
    vx, vy, x0, y0 = cv2.fitLine(
        pts, cv2.DIST_L2, 0, 0.01, 0.01
    )

    # Proyección para extremos
    projections = []
    for p in pts:
        t = (p[0] - x0) * vx + (p[1] - y0) * vy
        projections.append(t)

    t_min, t_max = min(projections), max(projections)

    p1 = (int(x0 + vx * t_min), int(y0 + vy * t_min))
    p2 = (int(x0 + vx * t_max), int(y0 + vy * t_max))

    return p1, p2

def process_needle(needle_s, img):
    needle_segments = []

    for needle in needle_s:
        needle_segments.append(needle.contour)

    groups = []
    used = set()

    for i, s1 in enumerate(needle_segments):
        if i in used:
            continue

        group = [s1]
        used.add(i)

        for j, s2 in enumerate(needle_segments):
            if j in used:
                continue
            if are_colinear(s1, s2):
                group.append(s2)
                used.add(j)

        groups.append(group)

    for group in groups:
        if len(group) >= 2:
            p1, p2 = merge_segments(group)
            cv2.line(img, p1, p2, (0,255,0), 3)


#############################################################
def show_oriented_cloth(mask, img):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 0):
        return

    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    # Corrección estándar OpenCV
    if h < w:
        angle += 90

    (h_img, w_img) = img.shape[:2]
    center = (int(cx), int(cy))

    rot_img = rotate_full(img, angle - 180, border=(0, 0, 0), interp=cv2.INTER_CUBIC)
    rot_mask = rotate_full(mask, angle - 180, border=(0, 0, 0), interp=cv2.INTER_NEAREST)

    vis = rot_img.copy()

    cnts_rot, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_rot = max(cnts_rot, key=cv2.contourArea)

    cv2.drawContours(vis, [cnt_rot], -1, (0,255,0), 2)

    cv2.putText(
        vis,
        f"Aligned angle: {angle:.1f} deg",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2)
        
    cv2.imshow("Oriented Cloth", vis)

def filtrar_hilos_validos(hilos_contours,
                          contorno_tela,
                          inside_ratio_min=0.8,
                          length_factor=2.0,
                          n_samples=20):
    """
    Descarta hilos muy largos o que salen de la tela.

    Retorna:
    - hilos_filtrados
    """

    if len(hilos_contours) == 0:
        return []

    # --------------------------------------------------
    # 1) Longitudes
    lengths = np.array([
        cv2.arcLength(cnt, False)
        for cnt in hilos_contours
    ])

    median = np.median(lengths)
    mad = np.median(np.abs(lengths - median)) + 1e-6

    max_length = median + length_factor * mad

    hilos_validos = []

    # --------------------------------------------------
    # 2) Filtrado
    for cnt, L in zip(hilos_contours, lengths):

        # ---- (a) largo razonable
        if L > max_length:
            continue

        # ---- (b) porcentaje dentro de la tela
        pts = cnt.reshape(-1, 2)

        if len(pts) < 2:
            continue

        # sampleo uniforme
        idx = np.linspace(0, len(pts) - 1, n_samples).astype(int)

        inside = 0
        for i in idx:
            p = tuple(pts[i])
            if cv2.pointPolygonTest(contorno_tela, (int(pts[i][0]), int(pts[i][1])), False) >= 0:
                inside += 1

        inside_ratio = inside / len(idx)

        if inside_ratio < inside_ratio_min:
            continue

        hilos_validos.append(cnt)

    return hilos_validos

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
#################################################
###  Compute ribs
def compute_ribs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    _, bw = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cv2.imshow("Binarized", bw)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    holes = []
    min_area = 20
    max_area = 300

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            holes.append([centroids[i][0], centroids[i][1]])

    
    vis = frame.copy()

    for hole in holes:
        cx, cy = hole
        cv2.circle(vis, (int(cx), int(cy)), 2, (0,0,255), -1)

   # cv2.imshow("Huecos en tela segmentada", vis)
    return vis , holes


def serializeFrame(self, frame, resizeFactor = 1):
    scale = resizeFactor
    resizedFrame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) #Reduce frame size
    _,buffer = cv2.imencode(".jpg", resizedFrame)
    buffer_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')   #Serialized frame
    return buffer_b64
##############################################################333
def get_front_end_data(nFrame,  frame, SOP):
    try:
        sFrame = serializeFrame(None, frame, resizeFactor=0.5)
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

def flujo_denso(frame_prev, frame_curr,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                draw=True):
    """
    Flujo óptico denso entre dos frames consecutivos

    Parámetros:
    - frame_prev, frame_curr: imágenes BGR consecutivas
    - draw: visualiza el flujo sobre frame_curr

    Retorna:
    - frame_vis
    - flow (H x W x 2)
    - mag, ang
    """
    prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    frame_vis = frame_curr.copy()

    if draw:
        # muestreo para no saturar la imagen
        step = 16
        h, w = prev_gray.shape

        for y in range(0, h, step):
            for x in range(0, w, step):
                dx, dy = flow[y, x]
                if mag[y, x] < 0.2:
                    continue

                cv2.arrowedLine(
                    frame_vis,
                    (x, y),
                    (int(x + dx), int(y + dy)),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )

    return frame_vis, flow, mag, ang

class SOP_Manager:
    def __init__(self):
        self.current_sop = None
        self.model = YOLO(MODEL_PATH)
        self.yaw = 0.20
        self.tilt = 0.20
        self.sop_index = 10
        self.prev_cloth_frame = None

        self.renderer = CylinderRenderer(eps=1e-3)
        self.action_mgr=ActionManager(ACTIONS)
        self.tracking = 100
    
   
    def estimate_SOP(self, frame, frame_number, detections):
        best_conf = 0
        self.sop_index = 10

        results = self.model.predict(frame, conf=0.3, verbose=False)[0]

        self.tracking -= 1
        for i, cls_id in enumerate(results.boxes.cls):
            cls_name = self.model.names[int(cls_id)]
            if cls_name == "needle" or cls_name == "scissors":
                self.tracking += 1
                break

        return {"name": f"SOP{self.sop_index}", "index": 0, "end": frame_number ,"step_order" :self.sop_index,"orientation" : 0,"rank":best_conf, "index" : self.sop_index}
    
    ################################################
    def estimate_orientation_sop30(self,frame, detections, index):
        
        return index / 1000
    
    ################################################
    def estimate_orientation_sop10(self,frame, detections, index):

        return index / 1000
    
    def run_frame(self, frame,frame_number, tracker_mgr, review_mode):
        # ----------------------------------------------
        # ----------------------------------------------
        needle_visible = False

        mask_cloth = None
      #  
        if self.tracking <=0 :
            results = self.model.track(    frame,    persist=True,
                     conf=0.3,     classes=None,     # detecta todo
                        tracker="bytetrack.yaml",
                        verbose=False)[0]
   
                      # bbox de la tela
           
            annotated = results.plot()
            #cv2.imshow("tracking", annotated)
          
        else:
            results = self.model.predict(frame, conf=0.3, verbose=False)[0]
            self.valid_tracks = []

        x1, y1, x2, y2 = 0, 0, 0, 0
        
        needle_masks = []

        needle_contours = []

        cloth_contour = None

        self.detections = []
        
        #### compute all contours
        for i, cls_id in enumerate(results.boxes.cls):
            cls_name = self.model.names[int(cls_id)]
            mask_np = None
            box = None
            if cls_name == "thread":
                box = results.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            if cls_name == "metal_framework":
                box = results.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
            if cls_name == 'needle' and review_mode:
                needle_visible = True
                box = results.boxes.xyxy[i].cpu().numpy()
                mask = results.masks.data[i]
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                
            if cls_name == "cloth":
                mask = results.masks.data[i]
                box = results.boxes.xyxy[i].cpu().numpy()
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                mask_cloth = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                #frame = draw_segmentation(frame, results, model)
            if mask_np is not None:
                cnt = segment_from_mask(mask_np, frame.shape)["contour"]
                if len(cnt) == 0:
                    continue
                s = SegmentedObject(  box=box,  contour=cnt,name=cls_name,  color=(0, 255, 0)   )
                s.mask = mask_np
                s.smooth(epsilon=5.0)

                if cls_name == "needle":
                    needle_masks.append(s)
 
                self.detections.append(s)  
        
        ### calculate tracking
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
        
            ids = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            contours = results.masks.data.cpu().numpy() if results.masks is not None else None

            tela_boxes = boxes[classes == 0]
            tela_bbox = tela_boxes[0] if len(tela_boxes) > 0 else None

            self.valid_tracks = tracker_mgr.update(frame_number, boxes,contours, ids, classes, tela_bbox)
        
        frame_render = frame.copy() ##results.plot()
       
        ribs = None
        ########## compute ribs
        if mask_cloth is not None:
            frame_cloth = cv2.bitwise_and(frame, frame, mask=mask_cloth)
            frame_cloth, ribs = compute_ribs(frame_cloth)
            #### extract cloth
            show_oriented_cloth(mask_cloth, frame_cloth)
            
            #if  self.prev_cloth_frame is not None:
            #    frame_render, _, _, _, = flujo_denso(  self.prev_cloth_frame,   frame_cloth,   draw=True     )

        if ribs is not None:
            try:
                px_to_cm, cm_to_px, dist_px_mean = estimate_pixel_to_cm(ribs, real_dist_mm=1.0, k=4)
                #print(f"Distancia media: {dist_px_mean:.2f} px")
            except Exception as e:
                print(f"Exception computing grid: {e}")
        map_2d =  self.action_mgr.texture

        yaw = self.action_mgr.current.yaw
        tilt = self.action_mgr.current.tilt
        
        if not review_mode:
            process_needle(needle_masks, frame_render)

        thread_contours = [n.contour for n in self.detections if n.name == "thread"]
        self.cloth_contours = [n.contour for n in self.detections if n.name == "cloth" ]
        
        #################################################################3
        if len(thread_contours) >= 3 and len(self.cloth_contours) > 0:
            
            self.hilos_ok = filtrar_hilos_validos(
                 thread_contours, self.cloth_contours[0],
                  inside_ratio_min=0.85, length_factor=2.5)
        else:
            self.hilos_ok = []

            
        # ----------------------------------------------
        # Render cilindro
        # ----------------------------------------------
        # Render cilindro usando la acción actual
        if self.action_mgr.current.name == "sop10_1":
            schema = draw_SOP10_1(needle_theta=yaw, needle_visible=needle_visible)
        elif self.action_mgr.current.name == "sop30_1":
            schema = draw_SOP30_1(needle_theta=yaw, needle_visible=needle_visible)

        elif self.action_mgr.current.name == "sop30_2":
            schema = draw_SOP30_2(needle_x_norm=(yaw % (2*np.pi)) / (2*np.pi), needle_visible=needle_visible)

        ###
        for det in self.detections :
            if det.name == "thread":
                det.draw(frame_render)

      
         #### compute proposed guide line 
        if x1 > 0:
            line_x = x2 + 10  # 10 pixels below detected framework
            y_start = y1 - 50
            y_end = y2 + 50
            draw_dashed_line(frame_render, (line_x, y_start), (line_x, y_end), (0, 255, 0), thickness=2, dash_len=15, gap=10)

        self.prev_cloth_frame = frame_cloth.copy() if mask_cloth is not None else None

        return frame_render, map_2d, schema, self.detections 
        

def main_loop(video_path,monitor_id = 0, maximized = False, start_frame=18000, has_rectangle=False):
    global RUNNING_APP, GLOBAL_KEY_BOARD
    # ==================================================
    # Cargas
    # ==================================================
    if os.path.exists(video_path) is False:
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
   
    ## sop30 = first frame = 10000
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame )


    # ==================================================
    # Loop principal
    # ==================================================
   
    sop_Manager = SOP_Manager()
    action_mgr = sop_Manager.action_mgr

    WINDOW_HEIGHT=720
    WINDOW_WIDTH=1280
    wm=load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT, monitor_id=monitor_id, maximized=maximized)
    font = Font.get_font()
    summary_drawer = Drawer(font, WINDOW_HEIGHT, WINDOW_WIDTH)
        
    tracker_mgr = TrackerManager(    hilo_class_id = 6,    min_samples=10)
    # Acción inicial
    current_action = action_mgr.set_action("sop10_1")
    yaw = current_action.yaw
    tilt=current_action.tilt
    framework_yaw = 0.0
    #########################
    SOP = {"name": 0, "index":0, "step_order":0}

    step_frame = 2
    
    review_mode = True

    paused = False
    last_frame = None

    while cap.isOpened() and RUNNING_APP:

        if paused:
            frame = last_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
            
        original_frame = frame.copy()
        last_frame = frame.copy()

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (RESIZE_CAM_WIDTH, RESIZE_CAM_HEIGHT))

        if frame_idx % step_frame != 0:
            continue

        fh, fw = frame.shape[:2]
        if has_rectangle:
            cv2.rectangle(frame, (870, 500), (1280, 1050), (0, 0, 0), -1)  # background for text
        
        ### estimate SOP
        SOP = sop_Manager.estimate_SOP(frame, frame_idx, None)

        ### run detections
        frame_render, map_2d, schema, detections = sop_Manager.run_frame(frame,frame_idx,tracker_mgr, review_mode)

        ### estimate orientation
        if SOP["name"] == "SOP30":
            framework_yaw = sop_Manager.estimate_orientation_sop30(frame, detections, frame_idx)
        else:
            framework_yaw = sop_Manager.estimate_orientation_sop10(frame, detections, frame_idx)
        
        SOP["orientation"] = framework_yaw
        # ----------------------------------------------
        # Ajuste de tamaños
        # ----------------------------------------------
        right_w = schema.shape[1]

        map_resized = cv2.resize(map_2d, (right_w, fh // 2))
        cyl_resized = cv2.resize(schema, (right_w, fh // 2))

        right_stack = np.vstack([map_resized, cyl_resized])

        frame_resized = cv2.resize(frame_render, (fw, fh))

        # ----------------------------------------------
        # Composición final
        # ----------------------------------------------
        combined = np.hstack([frame_resized, right_stack])

        cv2.imshow("Frame | Imagen 2D + Cilindro 3D", combined)

        data = get_front_end_data(frame_idx,  frame, SOP)

        ###  Draw using lib
        summary_frame = np.zeros((RESIZE_CAM_HEIGHT, RESIZE_CAM_WIDTH, 3), dtype=np.uint8)
      #  summary_frame = cv2.resize(original_frame, (RESIZE_CAM_WIDTH, RESIZE_CAM_HEIGHT))
        for det in detections:
            if det.name == "cloth":
                det.draw(summary_frame, (255, 255, 255),width = 1)
            elif det.name == "thread":
                det.draw(summary_frame, (220, 55, 55),width = 2)
        summary_frame = cv2.flip(summary_frame, 0)
        summary_frame = cv2.cvtColor(summary_frame, cv2.COLOR_BGR2RGB)
        if sop_Manager.hilos_ok is not None and len(sop_Manager.hilos_ok) > 0 and len(sop_Manager.cloth_contours) > 0:
            try:
                render_guideline(summary_frame, sop_Manager.hilos_ok, sop_Manager.cloth_contours[0])
            except Exception as e:
                print(f"Exception rendering guideline: {e}")
        if sop_Manager.tracking <= 0:
            summary_drawer.draw_text(summary_frame, "TEST RESULTS", 50, 50, (0, 255, 0), scale=3)
        else:
            summary_drawer.draw_text(summary_frame, "WORKING", 50, 50, (0, 255, 0), scale=3)

        wm.render("test_resize", summary_frame)
        # ----------------------------------------------
        # Controles
        # ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            RUNNING_APP = False
            break
        elif key == ord('p'):  # +30 frames
            paused = not paused
        elif key == ord('f'):  # +30 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + 30)
        elif key == ord('a'):
            yaw -= 0.05
        elif key == ord('d'):
            yaw += 0.05
        elif key == ord('w'):
            tilt += 0.05
        elif key == ord('s'):
            tilt -= 0.05
        elif key == ord('v'):
            for tv in sop_manager.valid_tracks:
                print(tv)

        if key == ord('1'):
            current_action = action_mgr.set_action("sop30_1")
            yaw, tilt = current_action.yaw, current_action.tilt
            sop_Manager.renderer.last_img = None

        elif key == ord('2'):
            current_action = action_mgr.set_action("sop30_2")
            yaw, tilt = current_action.yaw, current_action.tilt
            sop_Manager.renderer.last_img = None

        elif key == ord('3'):
            current_action = action_mgr.set_action("sop10_1")
            yaw, tilt = current_action.yaw, current_action.tilt
            sop_Manager.renderer.last_img = None
    
    #################################
    cap.release()
    cv2.destroyAllWindows()
    #####
    try:
        glfw.terminate()
    except Exception as e:
        print(f"Exception terminating GLFW: {e}")
        pass 
###############################################################################
def parse_args():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Parse command-line arguments for video processing.")
   # Define command-line arguments with optional flags
    parser.add_argument("--port", default="5102", help="Port for exposing")
    parser.add_argument("--src", default="E:/Resources/Novathena/INSIPIRIS/operation_10A_cut.mp4", help="Video source")
    parser.add_argument("--device", default="", help="Video source")

    parser.add_argument("--monitor_id", default=0, help="Monitor ID")
    parser.add_argument("--start_frame", default=100, help="Enable testing")
    parser.add_argument("--maximized", default=False, help="Enable testing")
   
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


def get_mp4_from_path(path: str) -> str | None:
    """
    If 'path' is an mp4 file, return it.
    If 'path' is a directory, return the largest mp4 file contained inside.
    Returns None if no mp4 file is found.
    """

    # Normalize
    path = os.path.abspath(path)

    # Case 1: path is an mp4 file
    if os.path.isfile(path):
        if path.lower().endswith(".mp4"):
            return path
        else:
            return None

    # Case 2: path is a directory
    if os.path.isdir(path):
        mp4_files = []

        # Search recursively or only top-level?
        # If only inside the given folder (not recursive):
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isfile(full) and full.lower().endswith(".mp4"):
                size = os.path.getsize(full)
                mp4_files.append((size, full))

        if not mp4_files:
            return None

        # Select largest mp4
        mp4_files.sort(reverse=True)  # largest first
        return mp4_files[0][1]

    # Not a file nor directory
    return None

if __name__ == "__main__":

    args = parse_args()
    print(args)

    if args.device != "":
        main_loop(args.device, args.monitor_id, args.maximized, args.start_frame)
    elif not os.path.exists(args.src):
        print("video file not found. exit")
    else:
        mp4File = get_mp4_from_path(args.src)

        if mp4File is None:
            print("Failed to open file")
            exit()

        main_loop(mp4File, args.monitor_id, args.maximized, args.start_frame)