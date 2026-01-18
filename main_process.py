import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import argparse
import json
import base64
import glfw
from frame_renderer.window_manager import WindowManager
from frame_renderer.drawer import Drawer
from frame_renderer.fonts import Font
# ==================================================
# Configuración
# ==================================================

VIDEO_PATH = "E:/Resources/Novathena/INSIPIRIS/INSPIRIS Stent ID10.mp4"
IMAGE_2D_PATH = "SOP30-Lateral_B.png"
MODEL_PATH = "edwards_insipiris_best_14jan.pt"
USE_YOLO = True            # poner True si querés detección
AGUJA_CLASS = "needle"

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

################################################
### Classes

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

def load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT):
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
    wm.create_window(
        WINDOW_HEIGHT,
        WINDOW_WIDTH,
        "test_resize",
        position=(100, 100),
        maximized=False,
        decorators=False,
    )

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
def segment_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 0):
        return None
    cnt = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    if w < h:
        angle += 90

    return {
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

def process_needle(masks_de_vara, img):
    vara_segments = []

    for mask in masks_de_vara:
        vara_segments.append(segment_from_mask(mask))

    groups = []
    used = set()

    for i, s1 in enumerate(vara_segments):
        if i in used:
            continue

        group = [s1]
        used.add(i)

        for j, s2 in enumerate(vara_segments):
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

    cv2.imshow("Binarized", bw)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    holes = []
    min_area = 20
    max_area = 300

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            holes.append(i)

    print("Huecos detectados:", len(holes))

    vis = frame.copy()

    for i in holes:
        cx, cy = centroids[i]
        cv2.circle(vis, (int(cx), int(cy)), 2, (0,0,255), -1)

    cv2.imshow("Huecos en tela segmentada", vis)
    return vis


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

def main_loop():
    # ==================================================
    # Cargas
    # ==================================================
    cap = cv2.VideoCapture(VIDEO_PATH)
   

    if USE_YOLO:
        model = YOLO(MODEL_PATH)

    yaw = 0.20
    tilt = 0.20

    renderer = CylinderRenderer(eps=1e-3)
    ## sop30 = first frame = 10000
    cap.set(cv2.CAP_PROP_POS_FRAMES, 18000)

    # ==================================================
    # Loop principal
    # ==================================================
    action_mgr = ActionManager(ACTIONS)
    
    WINDOW_HEIGHT=720
    WINDOW_WIDTH=1280
    wm=load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT)
    font = Font.get_font()
    summary_drawer = Drawer(font, WINDOW_HEIGHT, WINDOW_WIDTH)
        

    # Acción inicial
    current_action = action_mgr.set_action("sop10_1")
    yaw = current_action.yaw
    tilt=current_action.tilt

    #########################
    SOP = {"name": 0, "index":0, "step_order":0}

    has_rectangle=True
    step_frame=3
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_idx % step_frame != 0:
            continue

        fh, fw = frame.shape[:2]
        if has_rectangle:
            cv2.rectangle(frame, (870, 500), (1280, 1050), (0, 0, 0), -1)  # background for text
        
       
        # ----------------------------------------------
        # YOLO (opcional)
        # ----------------------------------------------
        needle_visible = False

        mask_total = np.zeros(frame.shape[:2], dtype=np.uint8)

        if USE_YOLO:
            results = model(frame, verbose=False)[0]

            x1, y1, x2, y2 = 0, 0, 0, 0
            
            needle_masks = []
            
            for i, cls_id in enumerate(results.boxes.cls):
                cls_name = model.names[int(cls_id)]
                if cls_name == "metal_framework":
                    box = results.boxes.xyxy[i].cpu().numpy()

                    x1, y1, x2, y2 = map(int, box)        
                if cls_name == 'needle':
                    needle_visible = True
                    mask = results.masks.data[i]
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                    needle_masks.append(mask_np)
                    
                if cls_name == "cloth":
                    mask = results.masks.data[i]
                    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                    mask_total = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                    #frame = draw_segmentation(frame, results, model)
            frame_render = results.plot()

       
        ########## compute ribs
        frame_cloth = cv2.bitwise_and(frame, frame, mask=mask_total)
        frame_cloth = compute_ribs(frame_cloth)

        #### extract cloth
        show_oriented_cloth(mask_total, frame_cloth)

        map_2d = cv2.imread(current_action.texture_path)

        yaw = current_action.yaw
        tilt = current_action.tilt

        process_needle(needle_masks, frame_render)
        # ----------------------------------------------
        # Render cilindro
        # ----------------------------------------------
        # Render cilindro usando la acción actual
        if current_action.name == "sop10_1":
            schema = draw_SOP10_1(needle_theta=yaw, needle_visible=needle_visible)
        elif current_action.name == "sop30_1":
            schema = draw_SOP30_1(needle_theta=yaw, needle_visible=needle_visible)

        elif current_action.name == "sop30_2":
            schema = draw_SOP30_2(needle_x_norm=(yaw % (2*np.pi)) / (2*np.pi), needle_visible=needle_visible)

         #### compute proposed guide line 
        if x1 > 0:
            line_x = x2 + 10  # 10 pixels below detected framework
            y_start = y1 - 50
            y_end = y2 + 50
            draw_dashed_line(frame_render, (line_x, y_start), (line_x, y_end), (0, 255, 0), thickness=2, dash_len=15, gap=10)


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

        summary_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        summary_drawer.draw_text(   summary_frame, "TEST COMPLETADO", 50, 50, (0, 255, 0), scale=5    )
        
        wm.render("test_resize", summary_frame)
        # ----------------------------------------------
        # Controles
        # ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
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

        if key == ord('1'):
            current_action = action_mgr.set_action("sop30_1")
            yaw, tilt = current_action.yaw, current_action.tilt
            renderer.last_img = None

        elif key == ord('2'):
            current_action = action_mgr.set_action("sop30_2")
            yaw, tilt = current_action.yaw, current_action.tilt
            renderer.last_img = None

        elif key == ord('3'):
            current_action = action_mgr.set_action("sop10_1")
            yaw, tilt = current_action.yaw, current_action.tilt
            renderer.last_img = None
    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Parse command-line arguments for video processing.")
   # Define command-line arguments with optional flags
    parser.add_argument("--port", default="5102", help="Port for exposing")
    
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


if __name__ == "__main__":
    main_loop()