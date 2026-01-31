import cv2
import numpy as np
import tools as tools
import platform
import math

##########################
## Base class for visual helpers
class VisualHelper:
    def __init__(self):
        pass

    def update(self):
        pass

    def draw(self, frame, detections):
        pass
###############################################
class VisualGuideline:
    def __init__(self, color=(0,255,255),offset=0, thickness=2, curvature=0.02 ):
        self.offset = offset
        self.color=color
        self.thickness=thickness
        self.curvature=curvature
        self.contour = None
        self.metal_framework = None
        self.clipped= None

    def update(self, cloth, metal_framework):
        self.contour =cloth.contour
        self.metal_framework = metal_framework
        ### Juan BUG
        if self.contour is None:
            return
      
        self.center, self.main_dir, self.perp_dir = tools.contour_axes(self.contour)

        curve = tools.generate_perpendicular_curve(
            self.center,
            self.main_dir,
            self.perp_dir,
            curvature=self.curvature
        )

        self.clipped = tools.clip_curve_to_contour(curve,self.contour)

    def draw(self,   image):        
        if self.clipped is None:
            return

        if self.offset != 0:
            self.clipped = offset_curve(    self.clipped,    self.main_dir, offset_px=self.offset )

        draw_dotted_curve(image, self.clipped, self.color)
    #cv2.polylines(        image,        [clipped],        isClosed=False,        color=color,
    #    thickness=thickness,        lineType=cv2.LINE_AA    )

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

def draw_dotted_curve(img, points, color=(0, 255, 255),
                      thickness=2, dash_len=8, gap_len=6):
    """
    Dibuja una curva punteada a partir de una lista de puntos (Nx2)
    """
    draw = True
    acc_len = 0.0

    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]

        seg_len = np.linalg.norm(p1 - p0)
        acc_len += seg_len

        if draw:
            cv2.line(
                img,
                tuple(p0.astype(int)),
                tuple(p1.astype(int)),
                color,
                thickness,
                cv2.LINE_AA
            )

        if draw and acc_len >= dash_len:
            draw = False
            acc_len = 0.0
        elif not draw and acc_len >= gap_len:
            draw = True
            acc_len = 0.0

def offset_curve(curve_pts, direction, offset_px):
    """
    Desplaza una curva en la dirección dada

    curve_pts: (N,2)
    direction: vector unitario (2,)
    offset_px: pixeles (+ o -)
    """
    direction = direction / np.linalg.norm(direction)
    return curve_pts + offset_px * direction

def render_event_grid(event_frames, cols=4, scale=0.5, pad=5, bg_color=(0, 0, 0)):
    """
    images: list[np.ndarray] (BGR)
    cols: cantidad de columnas
    scale: escala de resize (ej 0.5)
    pad: padding en px entre celdas
    bg_color: color de fondo (B,G,R)

    return: np.ndarray collage
    """
    if event_frames is None or len(event_frames) == 0:
        return None

    cols = max(1, int(cols))
    scale = float(scale)

    
    # Redimensionar todas según scale
    resized = []
    for ev in event_frames:
        if ev.frame is None:
            continue

        h, w = ev.frame.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized.append(cv2.resize(ev.frame, (new_w, new_h), interpolation=cv2.INTER_AREA))

    # Usamos el tamaño máximo para cada celda (para alinear bien)
    cell_w = max(im.shape[1] for im in resized)
    cell_h = max(im.shape[0] for im in resized)

    n = len(resized)
    rows = math.ceil(n / cols)

    grid_w = cols * cell_w + (cols - 1) * pad
    grid_h = rows * cell_h + (rows - 1) * pad

    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    for idx, im in enumerate(resized):
        r = idx // cols
        c = idx % cols

        x0 = c * (cell_w + pad)
        y0 = r * (cell_h + pad)

        # centrado dentro de la celda
        h, w = im.shape[:2]
        x = x0 + (cell_w - w) // 2
        y = y0 + (cell_h - h) // 2

        grid[y:y+h, x:x+w] = im

    return grid

def render_event_strip(event_frames,
                       scale=0.3,
                       max_height=None,
                       bg_color=(0, 0, 0)):
    """
    event_frames: lista de imágenes (BGR)
    scale: factor de escala (ej: 0.3)
    max_height: fuerza una altura común (opcional)
    """

    resized = []

    for event in event_frames:
        if event.frame is None:
            continue

        h, w = event.frame.shape[:2]

        if max_height is not None:
            scale = max_height / h

        new_size = (int(w * scale), int(h * scale))
        small = cv2.resize(event.frame, new_size, interpolation=cv2.INTER_AREA)
        resized.append(small)

    if not resized:
        return None

    # --- igualar alturas (por seguridad)
    max_h = max(img.shape[0] for img in resized)

    padded = []
    for img in resized:
        h, w = img.shape[:2]
        if h < max_h:
            pad = max_h - h
            img = cv2.copyMakeBorder(
                img, 0, pad, 0, 0,
                cv2.BORDER_CONSTANT,
                value=bg_color
            )
        padded.append(img)

    return cv2.hconcat(padded)


#############################################################
def draw_dashed_ellipse(
    img,    ellipse,
    color=(0, 255, 0),    thickness=2,    dash_length=10,
    gap_length=6,    samples=200):
    """
    Dibuja una elipse punteada (dashed).

    ellipse: ((cx, cy), (MA, ma), angle)
    """

    (cx, cy), (MA, ma), angle = ellipse

    # muestrear elipse
    t = np.linspace(0, 2*np.pi, samples)

    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))

    pts = []
    for ti in t:
        x = (MA / 2) * np.cos(ti)
        y = (ma / 2) * np.sin(ti)

        xr = x * cos_a - y * sin_a + cx
        yr = x * sin_a + y * cos_a + cy

        pts.append((int(xr), int(yr)))

    # dibujar segmentos alternados
    i = 0
    while i < len(pts) - 1:
        j = min(i + dash_length, len(pts) - 1)
        cv2.line(img, pts[i], pts[j], color, thickness, cv2.LINE_AA)
        i += dash_length + gap_length

def show_oriented_cloth(mask, img, is_maximized=False):
   
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
    
    if platform.system() == "Windows":
        cv2.imshow("Oriented Cloth", vis)


def fit_robust_ellipse(points, keep_ratio=0.9):
    """
    Ajusta una elipse usando el % central de los puntos
    """
    if len(points) < 20:
        return None

    pts = points.reshape(-1, 1, 2).astype(np.float32)

    ellipse = cv2.fitEllipse(pts)
    (cx, cy), (MA, ma), angle = ellipse

    # medir distancia normalizada a la elipse
    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a

    d = (x_rot / (MA / 2))**2 + (y_rot / (ma / 2))**2

    # quedarnos con los más cercanos
    thresh = np.quantile(d, keep_ratio)
    inliers = points[d <= thresh]

    if len(inliers) < 20:
        return ellipse

    return cv2.fitEllipse(inliers.reshape(-1, 1, 2))

def ellipse_inside_contour(ellipse, contour, samples=72):
    (cx, cy), (MA, ma), angle = ellipse

    theta = np.linspace(0, 2*np.pi, samples)
    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))

    for t in theta:
        x = (MA/2) * np.cos(t)
        y = (ma/2) * np.sin(t)

        xr = x * cos_a - y * sin_a + cx
        yr = x * sin_a + y * cos_a + cy

        if cv2.pointPolygonTest(contour, (xr, yr), False) < 0:
            return False

    return True

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

###########################################################################################
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



def draw_helper_SOP10_Task16(frame_render, detections , sop_maanager):
    x1, y1, x2, y2 = 0, 0, 0, 0
    frame_work_found = False
    threads = []
    cloth_visible = False
    ### try to render a guide line below metal framework
    for det in detections:
        if det.name == "thread":
            threads.append(det)
            break
        if det.name == "metal_framework":
            x1, y1, x2, y2 = det.box
            frame_work_found = True
            break
        if det.name == "cloth":
            x1, y1, x2, y2 = det.box

            break

    ### render a guideline below metal framework
    if frame_work_found:
        line_x = x2 + 10  # 10 pixels below detected framework
        y_start = y1 - 50
        y_end = y2 + 50
        draw_dashed_line(frame_render, (line_x, y_start), (line_x, y_end), (0, 255, 0), thickness=2, dash_len=15, gap=10)
    else:
        ## draw a line according to RIBs
        line_x = x2 + 10  # 10 pixels below detected framework
        y_start = y1 - 50
        y_end = y2 + 50
        draw_dashed_line(frame_render, (line_x, y_start), (line_x, y_end), (0, 255, 0), thickness=2, dash_len=15, gap=10)
        

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

