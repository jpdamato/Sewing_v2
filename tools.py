import cv2
import numpy as np
import base64
import time
from collections import defaultdict
from scipy.spatial import cKDTree

import threading
# --- storage interno ---
_process_start = {}
_process_times = defaultdict(list)

material_colors = { "ribs" : (130,130,150),
                     "cloth" : (200,180,180),
                     "thread" : (220, 100,50),
                     "globes" : (50, 130,130),
                      "needle" : (220, 50,120),
                       "metal_framework" : (130,60,60),
                        "framework" : (130,60,60)  }

def dist(p1, p2):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))

def draw_center_line(img,  center, direction, length, color = (0,0,255) ):
    
    cX, cY = int(center[0]), int(center[1])
    vx, vy = direction

    # Draw line
    pt1 = (int(cX - vx * length/2), int(cY - vy * length/2))
    pt2 = (int(cX + vx * length/2), int(cY + vy * length/2))
    cv2.line(img, pt1, pt2, color, 2)
   # cv2.circle(img, (cX, cY), 4, (255, 0, 0), -1)

    return img


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


# ----------------------
def startProcess(process_id: str):
    """
    Marca el inicio de un proceso
    """
    _process_start[process_id] = time.perf_counter()


def endProcess(process_id: str):
    """
    Marca el fin de un proceso y guarda el tiempo
    """
    if process_id not in _process_start:
        return

    dt = time.perf_counter() - _process_start[process_id]
    _process_times[process_id].append(dt)

    del _process_start[process_id]


def is_straight_segment(    contour,
    max_mean_error=6.0,    max_max_error=12.0,
    min_aspect_ratio=0.50
):
    """
    Decide si un contorno corresponde a un segmento recto.

    Args:
        contour: np.ndarray (N,1,2)
        max_mean_error: error medio permitido (pixeles)
        max_max_error: error máximo permitido
        min_aspect_ratio: alargamiento mínimo

    Returns:
        True si es recto, False si es curvo
    """

    if contour is None or len(contour) < 10:
        return False

    pts = contour.reshape(-1, 2).astype(np.float32)

    # --- filtro 1: alargamiento ---
    x, y, w, h = cv2.boundingRect(pts)
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < min_aspect_ratio:
        return False

    # --- ajuste de recta (PCA) ---
    mean = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - mean)
    direction = Vt[0]  # dirección principal

    # --- distancia punto-recta ---
    diffs = pts - mean
    perp_dist = np.abs(
        diffs[:, 0] * direction[1] -
        diffs[:, 1] * direction[0]
    )

    mean_error = perp_dist.mean()
    max_error = perp_dist.max()

    return mean_error < max_mean_error and max_error < max_max_error

def printAverageTimes(min_samples=1):
    """
    Imprime los tiempos promedio de cada proceso
    """
    print("\n--- Average process times ---")
    for pid, times in _process_times.items():
        if len(times) < min_samples:
            continue

        avg = sum(times) / len(times)
        print(
            f"{pid:20s} | "
            f"avg: {avg*1000:7.2f} ms | "
            f"samples: {len(times)}"
        )
        
def serializeFrame(self, frame, resizeFactor = 1):
    scale = resizeFactor
    resizedFrame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) #Reduce frame size
    _,buffer = cv2.imencode(".jpg", resizedFrame)
    buffer_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')   #Serialized frame
    return buffer_b64

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
    if contorno_tela is None:
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
#########################################################
def contour_main_axis(contour):
    """
    Devuelve centro y dirección principal del contorno
    """
    pts = contour.reshape(-1, 2).astype(np.float32)

    mean = pts.mean(axis=0)
    pts_centered = pts - mean

    _, _, Vt = np.linalg.svd(pts_centered)
    direction = Vt[0]  # eje principal

    return mean, direction

def generate_perpendicular_curve(
    center,
    main_dir,
    perp_dir,
    length=1200,
    step=5,
    curvature=0.002
):
    """
    Genera puntos de una curva perpendicular suave
    """
    points = []

    for s in np.arange(-length, length, step):
        # desplazamiento principal para curvatura
        offset = curvature * (s ** 2)

        p = (
            center
            + s * perp_dir
            + offset * main_dir
        )
        points.append(p)

    return np.array(points, dtype=np.float32)

def clip_curve_to_contour(curve_pts, contour):
    inside = []

    for p in curve_pts:
        if cv2.pointPolygonTest(contour, tuple(p), False) >= 0:
            inside.append(p)

    if len(inside) < 2:
        return None

    return np.array(inside, dtype=np.int32)
##################################################
def is_contour_almost_circular(contour,
                               min_area=50,
                               circularity_thresh=0.75):
    """
    Retorna (is_circular, circularity)

    contour: (N,1,2)
    """

    area = cv2.contourArea(contour)
    if area < min_area:
        return False, 0.0

    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False, 0.0

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    return circularity >= circularity_thresh, circularity
###################################################
def contour_axes(contour):
    if contour is None:
        return
    
    pts = contour.reshape(-1, 2).astype(np.float32)
    center = pts.mean(axis=0)

    _, _, Vt = np.linalg.svd(pts - center)
    main_dir = Vt[0]
    perp_dir = np.array([-main_dir[1], main_dir[0]])  # rotado 90°

    return center, main_dir, perp_dir
#########################################################
def line_contour_intersections(center, direction, contour, length=2000, step=1.0):
    """
    Devuelve los dos puntos donde la línea corta el contorno
    """
    pts_inside = []

    for s in np.arange(-length, length, step):
        p = center + s * direction
        if cv2.pointPolygonTest(contour, tuple(p), False) >= 0:
            pts_inside.append(p)

    if len(pts_inside) < 2:
        return None, None

    return pts_inside[0], pts_inside[-1]

#########################################################
def _cross2(a, b):
    return a[0]*b[1] - a[1]*b[0]

def _dot2(a, b):
    return a[0]*b[0] + a[1]*b[1]

def segments_colinear_but_not_overlapped(p1, p2, q1, q2,
                                        eps_angle=1e-6,
                                        eps_dist=1e-6,
                                        eps_overlap=1e-6):
    """
    Retorna True si:
      - los segmentos (p1,p2) y (q1,q2) son colineales
      - y NO están superpuestos (overlap) en la misma recta

    Retorna False si:
      - no son colineales
      - o son colineales pero se solapan (superpuestos)
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)

    dp = p2 - p1
    dq = q2 - q1

    lp = np.linalg.norm(dp)
    lq = np.linalg.norm(dq)
    if lp < eps_dist or lq < eps_dist:
        return False  # segmento degenerado

    # 1) Paralelos (cross ~ 0)
    if abs(_cross2(dp, dq)) > eps_angle * (lp * lq):
        return False

    # 2) En la misma recta (q1 - p1) también paralelo a dp
    if abs(_cross2(q1 - p1, dp)) > eps_dist * lp:
        return False

    # 3) Chequeo de solapamiento:
    # Proyectamos los extremos de Q sobre el eje de P y vemos intervalos
    u = dp / lp  # dirección unitaria sobre la recta
    p_min, p_max = 0.0, lp
    t1 = _dot2(q1 - p1, u)
    t2 = _dot2(q2 - p1, u)
    q_min, q_max = (min(t1, t2), max(t1, t2))

    overlap = min(p_max, q_max) - max(p_min, q_min)

    # Si overlap > 0 => se superponen (aunque sea parcialmente)
    if overlap > eps_overlap:
        return False  # colineales pero superpuestos => DESCARTAR

    return True  # colineales y NO superpuestos

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

def merge_segments(needles):
    pts = []

    for needle in needles:
        rect = needle.segment["rect"]
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


def box_intersection_area(boxA, boxB):
    """
    boxA, boxB: np.array([x1, y1, x2, y2])
    devuelve: area_interseccion (float)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    w = max(0.0, xB - xA)
    h = max(0.0, yB - yA)

    return w * h

def process_needle(needle_s, img, render = False):
    needle_segments = []
    split_needle = []

    try:
       

        for needle in needle_s:
            if needle.get_length() > 40:
                needle_segments.append(needle)

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
                if are_colinear(s1.segment, s2.segment) and box_intersection_area(s1.box, s2.box)<50:
                    group.append(s2)
                    used.add(j)

            groups.append(group)

        
        for group in groups:
            if len(group) >= 2:
                p1, p2 = merge_segments(group)
                split_needle.append([p1,p2, group])
                if render :
                    cv2.line(img, p1, p2, (0,255,0), 1)

                    for s in group:
                        p1,p2 = s.get_center_line()
                        cv2.line(img, p1, p2, (0,200,110), 2)
    except  :
        print ("Error at computing segments")

    return split_needle

###################################################################
      
#################################################
###  Compute ribs
def render_ribs(frame, ribs, color):
    for hole in ribs:
        cx, cy = hole
        cv2.circle(frame, (int(cx), int(cy)), 2, color, -1)


def compute_ribs( frame, mask_cloth,do_render_ribs=False, clahe=None):
    startProcess("compute_ribs")
    frame_cloth = cv2.bitwise_and(frame, frame, mask=mask_cloth)
    
    gray = cv2.cvtColor(frame_cloth, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste local
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    gray = clahe.apply(gray)

    _, bw = cv2.threshold(  gray, 0, 255,   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cv2.imshow("Binarized", bw)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    holes = []
    min_area = 20
    max_area = 300

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            holes.append([centroids[i][0], centroids[i][1]])
    
    #vis = frame.copy()

    if do_render_ribs:
        render_ribs(frame_cloth, holes)
   # cv2.imshow("Huecos en tela segmentada", vis)
    endProcess("compute_ribs")
    return  holes,frame_cloth
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

class CLAHEThread:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._running = True
        self._new_frame = False
        self.ribs = []
        self.px_to_cm = 0
        self.cm_to_px = 0
        self.dist_px_mean = 0

        self._frame = None
        self._result = None
        self.mask_cloth = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, frame_bgr, mask_cloth):
        """Recibe una imagen BGR uint8 (H,W,3)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        with self._cv:
            self._frame = frame_bgr.copy()
            self.mask_cloth = mask_cloth.copy()
            self._new_frame = True
            self._cv.notify()
        return True

    def get(self):
        """Devuelve el último resultado (o None si todavía no hay)."""
        with self._lock:
            return None if self._result is None else self._result.copy()
    def draw(self, frame, color):
        render_ribs(frame, self.ribs, color)

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify()

        self._thread.join(timeout=2.0)

    def _loop(self):
        while True:
            with self._cv:
                while self._running and not self._new_frame:
                    self._cv.wait()

                if not self._running:
                    break

                frame = self._frame
                self._new_frame = False

            # Procesar fuera del lock
            if frame is not None and self.mask_cloth is not None:
                out = self._apply_clahe(frame,self.mask_cloth)

                with self._lock:
                    self._result = out

    def _apply_clahe(self, frame_bgr,mask_cloth):
        try:
          
            self.ribs , _  = compute_ribs(frame_bgr, mask_cloth, False, self._clahe)
            self.px_to_cm, self.cm_to_px, self.dist_px_mean = estimate_pixel_to_cm(self.ribs, real_dist_mm=1.0, k=4)
        except Exception as e:
            print(f"Exception computing ribs: {e}")
        return 
  


##########################################
def extract_cloth_contour_and_bbox(mask, return_convex_hull=False):
    """
    A partir de una máscara binaria:
    - devuelve el contorno externo principal
    - devuelve su bounding box (xyxy)

    Returns:
        contour: np.ndarray (N,1,2)
        bbox: (x1, y1, x2, y2)
        o (None, None) si no hay detección
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

      # unir todos los puntos de contornos
    all_pts = np.vstack(contours)

    # convex hull global
    if return_convex_hull:
        hull = cv2.convexHull(all_pts)

        x, y, w, h = cv2.boundingRect(hull)
        bbox = (x, y, x + w, y + h)

        return hull, bbox
    else:
        mx_cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(mx_cnt)
        bbox = (x, y, x + w, y + h)

        return mx_cnt, bbox
        
def merge_cloth_masks(result, cloth_class_id, frame_shape):
    """
    Une todas las instancias de 'tela' en una sola máscara binaria
    """
    if result.masks is None:
        return None

    masks = result.masks.data.cpu().numpy()  # (N, Hm, Wm)
    classes = result.boxes.cls.cpu().numpy().astype(int)

    if len(masks) > 0:
        merged = np.zeros_like(masks[0], dtype=np.uint8)
    else:
        return None
    
    for i, cls in enumerate(classes):
        if cls == cloth_class_id:
            merged |= (masks[i] > 0.5).astype(np.uint8)

    # escalar al tamaño del frame
    merged = cv2.resize(
        merged,
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_NEAREST    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel)


    return merged
