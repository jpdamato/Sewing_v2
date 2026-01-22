import cv2
import numpy as np
import base64

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

    merged = np.zeros_like(masks[0], dtype=np.uint8)

    for i, cls in enumerate(classes):
        if cls == cloth_class_id:
            merged |= (masks[i] > 0.5).astype(np.uint8)

    # escalar al tamaño del frame
    merged = cv2.resize(
        merged,
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel)


    return merged
