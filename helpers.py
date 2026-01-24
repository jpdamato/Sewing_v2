import cv2
import numpy as np
import tools as tools

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

def render_perpendicular_curved_guideline(
    image,
    contour,offset = 0,
    color=(0, 255, 255),
    thickness=2,
    curvature=0.002
):
    center, main_dir, perp_dir = tools.contour_axes(contour)

    curve = tools.generate_perpendicular_curve(
        center,
        main_dir,
        perp_dir,
        curvature=curvature
    )

    clipped = tools.clip_curve_to_contour(curve, contour)

    if clipped is None:
        return

    if offset != 0:
        clipped = offset_curve(    clipped,    main_dir, offset_px=offset )

    draw_dotted_curve(image, clipped, color)
    #cv2.polylines(        image,        [clipped],        isClosed=False,        color=color,
    #    thickness=thickness,        lineType=cv2.LINE_AA    )
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
    if not is_maximized:
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

