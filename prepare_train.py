import cv2
import os
from datetime import datetime


from ultralytics import YOLO

#model = YOLO("D:/Proyects/Novathena/INSIPIRIS/Sewing_v2/edwards_insipiris_best_14jan.pt")
#model.export(format="onnx", opset=12, dynamic=True)


# ---------------- CONFIG ----------------
video_path = "E:/Resources/Novathena/INSIPIRIS/record_stop.mp4"          # <-- tu video
output_dir = "E:/Resources/Novathena/Train/video4"         # <-- carpeta donde guardar
resize_w, resize_h = 1920, 1080   # tamaño final
step_frames = 10
# ---------------------------------------
# ---------------------------------------

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

saved_count = 0

print("Controles:")
print("  s = guardar frame (resize 1920x1080)")
print("  m = avanzar 10 frames")
print("  n = retroceder 10 frames")
print("  q = salir")

while True:
    # Leer frame actual
    ret, frame = cap.read()
    if not ret:
        print("Fin del video o error leyendo frame.")
        break

    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # posición actual (después del read)

    frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    cv2.putText(frame,f"{pos}", (50,50),1, 0.50, (255,0,0), 1)
    # Mostrar
    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF  # waitKey(0) para que espere tecla (más cómodo para navegar)

    if key == ord("q"):
        break

    elif key == ord("s"):

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{ts}_pos{pos}.jpg"
        out_path = os.path.join(output_dir, filename)

        cv2.imwrite(out_path, frame)
        saved_count += 1
        print(f"[OK] Guardado: {out_path} (total: {saved_count})")

    elif key == ord("m"):
        new_pos = min(pos + step_frames, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

    elif key == ord("n"):
        new_pos = max(pos - step_frames, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

cap.release()
cv2.destroyAllWindows()