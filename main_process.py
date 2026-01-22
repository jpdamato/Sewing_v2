import cv2
import numpy as np
from ultralytics import YOLO
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
import json
from dataclasses import dataclass
import argparse
import json
import base64
import glfw
import math
import os
from frame_renderer.window_manager import WindowManager
from frame_renderer.drawer import Drawer
from frame_renderer.fonts import Font
from inference_core import SOP_Manager, TrackerManager , get_data_for_unity_sop10
import tools as tools
import helpers as helpers
import time

############################
### Front End interaction
from confluent_kafka import Producer
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import threading
from datetime import datetime
import platform
from confluent_kafka import Producer
# ==================================================
# Configuración
# ==================================================
VERSION = "Sewing V2.0 - 21st jan2025"
IMAGE_2D_PATH = "SOP30-Lateral_B.png"
MODEL_PATH = "edwards_insipiris_best_14jan.pt"

RESIZE_CAM_WIDTH = 1280
RESIZE_CAM_HEIGHT = 720
MAXIMIZED = False
ENABLED_CAM = False

GLOBAL_KEY_BOARD =   0
RUNNING_APP = True
DEBUG = False

CLASE_HILO = 6
CLASE_TELA = 0

ZOOM = 1.0
TX = 0.0
TY = 0.0
kafka_data = {}
kafka_instance = None
sop_Manager = None

########################################################
###  Class for handling real time streaming usin REST methods
########################################################
app = Flask(__name__)
CORS(app)  # allow all origins

@app.route("/overlay", methods=["POST"])
def overlay_zoom():
    global ZOOM , TX, TY , WORKSTATION_ID
     # Check if the incoming request is JSON
    if request.is_json:
        parameters = request.json
        WORKSTATION_ID = parameters.get("workstation_id", 0)
        ZOOM = parameters.get("zoom_factor", 1.0) ## range 0.5 to 2.0   
        TX = parameters.get("offset_x", 0) ### lateral offset X
        TY = parameters.get("offset_y", 0) ### lateral offset Y

        return jsonify({"status": "overlay zoomed"})
    else:
        return "Content-Type must be application/json", 400

@app.route("/main_alive", methods=["POST"])
def main_alive():
    global RUNNING_APP
    response = Response(
        response=[f"Video is playing? {RUNNING_APP}"],
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/main_stop", methods=["POST"])
def main_stop():
    global RUNNING_APP
    print("Received stop")
    response = Response(
        response=[],
        status=200,
        mimetype='application/json'
    )
    RUNNING_APP = False
    return response

@app.route("/main_start", methods=["POST"])
def main_start():
    global RUNNING_APP, WS_ID , SOPCode , SessionID
    print("Received start")
    
    data = request.json
    WS_ID = data.get("WS", "SOP10")
    SOPCode = data.get("SOPCode", "SOP10")
    SessionID = data.get("SessionID", "SOP10")

    response = Response(
        response=[],
        status=200,
        mimetype='application/json'
    )
    RUNNING_APP = True
    return response

#########################################################
### Kafka Server
class Kafka:
    def __init__(self, KafkaServerURL = '172.31.1.7:9094', KafkaPartition = 0):
        # self.Kafka_bootstrap_servers = os.environ.get('BOOTSTRAP_SERVERS', '172.31.1.7:9092')
        # self.Kafka_bootstrap_servers = '172.31.1.7:9094'
        # self.Kafka_topic = os.environ.get('TOPIC', 'nova')
        # self.Kafka_partition = int(os.environ.get('PARTITION', '0'))
        self.Kafka_bootstrap_servers = KafkaServerURL   #IP:Port
        self.Kafka_partition = KafkaPartition
        kafka_conf = {
            'bootstrap.servers': self.Kafka_bootstrap_servers,
        }
        self.Kafka_producer = Producer(**kafka_conf)  
        # producer.flush()
        # producer.close()


    def delivery_report(self, err, msg):
        """ Called once for each message produced to indicate delivery result.
            Triggered by poll() or flush(). """
        if err is not None:
            print('KAFKA Message delivery failed: {}'.format(err))
        else:
            print('KAFKA Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

        

    
    def produce_kafka_msg(self, kafka_jsonANDframe_str, topic):    
        try:
            self.Kafka_producer.produce(topic, kafka_jsonANDframe_str, partition=self.Kafka_partition, callback=self.delivery_report)
            
            self.Kafka_producer.poll(0)
            print(f"Kafka msg delivered: len = {len(kafka_jsonANDframe_str)} >>>>>>>>>")
            # time.sleep(0.01)  # Optional: Adjust the sleep time between each message
        except KeyboardInterrupt:
            pass

    def serializeFrame(self, frame, resizeFactor = 1):
        scale = resizeFactor
        resizedFrame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) #Reduce frame size
        _,buffer = cv2.imencode(".jpg", resizedFrame)
        buffer_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')   #Serialized frame
        return buffer_b64


#########################
## run kafka in other thread
def run_kafka_server():
    global  RUNNING_APP, kafka_data , kafka_instance, WS_ID
    try:
        kafka_instance = Kafka()
        print ("Kafka server is RUNNING ")
    except:
        print ("Kafka server is not available ")
        kafka_instance = None
    scale = 0.5

    while True:
        if "frame" in kafka_data and kafka_data["frame"] is not None and RUNNING_APP:
            try:
                # Get the current time as a float (seconds since the epoch)
                current_timestamp = time.time()
                # Convert the timestamp to a struct_time object for local time
                local_time_struct = time.localtime(current_timestamp)
                # Format the struct_time object into a string
                # Example format: YYYY-MM-DD HH:MM:SS
                formatted_time_string = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)
              
                msgKafka = { 
                            "timeStamp": formatted_time_string,
                            "Data": kafka_data["data"],
                        }
                
                msgKafkaStr = json.dumps(msgKafka)

                #Send message to Kafka
                if kafka_instance is not None  :
                    kafka_instance.produce_kafka_msg(msgKafkaStr, topic="ws"+str(WS_ID))
                # sleep for 500 ms
                time.sleep(1)
            except Exception as e:
                print (f"Kafka exception : {e}")
        else:
            time.sleep(1)

    pass
###################
## debug port = 8031
def start_API(port):
    print("Starting API service")
    app.run(host="0.0.0.0", port=int(port), debug=False)


def on_key(window, key, scancode, action, mods):
    global RUNNING_APP, GLOBAL_KEY_BOARD, ENABLED_CAM
    if action == glfw.PRESS:
        print(f"Tecla presionada: {key}")
    elif action == glfw.RELEASE:
        print(f"Tecla liberada: {key}")

    if key == glfw.KEY_Q and action == glfw.PRESS:
        ENABLED_CAM = not ENABLED_CAM
    
    if key == glfw.KEY_T and action == glfw.PRESS:
        tools.printAverageTimes()

    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        print("Escape pressed, closing window.")
        RUNNING_APP = False

def load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT, monitor_id=0):
    global MAXIMIZED
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
        RESIZE_CAM_HEIGHT,
        RESIZE_CAM_WIDTH,
        "test_resize",
        position=(monitor_id*WINDOW_WIDTH, 100),
        maximized=MAXIMIZED,
        decorators=False,
    )

    glfw.make_context_current(window.glfw_window)
    glfw.set_key_callback( window.glfw_window , on_key)

    return wm

###################################################
def run_glfw_loop( monitor_id=0):
    global RUNNING_APP ,ENABLED_CAM , sop_Manager
    
    WINDOW_HEIGHT=1920
    WINDOW_WIDTH = 1080
    
    print ("Loading GLFW window manager")
    wm=load_window_manager(WINDOW_WIDTH, WINDOW_HEIGHT, monitor_id=monitor_id)
    font = Font.get_font()
    summary_drawer = Drawer(font, WINDOW_HEIGHT, WINDOW_WIDTH)
    
    while RUNNING_APP:
        try:
            ###  Draw using lib
            summary_frame = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3), dtype=np.uint8)

            if sop_Manager is not None:
                if ENABLED_CAM:
                    summary_frame = cv2.resize(sop_Manager.original_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                for det in sop_Manager.detections:
                    if det.name == "cloth":
                        det.draw_contours(summary_frame, (255, 255, 255),width = 2)
                    elif det.name == "thread":
                        det.draw(summary_frame, (220, 55, 55),width = 2)
            summary_frame = cv2.flip(summary_frame, 0)
            summary_frame = cv2.cvtColor(summary_frame, cv2.COLOR_BGR2RGB)
            
            if sop_Manager is not None:
                if sop_Manager.tracking <= 0:
                    summary_drawer.draw_text(summary_frame, "TEST RESULTS", 50, 50, (0, 255, 0), scale=3)
                else:
                    summary_drawer.draw_text(summary_frame, "WORKING", 50, 50, (0, 255, 0), scale=3)
                summary_drawer.draw_text(summary_frame, f"Threads {len(sop_Manager.selected_stitches)}", 50, 80, (0, 255, 0), scale=2)
            
            else:
                summary_drawer.draw_text(summary_frame, "WAITING FOR START", 50, 50, (0, 255, 0), scale=3)
            wm.render("test_resize", summary_frame)
            
        except Exception as e:
            print(f"Exception terminating GLFW: {e}")
            pass

    print ("Terminate rendering")
    glfw.terminate()
    
def translate_and_zoom(    image,    zoom=1.0,    dx=0,    dy=0,
    border_value=(0, 0, 0) ):
    """
    Aplica zoom centrado y traslación a una imagen.

    Args:
        image: np.ndarray (H,W,C)
        zoom: float (>1 zoom in, <1 zoom out)
        dx: int (pixeles en X)
        dy: int (pixeles en Y)
        border_value: color para huecos (default negro)

    Returns:
        transformed image
    """

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    # Matriz de zoom centrado
    M_zoom = cv2.getRotationMatrix2D((cx, cy), 0, zoom)

    # Agregar traslación
    M_zoom[0, 2] += dx
    M_zoom[1, 2] += dy

    out = cv2.warpAffine(        image,
        M_zoom,        (w, h),        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return out

def main_loop(video_path, monitor_id = 0, start_frame=18000, has_rectangle=False, working_path="./data/"):
    global RUNNING_APP, GLOBAL_KEY_BOARD, MAXIMIZED, ENABLED_CAM, kafka_data
    global ZOOM , TX, TY , WORKSTATION_ID, sop_Manager
    # ==================================================
    # Loop principal
    # ==================================================   
    sop_Manager = SOP_Manager(model_path = MODEL_PATH)
    action_mgr = sop_Manager.action_mgr

    tracker_mgr = TrackerManager(    hilo_class_id = 6,    min_samples=10)
    # Acción inicial
    current_action = action_mgr.set_action("sop10_1")
    yaw = current_action.yaw
    tilt=current_action.tilt
    framework_yaw = 0.0
    #########################
    SOP = {"name": 0, "index":0, "step_order":0}
 
    review_mode = True
    paused = False
    last_frame = None
    all_data = []

    os.makedirs(working_path, exist_ok=True)

    if video_path == "0":
        video_path = 0  # Use camera

    cap = cv2.VideoCapture(video_path)
  
    ## sop30 = first frame = 10000
    if (video_path == "/dev/video0" or video_path == "0"):
        print("using real time camera")
        step_frame = 1
    else:   
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        step_frame = 2
    
    # ==================================================
    # Cargas
    # ==================================================
    if not cap.isOpened():
        print(f"Video file not found: {video_path}")
        return

    prev_time = time.perf_counter()

    while cap.isOpened() and RUNNING_APP:

        if paused:
            frame = last_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                break
            
        sop_Manager.original_frame = frame.copy()
        last_frame = frame.copy()

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.resize(frame, (RESIZE_CAM_WIDTH, RESIZE_CAM_HEIGHT))

        if frame_idx % step_frame != 0:
            continue

        fh, fw = frame.shape[:2]
        ### videos have a rectangle with data, i want to cover it
        if has_rectangle:
            cv2.rectangle(frame, (870, 500), (1280, 1050), (0, 0, 0), -1)  # background for text

        if abs(TX) > 0 or abs(TY)>0 or ZOOM != 1.0:
            frame = translate_and_zoom(frame, zoom=ZOOM, dx=TX, dy=TY)

        ### estimate SOP
        tools.startProcess("inference")
        SOP = sop_Manager.estimate_SOP(frame, frame_idx, None)

        ### run detections
        frame_render, map_2d, schema, detections = sop_Manager.run_frame(frame,frame_idx,tracker_mgr, review_mode)
        tools.endProcess("inference")

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

        # ---- timing ----
        curr_time = time.perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time

        fps = 1.0 / dt if dt > 0 else 0.0

        # ---- texto ----
        text_fps = f"FPS: {fps:.2f}"
        text_dt = f"Infer time: {dt*1000:.1f} ms"

        cv2.putText(            combined, text_fps,            (10, 30),            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,            (0, 255, 0),            2,            cv2.LINE_AA        )

        cv2.putText(            combined, text_dt,            (10, 60),            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,            (0, 255, 255),            2,            cv2.LINE_AA        )

        if not MAXIMIZED:
            cv2.imshow("Frame | Imagen 2D + Cilindro 3D", combined)

        # ----------------------------------------------
        # Controles
        # ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            RUNNING_APP = False
            break
        elif key == ord('t'):  # +30 frames
            tools.printAverageTimes()
        
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
            for tv in sop_Manager.valid_tracks:
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


        ######## 
        ## export data to Kafka
        Data = {}

        try:
            
            #### 
            Data["id"] = frame_idx                    
            sFrame = tools.serializeFrame(None, frame, resizeFactor=0.1)                      
            Data["frame"] = sFrame
            Data["sop"] = SOP["name"] ## SOP 10 
            Data["step_number"] = SOP["index"]
            Data["step_order"] = SOP["step_order"] ## 16 or 17

            unity_points , unity_points_ids = get_data_for_unity_sop10(sop_Manager, frame)
            ## bug still hardcoded points
            Data["unity_points_coords"] =unity_points #json.dumps( [] if len(unity_points) == 0 else unity_points  )
            Data["unity_points_ids"] = unity_points_ids
            Data["grading_table"] = []

            ### this should be thread safe
            kafka_data["frame"] = frame_render
            kafka_data["data"] = Data

            ### 
            if DEBUG:
                all_data.append(Data)

                # guardar al final del loop
                cv2.imwrite(f"{working_path}/output_frame{frame_idx}.jpg", frame_render)
                with open(f"{working_path}/output_stitches.json", "w") as f:
                    json.dump(all_data, f, indent=4)


        except Exception as e:
            print (f"Exception at exporting : {e}")

        
        #return SOP,Data
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
    parser.add_argument("--debug", default="", help="Run debug")

    parser.add_argument("--monitor_id", default=0, help="Monitor ID")
    parser.add_argument("--start_frame", default=100, help="Enable testing")
    parser.add_argument("--maximized", default=False, help="Enable testing")
    parser.add_argument("--ws_id", default=0, help="workstation id")
   
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

    print("-------------------------------------")
    print(VERSION)
    print("-------------------------------------")
    
    args = parse_args()
    print(args)
    ### start API

    t = threading.Thread(target=start_API, args=(int(args.port),))
    t.daemon = True
    t.start()

    ### start Kafka
    start_kafka_thread = threading.Thread(target=run_kafka_server)
    start_kafka_thread.daemon = True
    start_kafka_thread.start()

    ### run GLFW in another thread
    render_thread = threading.Thread(target=run_glfw_loop, args=(int(args.monitor_id),))
    render_thread.daemon = True
    render_thread.start()
  
    MAXIMIZED = args.maximized
    WS_ID = args.ws_id

    if args.debug in args:
        DEBUG = True

    if args.device != "":
        main_loop(args.device, args.monitor_id, args.start_frame)
    else:
        mp4File = get_mp4_from_path(args.src)

        if mp4File is None:
            print("Failed to open file")
            exit()

        main_loop(mp4File, args.monitor_id, args.start_frame)