# Sewing_v2

################################################################################
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/dalehitt/Desktop/Deploy_Inference_sewing2:/app/src \
  --device /dev/dri  \
  --gpus all --runtime=nvidia -p 8081:8081   \
  -v /home/dalehitt/Desktop/Deploy_Inference_sewing2/data/:/app/data   sewing:latest bash

### install

pip install torchvision==0.19
pip install -U torch torchvision
pip install glfw
pip install moderngl



###########################################################################
## start inference
python3 /app/src/Sewing_v2/main_process.py --src /app/data/Operation10_A.mp4 --model /app/src/Sewing_v2/edwards_insipiris_best_14jan.pt


## run in a command line
curl -X POST http://localhost:8081/main_start \
  -H "Content-Type: application/json" \
  -d '{
    "SOPCode": "SOP10",
    "WS": "WS1",
    "SessionID": 1
  }'

 


/home/dalehitt/Desktop/Deploy_Inference_sewing2/data

