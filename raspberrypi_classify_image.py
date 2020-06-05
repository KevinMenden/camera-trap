#!/user/bin/python3

import os
import glob
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

# Paths
pic_dir = "/home/pi/cam_pics"
out_dir = "/home/pi/cam_pics/animals"
out_dir_empty = "/home/pi/cam_pics/empty"
model_path = "/home/pi/gitRepos/camera-trap/mobilenetv2_tflite"

# set up interpreter
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

# list pictures captured by motion
#pics = glob.glob(os.path.join(pic_dir, "*.jpg"))

pics = ["/home/pi/cam_pics/03-20200604211452-27.jpg"]
for pic in pics:
    contains_animal = True
    bn = os.path.basename(pic)
    # Check if picture contains animal
    image = Image.open(pic).convert('RGB').resize((width, height),Image.ANTIALIAS)
    result = classify_image(interpreter, image)[0][0]
    print(result)
    if result == 2:
      contains_animal = False  
    
    
    if contains_animal:
        cmd = f"mv {pic} {os.path.join(out_dir, bn)}"
    else:
        cmd = f"mv {pic} {os.path.join(out_dir_empty, bn)}"
    
    os.system(cmd)
