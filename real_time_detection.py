
import cv2
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
import surgery
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
PRETRAINED_MODEL_PATH = "/home/simon/logs/surgery_200/200_images_mask_rcnn_surgery.h5"

class InferenceConfig(surgery.SurgeryConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == '__main__':
     class_names = ['BG', 'arm', 'ring']
     #加载模型
     config = InferenceConfig()
     config.display()

     model = modellib.MaskRCNN(mode="inference", config=config, model_dir='/home/simon/logs/surgery_200')
     model_path = PRETRAINED_MODEL_PATH
     # or if you want to use the latest trained model, you can use :
     # model_path = model.find_last()[1]
     model.load_weights(model_path, by_name=True)
     colors = visualize.random_colors(len(class_names))

     cap = cv2.VideoCapture(0)
     while True:

         _, frame = cap.read()
         predictions = model.detect([frame],
                                    verbose=1)  # We are replicating the same image to fill up the batch_size
         p = predictions[0]

         output = visualize.display_instances(frame, p['rois'], p['masks'], p['class_ids'],
                                     class_names, p['scores'], colors=colors, real_time=True)
         cv2.imshow("Mask RCNN", output)
         k = cv2.waitKey(10)
         if k & 0xFF == ord('q'):
             break
     cap.release()
     cv2.destroyAllWindows()