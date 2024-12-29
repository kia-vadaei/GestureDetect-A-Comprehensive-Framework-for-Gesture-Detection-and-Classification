import os
import cv2
import random
import mediapipe as mp
from tqdm import *

class ImageProcessor:
    def __init__(self, root_dir, image_output_dir, label_output_dir, val_image_output_dir, val_label_output_dir):
        self.root_dir = root_dir
        self.image_output_dir = image_output_dir
        self.label_output_dir = label_output_dir
        self.val_image_output_dir = val_image_output_dir
        self.val_label_output_dir = val_label_output_dir
        
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.label_output_dir, exist_ok=True)
        os.makedirs(self.val_image_output_dir, exist_ok=True)
        os.makedirs(self.val_label_output_dir, exist_ok=True)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    def get_index(self, class_name):
        classes = ['Horiz_HFR','XSign','Punch_VFR','Two_VFR','Eight_VFR','One_VFR','Span_VFR','Seven_VFR','Nine_VFR',
                'Five_VFR','Collab','DissLike','Like','TimeOut','Six_VFR','Four_VFR','Three_VFR']
        return classes.index(class_name) 

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            
                h, w, _ = image.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                # Normalize to image dimensions
                center_x = (x_min + x_max) / 2 / w
                center_y = (y_min + y_max) / 2 / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h

                return [center_x, center_y, width, height]
        return None


    def enhance_bbox_width(self, bbox, image_width, image_height, enhancement_factor=0.4):
        center_x, center_y, width, height = bbox

        current_width = width * image_width

        enhancement = current_width * enhancement_factor

        new_width = current_width + enhancement
        new_center_x = center_x  

        new_x_min = max(0, new_center_x * image_width - new_width / 2)
        new_x_max = min(image_width, new_center_x * image_width + new_width / 2)

        new_width_normalized = (new_x_max - new_x_min) / image_width
        new_center_x_normalized = (new_x_min + new_x_max) / 2 / image_width

        return [new_center_x_normalized, center_y, new_width_normalized, height]


    def process_and_save(self):
        train_image_id = 1
        val_image_id = 1
        print(f'train_image_id: {train_image_id}')
        print(f'val_image_id: {val_image_id}')
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for image_name in tqdm(os.listdir(class_dir)):
                
                image_path = os.path.join(class_dir, image_name)
                if not os.path.isfile(image_path):
                    continue

                bbox = self.process_image(image_path)
                if bbox:
                    image = cv2.imread(image_path)
                    image_height, image_width, _ = image.shape

                    if class_name.lower() == 'collab':
                        bbox = self.enhance_bbox_width(bbox, image_width, image_height)


                    if random.random() < 0.3:
                        # Validation set
                        new_image_name_val = f'image{val_image_id}.jpg'
                        new_image_path = os.path.join(self.val_image_output_dir, new_image_name_val)
                        label_file_path = os.path.join(self.val_label_output_dir, f'image{val_image_id}.txt')
                        val_image_id += 1
                    else:
                        # Training set
                        new_image_name_train = f'image{train_image_id}.jpg'
                        new_image_path = os.path.join(self.image_output_dir, new_image_name_train)
                        label_file_path = os.path.join(self.label_output_dir, f'image{train_image_id}.txt')
                        train_image_id += 1

                    cv2.imwrite(new_image_path, image)

                    # Save the label file
                    label = f'{self.get_index(class_name)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'
                    with open(label_file_path, 'w') as label_file:
                        label_file.write(label)
