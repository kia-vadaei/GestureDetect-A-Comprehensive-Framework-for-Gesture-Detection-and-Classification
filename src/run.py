from ultralytics import YOLO
import cv2
import argparse
import os
parser = argparse.ArgumentParser(description="IP Packet Sender/Receiver using Scapy")

def main():    
    parser.add_argument(
    '--model-path',
    type=str,
    help='Model File Path (.pt format)',
    required=True,)

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'Kaggle token file not found: {args.model_path}')

    model = YOLO(args.model_path)
    camera = cv2.VideoCapture(0)  

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        exit()

    while True:

        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break


        results = model(frame)


        annotated_frame = results[0].plot()


        cv2.imshow("YOLO Predictions", annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
