from ultralytics import YOLO
import wandb
import argparse

def main():
    parser = argparse.ArgumentParser(description="YOLO Hand Gesture Detection")
    parser.add_argument(
    '--model-path',
    type=str,
    help='Model File Path (.pt format)',
    required=True,)
    args = parser.parse_args()

    parser.add_argument(
    '--final-model-path',
    type=str,
    help='Final Model File Path (.pt format)',
    required=True,)
    args = parser.parse_args()

    parser.add_argument(
    '--model-path',
    type=str,
    help='Model File Path (.pt format)',
    required=True,)
    args = parser.parse_args()

    parser.add_argument(
    '--config',
    type=str,
    help='Model config (.yaml) file',
    required=True,)

    parser.add_argument(
    '--project',
    type=str,
    help='Project Name',
    required=False,)

    parser.add_argument(
    '--name',
    type=str,
    help='Model Log name',
    required=True,)

    args = parser.parse_args()

    model = YOLO(args.model_path)
    model_path = args.final_model_path

    results = model.train(
        data= args.config,
        epochs=30,
        imgsz=640,
        project= args.project,  
        name= args.name     
    )


    model.save(model_path)


    wandb.init(args.project)
    wandb.log_model(model_path)
    wandb.finish()

if __name__ == '__main__':
    main()
