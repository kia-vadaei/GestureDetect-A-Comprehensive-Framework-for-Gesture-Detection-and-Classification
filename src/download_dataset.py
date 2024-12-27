import kagglehub
import argparse
import os 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Downloader...')
    
    parser.add_argument(
    '--kaggle-token',
    type=str,
    help='Path to the Kaggle API token JSON file',
    required=True,)

    parser.add_argument(
        '--download',
        nargs='+',
        choices=['custom', 'hagrid-512p', 'hagrid-384p'],
        help='''Specify what to download (choose from 'custom', 'hagrid-512p', 'hagrid-384p')''',
        required=True,
    )


    args = parser.parse_args()

    if not os.path.isfile(args.kaggle_token):
        raise FileNotFoundError(f'Kaggle token file not found: {args.kaggle_token}')

    selected_datasets = args.download
    
    if len(selected_datasets) == 0:
        raise ValueError("No datasets selected. Please specify at least one dataset to download.")

    for dataset in selected_datasets:
        if dataset == 'custom':
            path = kagglehub.dataset_download("kianooshvadaei/castume-hand-gesture-dataset")
            print("Path to custom dataset files:", path)
        elif dataset == 'hagrid-512p':
            path = kagglehub.dataset_download("innominate817/hagrid-classification-512p")
            print("Path to hagrid-512p dataset files:", path)

        elif dataset == 'hagrid-384p':
            path = kagglehub.dataset_download("innominate817/hagrid-sample-30k-384p")
            print("Path to hagrid-384p dataset files:", path)


    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

    os.rename(args.kaggle_token, os.path.expanduser("~/.kaggle/kaggle.json"))

    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    print('Done!')
            
    