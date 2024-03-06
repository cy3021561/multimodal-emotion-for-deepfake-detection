# Multimodal Emotion Recognition for Deepfake Detection

This repository is an implementation of the original multimodal-emotion-recognition project, adapted for the purpose of deepfake video detection. Our current focus is on utilizing the DFDC (DeepFake Detection Challenge) dataset available on Kaggle for training our models. You can find the dataset [here](https://www.kaggle.com/c/deepfake-detection-challenge/data).

## Dataset Preprocessing

The implementation includes several preprocessing steps to prepare the DFDC dataset for training:

1. **Detect People Count in Videos**: We currently focus on videos featuring a single person. The detection results are saved in `final_metadata.json`.
   - Script: `dfdc_preprocessing/face_track.py`

2. **Extract Audio from Video**: We separate the audio component from the raw video samples.
   - Script: `dfdc_preprocessing/prepare_raw_dataset.py`

3. **Extract Cropped Face Segments**: Faces are cropped from the raw video and saved as `.npy` files for training.
   - Script: `dfdc_preprocessing/extract_faces.py`

4. **Extract Cropped Audio**: Corresponding audio segments are extracted from the cropped videos and saved as `.wav` files.
   - Script: `dfdc_preprocessing/extract_audios.py`

5. **Balance the Dataset**: To address dataset imbalance, the raw dataset is split into a new, balanced dataset.
   - Script: `dfdc_preprocessing/split_dataset.py`

6. **Create Annotations**: We generate `annotation.txt` files from the processed dataset for use with PyTorch's DataLoader.
   - Script: `dfdc_preprocessing/create_annotations.py`

## Training

The model is trained for binary classification (real or fake), thus `n_classes` is set to 2. 

Training command:
`python main.py --annotation_path='./annotations.txt' --n_classes=2 --dataset='DFDC'`


## Future Improvements

We acknowledge the complexity of the preprocessing procedure and plan to introduce a comprehensive script to streamline these steps once the final architecture for this implementation is decided.

## Acknowledgments

This implementation modifies and extends the original multimodal-emotion-recognition project to focus on deepfake detection. We are grateful to the original authors and contributors of the multimodal-emotion-recognition repository and the creators of the DFDC dataset for making their resources available.
