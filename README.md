

## UrbanSound8K Audio Classification

This repository provides a comprehensive solution for classifying environmental sounds using the UrbanSound8K dataset. The project involves data preprocessing, augmentation, and model training with advanced techniques to achieve high accuracy in sound classification tasks.

### Key Features

- **Data Preprocessing**: 
  - Converts audio files into mel spectrograms.
  - Applies normalization and resizes spectrograms to a fixed size.
  - Handles stereo to mono conversion for consistent input.

- **Audio Augmentation**: 
  - Utilizes `audiomentations` to apply various augmentations:
    - Gaussian Noise
    - Time Stretch
    - Pitch Shift

- **Model Training**: 
  - Implements a custom Convolutional Neural Network (CNN) tailored for audio classification.
  - Uses dropout and batch normalization to prevent overfitting.

- **Hyperparameter Tuning**: 
  - Integrates `Optuna` for hyperparameter optimization.
  - Tunes parameters such as learning rate, weight decay, dropout rate, and batch size.

- **Cross-Validation**: 
  - Performs cross-validation to ensure model robustness and generalization.
  - Trains and evaluates the model across multiple folds.

### Project Structure

- `data_preprocessing.py`: Script for preprocessing audio files, applying augmentations, and saving processed data.
- `train_with_optuna.py`: Script for hyperparameter tuning using Optuna.
- `train.py`: Script for training the model using the best hyperparameters and applying cross-validation.
- `advanced_model.py`: Contains the definition of the advanced audio classification model.

### Getting Started

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/UrbanSound8K-Audio-Classification.git
    cd UrbanSound8K-Audio-Classification
    ```

2. **Install the required packages**:

    ```bash
    pip install torch torchaudio audiomentations optuna wandb numpy pandas
    ```

3. **Download the UrbanSound8K dataset** from [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html) and extract it to a directory.

4. **Update the `data_path` and `metadata_path`** variables in `data_preprocessing.py` to point to the location of your dataset.

5. **Run Data Preprocessing**:

    ```bash
    python data_preprocessing.py
    ```

6. **Hyperparameter Tuning**:

    ```bash
    python train_with_optuna.py
    ```

7. **Model Training**:

    ```bash
    python train.py
    ```

### Acknowledgements

This project uses the UrbanSound8K dataset. Special thanks to the authors of this dataset for making it publicly available.

