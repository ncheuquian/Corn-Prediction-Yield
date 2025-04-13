# Maize Yield Prediction Using TensorFlow

A machine learning application for predicting maize crop yields from multi-spectral satellite imagery using TensorFlow.

## Project Overview

This application analyzes satellite imagery data from 2022 and 2023 maize field experiments to predict crop yields. It leverages deep learning to extract meaningful features from multi-spectral satellite images and correlates them with ground truth yield data.

### Key Features

- **Multi-spectral image analysis**: Processes 6-band satellite images (near-infrared, red edge, red, green, blue, and deep blue)
- **Vegetation indices calculation**: Automatically computes NDVI, GNDVI, NDRE, and EVI indices
- **Temporal analysis**: Integrates data from multiple time points throughout the growing season
- **Deep learning architecture**: Uses CNNs to extract spatial features from satellite imagery
- **Yield prediction**: Predicts maize yield in bushels per acre

## Dataset

The model is designed to work with the maize crop performance dataset that includes:

- **Satellite imagery**: Multi-spectral TIFF files with 6 bands (near-infrared, red edge, red, green, blue, and deep blue)
- **Ground truth data**: CSV files containing yield measurements and other crop metrics
- **Multiple locations**: Data from fields in Nebraska and Iowa (2022-2023)
- **Temporal data**: Multiple time points (TP1-TP6) throughout the growing season

## Requirements

- Python 3.7+
- TensorFlow 2.x
- numpy
- pandas
- rasterio
- scikit-learn
- matplotlib
- tqdm

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/maize-yield-prediction.git
cd maize-yield-prediction

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

1. Organize your data according to the following directory structure:
   ```
   data/
   ├── Satellite/
   │   ├── Lincoln/
   │   │   ├── TP1/
   │   │   │   ├── Lincoln-TP1-hybrids_1_1.TIF
   │   │   │   ├── Lincoln-TP1-hybrids_1_2.TIF
   │   │   │   └── ...
   │   │   ├── TP2/
   │   │   └── ...
   │   ├── Missouri Valley/
   │   └── ...
   ├── GroundTruth/
   │   ├── HYBRID_HIPS_V3.5_ALLPLOTS.csv  # 2022 data
   │   ├── train_HIPS_HYBRIDS_2023_V2.3.csv  # 2023 data
   │   └── DateofCollection.xlsx
   └── Documentation/
       └── ...
   ```

2. Update the `DATA_DIR` path in the configuration section of the code.

## Usage

### Basic Training

```python
from maize_yield_predictor import MaizeYieldPredictor

# Initialize the predictor
predictor = MaizeYieldPredictor(data_dir="/path/to/data")

# Run the full pipeline
model, metrics = predictor.run_pipeline()

# Save the model
predictor.save_model(model)
```

### Custom Training

```python
# Customize training parameters
model, metrics = predictor.run_pipeline(
    years=[2023],  # Use only 2023 data
    locations=["Lincoln", "Missouri Valley"],  # Specific locations
    time_points=[1, 2, 3]  # Specific time points
)
```

### Making Predictions

```python
# Load a saved model and make predictions
from tensorflow.keras.models import load_model
import pickle

# Load model and scaler
model = load_model("models/maize_yield_predictor.h5")
with open("models/yield_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize predictor with scaler
predictor = MaizeYieldPredictor(data_dir="/path/to/data")
predictor.scaler = scaler

# Predict on new images
image_paths = [
    "/path/to/new/images/Lincoln-TP1-hybrids_1_1.TIF",
    "/path/to/new/images/Lincoln-TP1-hybrids_1_2.TIF"
]
predictions = predictor.predict_new_data(model, image_paths)

for path, yield_prediction in predictions:
    print(f"Image: {path}, Predicted Yield: {yield_prediction:.2f} bushels/acre")
```

## Model Architecture

The default model architecture is a Convolutional Neural Network (CNN) designed for spatial feature extraction:

- **Input**: Multi-spectral satellite image with vegetation indices (10 channels total)
- **Convolutional layers**: Multiple convolutional blocks with batch normalization and max pooling
- **Feature extraction**: Global average pooling to extract features
- **Fully connected layers**: Dense layers with dropout for regularization
- **Output**: Single regression output for yield prediction

The code also includes an alternate temporal model architecture that can process multiple time points:

```python
# Build a temporal model
time_points = 4  # Number of time points to use
input_shape = (64, 64, 10)  # (height, width, features)
temporal_model = build_temporal_model(time_points, input_shape)
```

## Evaluation

The model is evaluated using several metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Visualizations include:
- Training and validation loss curves
- Predicted vs actual yield scatter plots

## Future Improvements

- [ ] Weather data integration
- [ ] Ensemble modeling
- [ ] Transfer learning from pre-trained CNNs
- [ ] Explainability techniques (Grad-CAM)
- [ ] Web interface for easy model use
- [ ] Early-season yield forecasting

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Data sourced from multi-state maize yield trials
- Inspired by research in precision agriculture and remote sensing
