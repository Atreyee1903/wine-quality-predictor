<<<<<<< HEAD
WINE QUALITY PREDICTION
=======
# ai_wineQualitypredictor

A Streamlit-based wine quality prediction app that uses a LightGBM ensemble to estimate the wine quality score and a second ensemble to estimate the final quality verdict confidence.

## Project Overview

This project predicts wine quality from laboratory chemistry values such as acidity, chlorides, sulfur dioxide, density, pH, sulphates, and alcohol.

The app provides:

- Predicted wine quality score such as `5`, `6`, `7`, or `8`
- Quality verdict: `Good Quality` or `Needs Improvement`
- Prediction confidence for the final verdict
- Model snapshot metrics for score prediction and verdict prediction

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- LightGBM

## Model Design

The app uses two ensemble models at the same time:

- Score model:
  - Random Forest
  - LightGBM
  - Extra Trees
- Verdict model:
  - Random Forest
  - Extra Trees
  - LightGBM

Why two models are used:

- The score model predicts the exact wine class like `6` or `7`
- The verdict model predicts whether the wine is `Good Quality` or `Needs Improvement`
- This makes the confidence score more meaningful for the final decision

## Dataset

Dataset file used:

- `wine__12.csv`

The quality values in this dataset range from:

- `3` to `8`

## Features Used

Input features:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Engineered features inside the model:

- Total acidity
- Fixed to volatile ratio
- Sulfur ratio
- Sulfur load
- Alcohol density ratio
- Sulphates chlorides ratio
- Residual sugar alcohol ratio
- pH sulphates interaction

## Installation

Open PowerShell inside the project folder and run:

```powershell
cd "c:\Users\TUNNA\OneDrive\Desktop\wine quality prediction"
python -m pip install -r requirements.txt
```

If `python` does not work, use:

```powershell
py -m pip install -r requirements.txt
```

## Run the App

```powershell
python -m streamlit run wine.py
```

If needed, use:

```powershell
py -m streamlit run wine.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Input

The following batch values were tested in the app:

- Fixed Acidity: `7.2000`
- Volatile Acidity: `0.3000`
- Citric Acid: `0.3900`
- Residual Sugar: `2.0000`
- Chlorides: `0.0340`
- Free Sulfur Dioxide: `25.0000`
- Total Sulfur Dioxide: `80.0000`
- Density: `0.9907`
- pH: `3.2119`
- Sulphates: `0.8600`
- Alcohol: `12.8000`

## Example Output

For the above sample input, the app produced:

- Predicted Score: `6`
- Quality Verdict: `Good Quality`
- Prediction Confidence: `73.3%`

## Model Snapshot From Current Output

Current app output shown in the screenshot:

- Score CV Accuracy: `58.14%`
- Score Test Accuracy: `60.66%`
- Training Rows: `1359`
- Quality Range: `3 - 8`
- Score Model: `LightGBM Ensemble`
- Verdict Model: `LightGBM Ensemble`
- Verdict CV Accuracy: `87.67%`
- Verdict Test Accuracy: `86.03%`

## How to Use

1. Start the Streamlit app.
2. Enter the wine chemistry values in the batch input form.
3. Click `Predict Wine Quality`.
4. Read:
   - the exact predicted score
   - the verdict
   - the confidence

## Notes

- `Predicted Score` is the exact predicted wine class.
- `Quality Verdict` is based on whether the predicted/estimated quality belongs to the good-quality side.
- `Prediction Confidence` reflects how confident the verdict model is about the final verdict.

## Files

- `wine.py` - main Streamlit application
- `requirements.txt` - project dependencies
- `wine__12.csv` - dataset used for training

## Future Improvements

- Add screenshot images directly into the README
- Save trained models to disk for faster startup
- Add batch CSV upload for bulk predictions
- Add deployment instructions for Streamlit Cloud or Render
>>>>>>> eb3e4e8 (Final commit)
