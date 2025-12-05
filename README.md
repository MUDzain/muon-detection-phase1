# Muon Detection Analysis - Phase 1



A comprehensive muon signal discrimination system for the RIPTIDE detector using machine learning and statistical methods. This Phase 1 analysis evaluates four different approaches for detecting rare muon signals in detector data with extreme class imbalance (39 signals vs 16,089 background events).

##  Key Results

- **Random Forest**: 100% precision and recall (perfect detection of all 39 true muons)
- **Dataset**: 16,128 frames from 14 FLI files, 0.24% signal rate
- **Evaluation**: Balanced Accuracy (appropriate for imbalanced classification)

##  Methods Implemented

### 1. Gaussian Mixture Model (GMM) - Unsupervised
- Uses 2 features: temperature-corrected scintillator and mirror values
- Automatically discovers signal patterns without labeled data
- **Balanced Accuracy**: 97.8%, **Recall**: 97.4%

### 2. Random Forest - Supervised Ensemble
- Uses 5 features: corrected scint/mirror + raw scint/mirror + background
- Trained on all 13 labeled CSV files (39 true signals)
- **Balanced Accuracy**: 91.7%, **Precision**: 100.0%, **Recall**: 100.0% 

### 3. Artificial Neural Network (ANN) - Deep Learning
- 2 hidden layers (100, 50 neurons), ReLU activation
- Trained on full labeled dataset with early stopping
- **Balanced Accuracy**: 87.5%, **Precision**: 100.0%, **Recall**: 75.0%

### 4. Threshold-based Detection - Rule-based
- Simple rule: `scint_corr > 1.545`
- No training required, real-time capable
- **Balanced Accuracy**: 98.5%, **Recall**: 100.0%

## Features Used

1. **scint_corr**: Temperature-corrected scintillator intensity
   - Formula: `mean_pixels_on_scintillator - (0.07 × temperature + 27.54)`
2. **mirror_corr**: Temperature-corrected mirror intensity
   - Formula: `mean_pixels_on_mirror - (0.074 × temperature + 27.576)`
3. **mean_pixels_on_scintillator**: Raw scintillator pixel intensity
4. **mean_pixels_on_mirror**: Raw mirror pixel intensity
5. **mean_pixels_on_bkground**: Background pixel intensity

### Usage
```bash
# Navigate to the project directory
cd muon-detection-phase1

# Run the complete analysis
python PERFECT_FINAL_ANALYSIS.py
```

### Expected Output
- **PERFECT_All_Methods_Comparison.png**: Scatter plots comparing all 4 methods
- **PERFECT_True_vs_BestMethod.png**: True labels vs Random Forest predictions
- **PERFECT_Performance_Comparison.png**: Performance metrics comparison
- **PERFECT_FINAL_RESULTS.csv**: Complete dataset with all predictions

## Project Structure

```
├── PERFECT_FINAL_ANALYSIS.py     # Main analysis script
├── output/                       # Generated plots and results
│   ├── PERFECT_All_Methods_Comparison.png
│   ├── PERFECT_True_vs_BestMethod.png
│   ├── PERFECT_Performance_Comparison.png
│   └── PERFECT_FINAL_RESULTS.csv
└── README.md               
```

## Feature Documentation
**Detailed explanation of how `extracted_roi_data.csv` is created and temperature correction works:**

- **CSV Structure**: Explains the 6 raw columns (`filename`, `frame_index`, `temperature`, `mean_pixels_on_scintillator`, `mean_pixels_on_mirror`, `mean_pixels_on_bkground`)
- **Temperature Correction**: Details the mathematical formulas for `scint_corr` and `mirror_corr` features
- **Why Correction Matters**: Shows how temperature-dependent baseline drift is removed to enable proper signal detection
- **Feature Importance**: Documents which features are most important for muon detection (scint_corr: 34.7%, mirror_corr: 16.7%)
- **Complete Workflow**: From raw FLI files → ROI extraction → temperature correction → final 8-column CSV

**Key Formulas:**
- `scint_corr = mean_pixels_on_scintillator - (0.07 × temperature + 27.54)`
- `mirror_corr = mean_pixels_on_mirror - (0.074 × temperature + 27.576)`

##  Research Goals Achievement

- **"Keep as much signal as possible"**: Threshold method (100% recall)
- **"Get rid of background"**: Random Forest (100% precision)


## Technical Details

- **Data Source**: 14 FLI image stacks (16,128 frames total)
- **Labels**: 13 CSV files with ground truth (39 muon signals confirmed)
- **Temperature Correction**: Linear regression-derived formulas for baseline drift removal
- **Evaluation**: Balanced Accuracy = (Sensitivity + Specificity) / 2
- **Memory Optimization**: Data sampling for large scatter plots

##  Citation

If you use this code in your research, please cite:

```bibtex
@software{muon_detection_phase1,
  title = {Muon Detection Analysis - Phase 1},
  author = {Zain, Muhammad},
  url = {https://github.com/MUDzain/muon-detection-phase1},
  year = {2025}
}
```


**Muhammad Zain** - [GitHub](https://github.com/MUDzain)

---

**Note**: This is Phase 1 of a multi-phase muon detection project. Phase 1 focuses on method comparison and evaluation using existing labeled data.
