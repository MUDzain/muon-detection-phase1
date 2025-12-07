"""
Muon Detection: Train and Predict
trains a Random Forest model on labeled muon data(64 files),
then uses it to detect muons in new(128)FLI files.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flifile import FliFile
from flifile.readheader import readheader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ROI coordinates (Region of Interest) - where to look for muon signals
ROI_SCINTILLATOR = {'x': 164, 'y': 333, 'w': 745, 'h': 699}
ROI_MIRROR = {'x': 968, 'y': 329, 'w': 609, 'h': 582}
ROI_BACKGROUND = {'x': 229, 'y': 19, 'w': 1440, 'h': 249}

# Temperature correction formulas: corrected_value = raw_value - (slope * temp + intercept)
TEMP_CORRECTION_SCINTILLATOR = {'slope': 0.07, 'intercept': 27.54}
TEMP_CORRECTION_MIRROR = {'slope': 0.074, 'intercept': 27.57}

# Features used for ML
FEATURE_NAMES = [
    'scint_corr',                    # Temperature-corrected scintillator signal
    'mirror_corr',                   # Temperature-corrected mirror signal
    'mean_pixels_on_scintillator',  # Raw scintillator brightness
    'mean_pixels_on_mirror',         # Raw mirror brightness
    'mean_pixels_on_bkground'        # Raw background brightness
]

# Random Forest parameters(WELL ITS SAME)
RF_N_TREES = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42


class MuonDetector:
    """
    A machine learning detector that identifies muon signals in detector images.
    
    How it works:
    1. Train: Learns from labeled examples (muon vs background)
    2. Predict: Analyzes new images and identifies muon signals
    """
    
    def __init__(self):
        """Initialize the detector (model not trained yet)"""
        self.model = None
        self.scaler = None
    
    
    def train(self, training_data_csv):
        """
        Train the Random Forest model on labeled data.
        
        Args:
            training_data_csv: Path to CSV file with features and labels
            Must have columns: scint_corr, mirror_corr, etc. and 'isSignal'
        """
        print("\n" + "="*80);
        print("STEP 1: TRAINING RANDOM FOREST MODEL")
        print("="*80)
        
        # Load training data
        print(f"\nLoading training data from:\n  {training_data_csv}")
        data = pd.read_csv(training_data_csv)
        
        print(f"\nDataset summary:")
        print(f"  Total frames: {len(data)}")
        print(f"  Muon signals: {(data['isSignal'] == 1).sum()}")
        print(f"  Background: {(data['isSignal'] == 0).sum()}")
        
        # Extract features (X) and labels (y)
        X = data[FEATURE_NAMES].dropna()
        y = data['isSignal'].loc[X.index]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3,           # 30% for testing
            random_state=RF_RANDOM_STATE,
            stratify=y               # Keep same muon/background ratio in both sets
        )
        
        print(f"\nSplit into:")
        print(f"  Training set: {len(X_train)} frames")
        print(f"  Test set: {len(X_test)} frames")
        
        # Normalize features (makes training more stable)
        print("\nNormalizing features.")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the Random Forest
        print(f"\nTraining Random Forest ({RF_N_TREES} trees)...")
        self.model = RandomForestClassifier(
            n_estimators=RF_N_TREES,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
            class_weight='balanced'   # Handle imbalanced data (few muons, many background)
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Test the model
        print("\nEvaluating model on test set.")
        predictions = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print("\n" + "-"*80)
        print("MODEL PERFORMANCE:")
        print("-"*80)
        print(f"  Accuracy:  {accuracy:.1%} - How often the model is correct")
        print(f"  Precision: {precision:.1%} - When it says 'muon', how often is it right?")
        print(f"  Recall:    {recall:.1%} - What fraction of real muons does it catch?")
        print(f"  F1-Score:  {f1:.3f} - Overall balance of precision and recall")
        
        # Show which features are most important(OF COURSE, WE KNOW IT, FROM THE PLOT, PREVISE PLOTTING)
        importance = self.model.feature_importances_
        print("\n" + "-"*80)
        print("FEATURE IMPORTANCE (which signals matter most?):")
        print("-"*80)
        for feature, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True):
            print(f"  {feature:30s} {imp:.1%}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
    
    
    def extract_features(self, image, temperature):
        """
        Extract the 5 features from a detector image.
        
        Args:
            image: 2D numpy array (detector image)
            temperature: Detector temperature in °C
            
        Returns:
            Dictionary with 5 feature values
        """
        # Cut out the three regions of interest from the image
        scintillator = image[
            ROI_SCINTILLATOR['y']:ROI_SCINTILLATOR['y']+ROI_SCINTILLATOR['h'],
            ROI_SCINTILLATOR['x']:ROI_SCINTILLATOR['x']+ROI_SCINTILLATOR['w']
        ]
        
        mirror = image[
            ROI_MIRROR['y']:ROI_MIRROR['y']+ROI_MIRROR['h'],
            ROI_MIRROR['x']:ROI_MIRROR['x']+ROI_MIRROR['w']
        ]
        
        background = image[
            ROI_BACKGROUND['y']:ROI_BACKGROUND['y']+ROI_BACKGROUND['h'],
            ROI_BACKGROUND['x']:ROI_BACKGROUND['x']+ROI_BACKGROUND['w']
        ]
        
        # Calculate average brightness in each region
        mean_scint = np.mean(scintillator)
        mean_mirror = np.mean(mirror)
        mean_bg = np.mean(background)
        
        # Apply temperature correction(detector brightness changes with temperature, so we correct for it, HERE ONE THINK I NOITED 64FPS MOSTLY HAVE THE SAME RAW TEMP, IF WE COMPARED WITH THE 128 FPS, ITS DIFFERENT )
        temp_corr_scint = (TEMP_CORRECTION_SCINTILLATOR['slope'] * temperature + 
                          TEMP_CORRECTION_SCINTILLATOR['intercept'])
        temp_corr_mirror = (TEMP_CORRECTION_MIRROR['slope'] * temperature + 
                           TEMP_CORRECTION_MIRROR['intercept'])
        
        scint_corr = mean_scint - temp_corr_scint
        mirror_corr = mean_mirror - temp_corr_mirror
        
        return {
            'scint_corr': scint_corr,
            'mirror_corr': mirror_corr,
            'mean_pixels_on_scintillator': mean_scint,
            'mean_pixels_on_mirror': mean_mirror,
            'mean_pixels_on_bkground': mean_bg
        }
    
    
    def predict_muons_in_file(self, fli_file_path):
        """
        Analyze all frames in a FLI file and detect muons.
        
        Args:
            fli_file_path: Path to .fli file
            
        Returns:
            DataFrame with predictions for each frame
        """
        filename = Path(fli_file_path).name
        print(f"\n  Processing: {filename}")
        
        try:
            # Load the FLI file
            fli = FliFile(str(fli_file_path))
            images = fli.getdata()
            n_frames = images.shape[2]
            
            # Get detector temperature from file header (the correct way!)
            try:
                header = readheader(str(fli_file_path))
                temperature = float(header[0]['PARAMETERS']['DEVICE SETTINGS']['Intensifier_TECobjectTemp'])
            except:
                # Fallback to default if header read fails
                temperature = 27.5
                print(f"    WARNING: Could not read temperature from header, using default {temperature}°C")
            
            print(f"    Frames: {n_frames}, Temperature: {temperature:.1f}°C")
            
        except Exception as e:
            print(f"    ERROR: Could not load file - {e}")
            return None
        
        # Analyze each frame
        results = []
        for frame_idx in range(n_frames):
            # Get the frame and transpose it
            frame = images[:, :, frame_idx].T
            
            # Extract features
            features = self.extract_features(frame, temperature)
            
            # Make prediction
            feature_vector = np.array([[features[f] for f in FEATURE_NAMES]])
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            is_muon = self.model.predict(feature_vector_scaled)[0]
            muon_probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            
            # Save result
            results.append({
                'frame_index': frame_idx,
                'prediction': is_muon,
                'probability': muon_probability,
                **features
            })
        
        df = pd.DataFrame(results)
        
        # Print summary
        n_muons = (df['prediction'] == 1).sum()
        muon_rate = n_muons / n_frames * 100
        print(f"    DETECTED: {n_muons} muons ({muon_rate:.1f}%)")
        
        return df
    
    
    def process_all_files(self, fli_file_list, output_folder):
        """
        Process multiple FLI files and save results.
        
        Args:
            fli_file_list: List of paths to FLI files
            output_folder: Where to save CSV results
            
        Returns:
            (all_predictions_df, summary_df)
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*80)
        print(f"STEP 2: ANALYZING {len(fli_file_list)} FLI FILES")
        print("="*80)
        
        all_predictions = []
        summary_stats = []
        
        # Process each file
        for i, fli_path in enumerate(fli_file_list, 1):
            print(f"\n[{i}/{len(fli_file_list)}]", end="")
            
            df_predictions = self.predict_muons_in_file(fli_path)
            
            if df_predictions is not None:
                # Add filename to results
                df_predictions['filename'] = Path(fli_path).name
                all_predictions.append(df_predictions)
                
                # Calculate summary
                n_frames = len(df_predictions)
                n_muons = (df_predictions['prediction'] == 1).sum()
                avg_prob = (df_predictions[df_predictions['prediction'] == 1]['probability'].mean() 
                           if n_muons > 0 else 0)
                
                summary_stats.append({
                    'filename': Path(fli_path).name,
                    'total_frames': n_frames,
                    'detected_muons': n_muons,
                    'muon_rate_%': n_muons / n_frames * 100,
                    'avg_confidence': avg_prob
                })
        
        # Save all predictions(WELL VERY IMP FILE THE all_predictions.csv, WHICH USED LATER ALSO)
        df_all = None
        df_summary = None
        
        if all_predictions:
            df_all = pd.concat(all_predictions, ignore_index=True)
            predictions_file = output_folder / "all_predictions.csv"
            df_all.to_csv(predictions_file, index=False)
            print(f"\n\n Saved detailed predictions: {predictions_file}")
        
        if summary_stats:
            df_summary = pd.DataFrame(summary_stats)
            summary_file = output_folder / "detection_summary.csv"
            df_summary.to_csv(summary_file, index=False)
            print(f" Saved summary: {summary_file}")
            
            # Print final summary
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            print(f"Files processed: {len(summary_stats)}")
            print(f"Total frames analyzed: {df_summary['total_frames'].sum():,}")
            print(f"Total muons detected: {df_summary['detected_muons'].sum()}")
            print(f"Overall detection rate: {df_summary['detected_muons'].sum() / df_summary['total_frames'].sum() * 100:.2f}%")
            print("\nPer-file statistics:")
            print(f"  Minimum muons in a file: {df_summary['detected_muons'].min()}")
            print(f"  Maximum muons in a file: {df_summary['detected_muons'].max()}")
            print(f"  Average muons per file: {df_summary['detected_muons'].mean():.1f}")
            print("="*80)
        
        return df_all, df_summary


# MAIN EXECUTION

def main():
    """Run the complete analysis: train model, then predict on new files."""
    
    print("\n" + "="*80)
    print("MUON DETECTION PIPELINE")
    print("="*80)
    
    #Training data (14 FLI files with labels, FROM THE 64 FPS DATASET)
    training_data_csv = r"C:\Users\zainy\Desktop\degree\muons-CsI-w-tube-fr64\PERFECT_FINAL_RESULTS.csv"
    
    # New FLI files to analyze (33 files, 128 frames each)
    fli_folder = Path(r"C:\Users\zainy\Desktop\degree\muons-CsI-w-tube-fr64\muons-CsI-w-tube-fr128")
    fli_files = [
        fli_folder / "muons-w-tube-fr64_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(1)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(2)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(3)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(4)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(5)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(6)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(7)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(8)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(9)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(10)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(11)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(12)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(13)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(14)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(15)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(16)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(17)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(18)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(19)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(20)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(21)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(22)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(23)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(24)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(25)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(26)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(27)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(28)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(29)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(30)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(31)_HiCAM FLUO_1200-2-418.fli",
        fli_folder / "muons-w-tube-fr64(32)_HiCAM FLUO_1200-2-418.fli",
    ]
    
    output_folder = Path("src/128frame_predictions")
    
    # Run analysis
    # Create and train detector
    detector = MuonDetector()
    detector.train(training_data_csv)
    
    df_all, df_summary = detector.process_all_files(fli_files, output_folder)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_folder}/")
    print(f"  - all_predictions.csv (detailed frame-by-frame)")
    print(f"  - detection_summary.csv (per-file summary)")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

#