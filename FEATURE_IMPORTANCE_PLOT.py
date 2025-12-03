"""
Feature Importance Analysis for Muon Signal Detection

This script creates visualizations showing which features contribute
most to muon signal discrimination using Random Forest feature importance.

Author: Muhammad Zain
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import os

def main():
    print("Creating Feature Importance Analysis...")
    print("=" * 70)

    try:
        # Load data
        data_path = r"C:\Users\zainy\Desktop\degree\muons-CsI-w-tube-fr64\PERFECT_FINAL_RESULTS.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded analysis results: {len(df)} frames")

        # Features
        feature_cols = ['scint_corr', 'mirror_corr', 'mean_pixels_on_scintillator', 'mean_pixels_on_mirror', 'mean_pixels_on_bkground']
        X = df[feature_cols].dropna()
        y = df['isSignal'].loc[X.index]

        # Train Random Forest
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
        rf.fit(X_scaled, y)
        importance = rf.feature_importances_ * 100

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': ['Temperature-Corrected\nScintillator', 'Temperature-Corrected\nMirror', 'Raw Scintillator\nIntensity', 'Raw Mirror\nIntensity', 'Background\nIntensity'],
            'Short_Name': ['scint_corr', 'mirror_corr', 'scint_raw', 'mirror_raw', 'background'],
            'Importance': importance,
            'Type': ['Temperature-Corrected', 'Temperature-Corrected', 'Raw Intensity', 'Raw Intensity', 'Raw Intensity']
        })
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

        # Display results
        print("\nFeature Importance Rankings:")
        print("-" * 40)
        for idx, row in importance_df.iterrows():
            rank = "1st" if idx == 0 else "2nd" if idx == 1 else "3rd" if idx == 2 else f"{idx+1}th"
            print(f"  {rank}: {row['Short_Name']} - {row['Importance']:.2f}%")

        # Create output directory
        os.makedirs('output', exist_ok=True)

        # Create plot
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#2E86AB' if t == 'Temperature-Corrected' else '#A23B72' for t in importance_df['Type']]
        bars = ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature Importance (%)', fontsize=16, fontweight='bold')
        ax.set_title('Feature Importance for Muon Detection\n(Random Forest Analysis)', fontsize=18, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(importance_df['Importance']) + 5)

        for i, (bar, imp) in enumerate(zip(bars, importance_df['Importance'])):
            ax.text(imp + 0.5, i, f'{imp:.1f}%', va='center', fontsize=13, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Temperature-Corrected', alpha=0.8),
            Patch(facecolor='#A23B72', label='Raw Intensity', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=13, framealpha=0.9)

        scintillator_importance = importance_df[importance_df['Short_Name'].str.contains('scint')]['Importance'].sum()
        key_finding = f'Key Finding:\nScintillator features = {scintillator_importance:.1f}%\nTemperature correction is crucial!'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.98, 0.5, key_finding, transform=ax.transAxes, fontsize=12, verticalalignment='center', bbox=props, ha='right', fontweight='bold')

        plt.tight_layout()
        plt.savefig('output/FEATURE_IMPORTANCE_Simple.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: FEATURE_IMPORTANCE_Simple.png")

        print("\nFeature importance analysis completed successfully!")
        print("Plot saved to: output/FEATURE_IMPORTANCE_Simple.png")

    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please ensure PERFECT_FINAL_RESULTS.csv exists.")

if __name__ == "__main__":
    main()