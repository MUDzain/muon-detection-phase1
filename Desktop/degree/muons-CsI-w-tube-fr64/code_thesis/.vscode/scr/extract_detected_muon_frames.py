"""
Extract and Save Detected Muon Frames as PNG Images
Finds all frames where muons were detected and saves them as PNG images
with ROI annotations (scintillator and mirror boxes).
"""

import pandas as pd
import numpy as np
import cv2
from flifile import FliFile
from pathlib import Path

print("="*80)
print("EXTRACTING DETECTED MUON FRAMES")
print("="*80)

# Load predictions
predictions_path = Path("src/128frame_predictions/all_predictions.csv")
df = pd.read_csv(predictions_path)

# Filter muon detections
muons = df[df['prediction'] == 1].copy()

print(f"\nTotal muon detections: {len(muons)}")
print(f"\nMuons by file:")

# Group by filename to see distribution
muon_counts = muons.groupby('filename').size()
print(muon_counts)

# ROI coordinates
roi_coords = {
    'scintillator': {'x': 164, 'y': 333, 'w': 745, 'h': 699},
    'mirror': {'x': 968, 'y': 329, 'w': 609, 'h': 582},
    'background': {'x': 229, 'y': 19, 'w': 1440, 'h': 249}
}

# Create output directory
output_dir = Path("src/128frame_predictions/detected_muon_frames")
output_dir.mkdir(exist_ok=True, parents=True)

# FLI directory
fli_dir = Path(r"C:\Users\zainy\Desktop\degree\muons-CsI-w-tube-fr64\muons-CsI-w-tube-fr128")

print("\n" + "="*80)
print("EXTRACTING FRAMES")
print("="*80)

extracted_count = 0

for idx, row in muons.iterrows():
    filename = row['filename']
    frame_idx = int(row['frame_index'])
    probability = row['probability']
    scint_corr = row['scint_corr']
    mirror_corr = row['mirror_corr']
    
    fli_path = fli_dir / filename
    
    print(f"\n[{extracted_count + 1}/{len(muons)}] {filename}")
    print(f"  Frame index: {frame_idx}")
    print(f"  Muon probability: {probability:.3f}")
    print(f"  scint_corr: {scint_corr:.3f}, mirror_corr: {mirror_corr:.3f}")
    
    try:
        # Load FLI file
        fli = FliFile(str(fli_path))
        data = fli.getdata()
        
        # Extract specific frame and transpose
        frame = data[:, :, frame_idx].T
        
        # Normalize to 8-bit for visualization
        frame_8bit = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        
        # Create color version for annotations
        frame_color = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)
        
        # Draw Scintillator ROI (RED)
        sc = roi_coords['scintillator']
        cv2.rectangle(frame_color,
                     (sc['x'], sc['y']),
                     (sc['x'] + sc['w'], sc['y'] + sc['h']),
                     (0, 0, 255), 3)
        cv2.putText(frame_color, 'SCINTILLATOR',
                   (sc['x'] + 10, sc['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # Draw Mirror ROI (BLUE)
        mr = roi_coords['mirror']
        cv2.rectangle(frame_color,
                     (mr['x'], mr['y']),
                     (mr['x'] + mr['w'], mr['y'] + mr['h']),
                     (255, 0, 0), 3)
        cv2.putText(frame_color, 'MIRROR',
                   (mr['x'] + 10, mr['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        
        # Add detection info
        info_text = f"MUON DETECTED | Frame {frame_idx} | P={probability:.3f}"
        cv2.putText(frame_color, info_text,
                   (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        values_text = f"Scint={scint_corr:.2f} | Mirror={mirror_corr:.2f}"
        cv2.putText(frame_color, values_text,
                   (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save frame
        file_prefix = filename.replace('muons-w-tube-fr64', 'file').replace('_HiCAM FLUO_1200-2-418.fli', '')
        if file_prefix.startswith('file('):
            file_prefix = file_prefix.replace('file(', 'file').replace(')', '')
        
        output_filename = f"muon_{file_prefix}_frame{frame_idx:04d}_prob{probability:.2f}.png"
        output_path = output_dir / output_filename
        
        cv2.imwrite(str(output_path), frame_color)
        print(f"   Saved: {output_filename}")
        
        extracted_count += 1
        
    except Exception as e:
        print(f"  Error extracting frame: {e}")

print("\n" + "="*80)
print("EXTRACTION COMPLETE!")
print("="*80)
print(f"Successfully extracted: {extracted_count}/{len(muons)} muon frames")
print(f"Saved to: {output_dir.absolute()}")
print("="*80)

# Create summary
summary_path = output_dir / "muon_frames_summary.txt"
with open(summary_path, 'w') as f:
    f.write("DETECTED MUON FRAMES SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total muon detections: {len(muons)}\n")
    f.write(f"Successfully extracted: {extracted_count}\n\n")
    f.write("Distribution by file:\n")
    f.write("-"*60 + "\n")
    for filename, count in muon_counts.items():
        f.write(f"{filename}: {count} muons\n")
    f.write("\n")
    f.write("Frame details:\n")
    f.write("-"*60 + "\n")
    for idx, row in muons.iterrows():
        f.write(f"File: {row['filename']}\n")
        f.write(f"  Frame: {int(row['frame_index'])}\n")
        f.write(f"  Probability: {row['probability']:.3f}\n")
        f.write(f"  Scintillator: {row['scint_corr']:.3f}\n")
        f.write(f"  Mirror: {row['mirror_corr']:.3f}\n")
        f.write("\n")

print(f"\n Summary saved: {summary_path}")

