import os
import time
import numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv

def compare_predictions(pt_result, engine_result, iou_threshold=0.5, conf_threshold=0.01):
    """Compare predictions between two models"""
    if pt_result[0].boxes is None or engine_result[0].boxes is None:
        return {
            'pt_detections': 0 if pt_result[0].boxes is None else len(pt_result[0].boxes),
            'engine_detections': 0 if engine_result[0].boxes is None else len(engine_result[0].boxes),
            'matched_detections': 0,
            'precision_diff': 0,
            'confidence_diff': 0
        }
    
    pt_boxes = pt_result[0].boxes
    engine_boxes = engine_result[0].boxes
    
    # Filter by confidence threshold
    pt_valid = pt_boxes.conf > conf_threshold
    engine_valid = engine_boxes.conf > conf_threshold
    
    pt_boxes_filtered = pt_boxes[pt_valid] if pt_valid.any() else None
    engine_boxes_filtered = engine_boxes[engine_valid] if engine_valid.any() else None
    
    if pt_boxes_filtered is None or engine_boxes_filtered is None:
        return {
            'pt_detections': 0 if pt_boxes_filtered is None else len(pt_boxes_filtered),
            'engine_detections': 0 if engine_boxes_filtered is None else len(engine_boxes_filtered),
            'matched_detections': 0,
            'precision_diff': 0,
            'confidence_diff': 0
        }
    
    # Calculate IoU between all pairs of boxes
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Find matching detections
    matched_count = 0
    confidence_diffs = []
    
    for i, pt_box in enumerate(pt_boxes_filtered.xyxy):
        pt_conf = pt_boxes_filtered.conf[i]
        pt_cls = pt_boxes_filtered.cls[i]
        
        for j, engine_box in enumerate(engine_boxes_filtered.xyxy):
            engine_conf = engine_boxes_filtered.conf[j]
            engine_cls = engine_boxes_filtered.cls[j]
            
            # Check if same class and IoU > threshold
            if pt_cls == engine_cls:
                iou = calculate_iou(pt_box.cpu().numpy(), engine_box.cpu().numpy())
                if iou > iou_threshold:
                    matched_count += 1
                    confidence_diffs.append(abs(pt_conf.item() - engine_conf.item()))
                    break
    
    avg_confidence_diff = np.mean(confidence_diffs) if confidence_diffs else 0
    
    return {
        'pt_detections': len(pt_boxes_filtered),
        'engine_detections': len(engine_boxes_filtered),
        'matched_detections': matched_count,
        'precision_diff': abs(len(pt_boxes_filtered) - len(engine_boxes_filtered)),
        'confidence_diff': avg_confidence_diff
    }
# Load environment variables from .env file
load_dotenv(override=True)
# Get model directory and name from environment variables
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo12x.pt")

# Construct model paths - handle case where MODEL_NAME includes extension
if MODEL_NAME.endswith('.pt'):
    model_base_name = MODEL_NAME[:-3]  # Remove .pt extension
    pt_model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    engine_model_path = os.path.join(MODEL_DIR, f"{model_base_name}.engine")
else:
    # Legacy support for MODEL_NAME without extension
    pt_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pt")
    engine_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.engine")

# Check if engine model already exists
if os.path.exists(engine_model_path):
    print(f"Engine model already exists at {engine_model_path}, skipping export...")
else:
    print(f"Engine model not found at {engine_model_path}, exporting...")
    model = YOLO(pt_model_path)
    # model.export(
    #     format="engine",
    #     dynamic=True,  
    #     batch=8,  
    #     workspace=4,  
    #     int8=True,
    #     data="coco.yaml",  
    # )
    model.export(
        format="engine",
        half=True,  # Use half precision for faster inference
       
    )


# Load both models for comparison
print("Loading PyTorch model...")
pt_model = YOLO(pt_model_path, task="detect")

print("Loading TensorRT model...")
engine_model = YOLO(engine_model_path, task="detect")

# Test image
test_image = "https://ultralytics.com/images/bus.jpg"

# Warm up both models (first inference is usually slower)
print("Warming up models...")
pt_model.predict(test_image, verbose=False)
engine_model.predict(test_image, verbose=False)

# Number of inference runs for average timing
num_runs = 5

print(f"\nRunning inference comparison ({num_runs} runs each)...")

# Time PyTorch model and collect results
print("Testing PyTorch model...")
pt_times = []
pt_results = []
for i in range(num_runs):
    start_time = time.time()
    pt_result = pt_model.predict(test_image, verbose=True)
    end_time = time.time()
    pt_times.append(end_time - start_time)
    pt_results.append(pt_result)
    print(f"  Run {i+1}: {(end_time - start_time)*1000:.2f}ms")

# Time TensorRT model and collect results
print("\nTesting TensorRT model...")
engine_times = []
engine_results = []
for i in range(num_runs):
    start_time = time.time()
    engine_result = engine_model.predict(test_image, verbose=True)
    end_time = time.time()
    engine_times.append(end_time - start_time)
    engine_results.append(engine_result)
    print(f"  Run {i+1}: {(end_time - start_time)*1000:.2f}ms")

# Calculate timing statistics
pt_avg = sum(pt_times) / len(pt_times)
engine_avg = sum(engine_times) / len(engine_times)
speedup = pt_avg / engine_avg

# Calculate precision statistics
print("\nAnalyzing precision differences...")
precision_comparisons = []
for i in range(num_runs):
    comparison = compare_predictions(pt_results[i], engine_results[i])
    precision_comparisons.append(comparison)

# Average precision metrics
avg_pt_detections = np.mean([comp['pt_detections'] for comp in precision_comparisons])
avg_engine_detections = np.mean([comp['engine_detections'] for comp in precision_comparisons])
avg_matched_detections = np.mean([comp['matched_detections'] for comp in precision_comparisons])
avg_precision_diff = np.mean([comp['precision_diff'] for comp in precision_comparisons])
avg_confidence_diff = np.mean([comp['confidence_diff'] for comp in precision_comparisons])

# Calculate precision metrics
max_detections = max(float(avg_pt_detections), float(avg_engine_detections))
detection_accuracy = (avg_matched_detections / max_detections) * 100 if max_detections > 0 else 0

print(f"\n{'='*60}")
print("SPEED AND PRECISION COMPARISON RESULTS")
print(f"{'='*60}")
print("\n--- SPEED COMPARISON ---")
print(f"PyTorch model average:  {pt_avg*1000:.2f}ms")
print(f"TensorRT model average: {engine_avg*1000:.2f}ms")
print(f"Speedup: {speedup:.2f}x faster")
print(f"Time saved per inference: {(pt_avg - engine_avg)*1000:.2f}ms")

print("\n--- PRECISION COMPARISON ---")
print(f"PyTorch avg detections:  {avg_pt_detections:.1f}")
print(f"TensorRT avg detections: {avg_engine_detections:.1f}")
print(f"Matched detections:      {avg_matched_detections:.1f}")
print(f"Detection accuracy:      {detection_accuracy:.1f}%")
print(f"Avg detection count diff: {avg_precision_diff:.1f}")
print(f"Avg confidence diff:     {avg_confidence_diff:.3f}")

print("\n--- DETAILED RUN ANALYSIS ---")
for i, comp in enumerate(precision_comparisons):
    print(f"Run {i+1}: PT={comp['pt_detections']}, TR={comp['engine_detections']}, "
          f"Matched={comp['matched_detections']}, ConfDiff={comp['confidence_diff']:.3f}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
if speedup > 1.5 and detection_accuracy > 90:
    print("✅ TensorRT optimization is EXCELLENT - significant speedup with high precision!")
elif speedup > 1.2 and detection_accuracy > 85:
    print("✅ TensorRT optimization is GOOD - noticeable speedup with good precision!")
elif speedup > 1.0 and detection_accuracy > 80:
    print("⚠️  TensorRT optimization is OKAY - some speedup but precision concerns")
else:
    print("❌ TensorRT optimization may not be worth it - low speedup or precision issues")
print(f"{'='*60}")