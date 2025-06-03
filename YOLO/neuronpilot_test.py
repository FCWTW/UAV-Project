import os
import numpy as np
import cv2
from utils.neuronpilot import neuronrt
import time, argparse

'''
Test single image with NeuroPilot SDK

Source:
https://github.com/R300-AI/ITRI-AI-Hub
https://github.com/R300-AI/MTK-genio-demo

Tool:
https://netron.app/
'''

class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Calculate the scaled ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Return Transformation Information
        transform_info = {
            'ratio': ratio,
            'pad': (left, top),
            'original_shape': shape,
            'new_shape': new_shape
        }
        
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            labels["transform_info"] = transform_info
            return labels
        else:
            return img, transform_info

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

def preprocess_image(image, input_shape=(640, 640)):
    """
    Preparing input images for NeuroPilot YOLO models
    """
    letterbox = LetterBox(new_shape=input_shape)
    resized_image, transform_info = letterbox(image=image)
    input_data = resized_image.astype(np.float32)
    
    # Convert BGR to RGB
    input_data = input_data[..., ::-1]
    
    # Normalization
    input_data /= 255.0
    
    # Ensure continuous data storage
    input_data = np.ascontiguousarray(input_data)
    
    # Adding batch dimensions
    input_data = np.expand_dims(input_data, axis=0)
    
    print(f"Preprocessed input shape: {input_data.shape}, type: {input_data.dtype}")
    return input_data, transform_info

def postprocess(preds, transform_info, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300, nc=80):
    """
    YOLO output post-processing functions for the NeuroPilot SDK
    
    preds: model output, shape (1, 8400, 84)
    transform_info: pre-processed transform information
    """
    results = []
    
    for i, pred in enumerate(preds):
        # Calculate the maximum class confidence for each box
        class_scores = np.max(pred[:, 5:5+nc], axis=1)
        class_ids = np.argmax(pred[:, 5:5+nc], axis=1)
        
        # Filter out boxes with confidence levels greater than the threshold
        conf_mask = class_scores > conf_thres
        filtered_pred = pred[conf_mask]
        filtered_scores = class_scores[conf_mask]
        filtered_class_ids = class_ids[conf_mask]
        
        print(f"Confidence threshold: {conf_thres}")
        print(f"The range of class confidence: {np.min(class_scores)} - {np.max(class_scores)}")
        print(f"Number of remaining boxes after filtering: {filtered_pred.shape[0]}")
        
        if filtered_pred.shape[0] == 0:  # Nothing detected
            results.append(None)
            continue
            
        # Convert coordinates to xyxy format (still normalized coordinates)
        boxes = filtered_pred[:, :4].copy()
        boxes = xywh2xyxy(boxes)
        
        # Converts normalized coordinates to absolute coordinates relative to the model input image
        input_h, input_w = transform_info['new_shape']
        boxes[:, [0, 2]] *= input_w  # x
        boxes[:, [1, 3]] *= input_h  # y
        
        # Remove letterbox padding and convert back to the original zoomed image coordinates
        pad_left, pad_top = transform_info['pad']
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        
        # Zoom back to original image size
        ratio_w, ratio_h = transform_info['ratio']
        boxes[:, [0, 2]] /= ratio_w  # x
        boxes[:, [1, 3]] /= ratio_h  # y
        
        # Ensure the coordinates are within the original image
        orig_h, orig_w = transform_info['original_shape']
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        # Combination result: [x1, y1, x2, y2, conf, class_id]
        det = np.column_stack((boxes, filtered_scores, filtered_class_ids))
        
        # Non Max Suppression
        if det.shape[0] > 1:
            boxes_for_nms, scores = det[:, :4], det[:, 4]
            nms_indices = non_max_suppression(boxes_for_nms, scores, iou_thres)
            if isinstance(nms_indices, list):
                nms_indices = np.array(nms_indices)
            det = det[nms_indices[:max_det]]
        
        print(f"Number of remaining boxes after NMS: {det.shape[0]}")
        results.append(det)
    return results

def generate_colors(num_classes):
    """
    Generate fixed color palettes
    """
    colors = []
    golden_angle = 137.508
    
    for i in range(num_classes):
        hue = (i * golden_angle) % 360
        
        # Alternate between different combinations of saturation and luminance to increase variation
        if i % 4 == 0:
            saturation, value = 255, 255
        elif i % 4 == 1:
            saturation, value = 200, 255
        elif i % 4 == 2:
            saturation, value = 255, 200
        else:
            saturation, value = 180, 255
        
        # Convert HSV to BGR
        hsv = np.uint8([[[int(hue//2), saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors[:num_classes]

def visualizer(image, results, labels, input_shape=(640, 640)):
    """
    Plot test results on original image using different colors for each class
    """
    if results is None or len(results) == 0 or results[0] is None:
        return image

    h, w = image.shape[:2]
    colors = generate_colors(len(labels))

    for det in results:
        if det is None:
            continue
        
        # Draw each bounding box
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_id = det[i]
            cls_id = int(cls_id) + 1
            
            # Ensure coordinates are integers and within the image
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # Check if the coordinates are valid
            if x1 >= x2 or y1 >= y2:
                continue

            color = colors[cls_id % len(colors)]
            cls_name = labels[cls_id] if cls_id < len(labels) else f"class_{cls_id}"
            print(f'Detected: {cls_name}, Confidence: {conf:.2f}, Coordinates: ({x1}, {y1}, {x2}, {y2}), Color: {color}')

            cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
            label_text = f'{cls_name} {conf:.2f}'
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
            brightness = sum(color) / 3
            text_color = (255, 255, 255) if brightness < 127 else (0, 0, 0)
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)  
    return image

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y

def non_max_suppression(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def analyze_yolo_output(output_data):
    """Analyzing the structure of YOLO output"""
    print(f"Output shape: {output_data.shape}")
    
    for i in range(0, min(84, output_data.shape[2]), 5):
        segment = output_data[0, :, i:i+5]
        print(f"The value range of the output index {i}~{i+4}: {np.min(segment)} ~ {np.max(segment)}")
        print(f"The percentage of non-zero values for output indexes {i}~{i+4}: {np.mean(segment != 0) * 100:.2f}%")

    for i in range(min(5, output_data.shape[1])):
        box_data = output_data[0, i]
        max_class_idx = np.argmax(box_data[5:])
        max_class_conf = box_data[5 + max_class_idx]
        print(f"box {i}: [{box_data[0]:.4f}, {box_data[1]:.4f}, {box_data[2]:.4f}, {box_data[3]:.4f}], " 
              f"confidence: {box_data[4]:.4f}, highest class: {max_class_idx}, class confidence: {max_class_conf:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--tflite_model", type=str, required=True, help="Path to .tflite")
    parser.add_argument("-d", "--device", type=str, default='mdla3.0', choices = ['mdla3.0', 'mdla2.0', 'vpu'], help="Device name for acceleration")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to test image")
    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.image):
        raise ValueError(f"Image file doesn't exist: {args.image}")
    if not os.path.exists(args.tflite_model):
        raise FileNotFoundError(f"Model file doesn't exist: {args.tflite_model}")
    
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./bin', exist_ok=True)

    # Initialize neuronrt.Interpreter
    interpreter = neuronrt.Interpreter(model_path=args.tflite_model, device=args.device)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Pre-process image
    if len(input_details) > 0 and len(input_details[0]['shape']) >= 3:
        input_shape = tuple(input_details[0]['shape'][1:3])  # [batch, height, width, channels]
        if input_shape[0] == 0 or input_shape[1] == 0:
            input_shape = (640, 640)
    else:
        input_shape = (640, 640)

    image = cv2.imread(args.image)
    input_data, transform_info = preprocess_image(image, input_shape)
    
    # Ensure the input data type matches the desired type
    input_dtype = input_details[0]['dtype']
    if input_data.dtype != input_dtype:
        print(f"Convert input data type from {input_data.dtype} to {input_dtype}.")
        input_data = input_data.astype(input_dtype)
    
    # Setting the Input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Inference
    start = time.time()
    interpreter.invoke()
    inference_time = time.time() - start
    print(f"Reasoning complete, time taken: {inference_time:.4f} seconds")

    # Post-process
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # analyze_yolo_output(output_data)
    output_data = output_data.transpose(0, 2, 1)    # Should be (1, 8400, 84)
    results = postprocess(output_data, transform_info, conf_thres=0.25, iou_thres=0.45, nc=80)
    
    # Plotting result
    output_image = visualizer(image, results, COCO_CLASSES)
 
    output_path = args.image.rsplit('.', 1)[0] + "_detection.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"Result saved to: {output_path}")