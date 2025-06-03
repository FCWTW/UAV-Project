import socket, pickle
import struct
import cv2
import numpy as np
import os
from utils.neuronpilot import neuronrt
import time, argparse
import logging
from typing import Optional, Tuple, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = '0.0.0.0'
PORT = 5001
SOCKET_TIMEOUT = 30.0

def receive_image(conn: socket.socket) -> Optional[np.ndarray]:
    """
    接收圖片數據，加入錯誤處理和超時機制
    """
    try:
        # 設置超時
        conn.settimeout(SOCKET_TIMEOUT)
        
        # 先接收 4 bytes，表示圖片大小
        size_data = conn.recv(4)
        if len(size_data) != 4:
            logger.warning("Failed to receive complete size data")
            return None
            
        data_len = struct.unpack('>I', size_data)[0]
        
        # 檢查數據大小是否合理 (最大50MB)
        if data_len > 50 * 1024 * 1024:
            logger.warning(f"Image size too large: {data_len} bytes")
            return None
            
        logger.debug(f"Expecting {data_len} bytes of image data")
        
        data = b''
        while len(data) < data_len:
            remaining = data_len - len(data)
            chunk_size = min(4096, remaining)
            packet = conn.recv(chunk_size)
            if not packet:
                logger.warning("Connection closed while receiving image data")
                break
            data += packet
            
        if len(data) != data_len:
            logger.warning(f"Incomplete image data: received {len(data)}, expected {data_len}")
            return None
        
        # === 添加調試信息 ===
        # 檢查原始數據的格式
        logger.info(f"Received raw data size: {len(data)} bytes")
        
        # 檢查文件頭以確定圖片格式
        if len(data) >= 4:
            header = data[:4]
            if header[:2] == b'\xff\xd8':
                logger.info("Detected JPEG format")
            elif header == b'\x89PNG':
                logger.info("Detected PNG format")
            elif header[:2] == b'BM':
                logger.info("Detected BMP format")
            else:
                logger.info(f"Unknown format, header: {header.hex()}")
            
        # 解碼圖片
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to decode image")
            # 嘗試不同的解碼方式
            logger.info("Trying alternative decoding methods...")
            
            # 嘗試直接讀取為numpy array（如果是原始像素數據）
            try:
                # 假設可能是原始RGB數據，嘗試不同的形狀
                total_pixels = len(data) // 3
                possible_shapes = []
                
                # 常見的圖片尺寸
                for width in [640, 480, 320, 1280, 1920]:
                    if total_pixels % width == 0:
                        height = total_pixels // width
                        possible_shapes.append((height, width, 3))
                
                logger.info(f"Possible shapes for raw data: {possible_shapes}")
                
                if possible_shapes:
                    # 嘗試第一個可能的形狀
                    shape = possible_shapes[0]
                    img = np.frombuffer(data, dtype=np.uint8).reshape(shape)
                    logger.info(f"Successfully reshaped as raw data: {shape}")
                
            except Exception as reshape_error:
                logger.error(f"Failed to reshape as raw data: {reshape_error}")
                
            return None
        
        # logger.info(f"Decoded image shape: {img.shape}")
        # logger.info(f"Decoded image dtype: {img.dtype}")
        # logger.info(f"Decoded image min/max values: {img.min()}/{img.max()}")
        
        # 檢查圖片是否有異常值
        if img.max() <= 1.0:
            logger.warning("Image values seem to be normalized (0-1), might need scaling")
        elif img.max() > 255:
            logger.warning("Image values exceed 255, might be in wrong format")        
        return img
        
    except socket.timeout:
        logger.warning("Socket timeout while receiving image")
        return None
    except Exception as e:
        logger.error(f"Error receiving image: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def send_result(conn: socket.socket, result_data: Dict[str, Any]) -> bool:
    """
    發送結果數據，加入錯誤處理
    """
    try:
        payload = pickle.dumps(result_data)
        
        # 檢查payload大小
        if len(payload) > 100 * 1024 * 1024:  # 100MB限制
            logger.warning(f"Result payload too large: {len(payload)} bytes")
            return False
            
        # 發送大小和數據
        conn.sendall(struct.pack('>I', len(payload)) + payload)
        logger.debug(f"Sent result: {len(payload)} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Error sending result: {e}")
        return False

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
        
        boxes = []
        # Draw each bounding box
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_id = det[i]
            cls_id = int(cls_id) + 1

            boxes.append({
                "cls": int(cls_id),
                "conf": float(conf),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)]
            })
            
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
    return image, boxes

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--tflite_model", type=str, required=True, help="Path to .tflite")
    parser.add_argument("-d", "--device", type=str, default='mdla3.0', choices = ['mdla3.0', 'mdla2.0', 'vpu'], help="Device name for acceleration")
    args = parser.parse_args()

    if not os.path.exists(args.tflite_model):
        raise FileNotFoundError(f"Model file doesn't exist: {args.tflite_model}")
    
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./bin', exist_ok=True)
    logging.getLogger().setLevel(logging.DEBUG)

    # 初始化 neuronrt.Interpreter
    logger.info(f"Loading model: {args.tflite_model}")
    interpreter = neuronrt.Interpreter(model_path=args.tflite_model, device=args.device)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 獲取輸入形狀
    if len(input_details) > 0 and len(input_details[0]['shape']) >= 3:
        input_shape = tuple(input_details[0]['shape'][1:3])  # [batch, height, width, channels]
        if input_shape[0] == 0 or input_shape[1] == 0:
            input_shape = (640, 640)
    else:
        input_shape = (640, 640)
    
    logger.info(f"Input shape: {input_shape}")

    # 啟動伺服器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允許重用地址
        s.bind((HOST, PORT))
        s.listen(1)
        logger.info(f"Server listening on {HOST}:{PORT}")
        
        while True:  # 外層循環處理多個連接
            try:
                conn, addr = s.accept()
                logger.info(f"Connected by {addr}")
                
                with conn:
                    number=0
                    while True:  # 內層循環處理單個連接的多個請求
                        try:
                            start_time = time.time()
                            
                            # 接收圖片
                            img = receive_image(conn)
                            if img is None:
                                logger.info("No image received. Closing connection.")
                                continue

                            # 預處理
                            input_data, transform_info = preprocess_image(img, input_shape)

                            # 確保dtype正確
                            input_dtype = input_details[0]['dtype']
                            if input_data.dtype != input_dtype:
                                input_data = input_data.astype(input_dtype)

                            # 推論
                            interpreter.set_tensor(input_details[0]['index'], input_data)
                            interpreter.invoke()
                            output_data = interpreter.get_tensor(output_details[0]['index'])

                            # 後處理
                            output_data = output_data.transpose(0, 2, 1)
                            results = postprocess(output_data, transform_info, 
                                                conf_thres=0.25, 
                                                iou_thres=0.45)

                            # 視覺化
                            vis_img, boxes = visualizer(img, results, COCO_CLASSES, input_shape)
                            cv2.imwrite(f'/home/ubuntu/MTK-genio-demo/tmp/vis_{number}.jpg', vis_img)
                            number+=1

                            # 準備結果
                            result_data = {
                                "image": vis_img,
                                "boxes": boxes,
                                "processing_time": time.time() - start_time
                            }
                            
                            # 發送結果
                            if not send_result(conn, result_data):
                                break
                                
                            logger.info(f"Processing time: {result_data['processing_time']:.3f}s, "
                                      f"Detected objects: {len(boxes)}")

                        except Exception as e:
                            logger.error(f"Error processing request: {e}")
                            break
                            
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
                break
            except Exception as e:
                logger.error(f"Server error: {e}")
                continue