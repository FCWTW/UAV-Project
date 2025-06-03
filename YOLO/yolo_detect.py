#!/usr/bin/env python3
import ros_numpy
import rospy
import time
import cv2
import cv_bridge
import message_filters
import numpy as np
import image_geometry
import struct
import socket, pickle
import sys

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

HOST = '172.18.0.1'
PORT = 5001
processing = False

# 檢查並顯示 NumPy 版本信息
rospy.loginfo(f"Python version: {sys.version}")
rospy.loginfo(f"NumPy version: {np.__version__}")

# 初始化ROS節點和YOLO模型
rospy.init_node("yolo_detector")

# 從 ROS 參數獲取處理頻率設定
target_fps = rospy.get_param('~target_fps', 1.0)  # 默認每秒1次
process_interval = 1.0 / target_fps
rospy.loginfo(f"Target processing FPS: {target_fps}, interval: {process_interval:.3f}s")

# 創建影像和點雲發布者
det_image_pub = rospy.Publisher("/yolo/detection/image", Image, queue_size=5)
# 發布整個場景的點雲
scene_pointcloud_pub = rospy.Publisher("/yolo/scene/pointcloud", PointCloud2, queue_size=5)

# CV Bridge instance
bridge = cv_bridge.CvBridge()
# Camera Model instance (for CameraInfo)
cam_model = image_geometry.PinholeCameraModel()

# Socket 連接狀態
sock = None
connection_established = False

# 頻率控制變數
last_process_time = 0.0
frame_count = 0
skip_count = 0

# 建立辨識索引
COCO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def connect_to_host():
    """建立到 host 的連接，包含重試機制"""
    global sock, connection_established
    max_retries = 5
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            if sock:
                sock.close()
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)  # 設置連接超時
            sock.connect((HOST, PORT))
            connection_established = True
            rospy.loginfo("Successfully connected to server at %s:%d", HOST, PORT)
            return True
            
        except socket.error as e:
            rospy.logwarn(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                rospy.loginfo(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指數退避
            else:
                rospy.logerr("Failed to connect after %d attempts", max_retries)
                connection_established = False
                return False
    
    return False

def safe_recv(sock, size):
    """安全的數據接收函數，確保接收完整數據"""
    data = b''
    while len(data) < size:
        try:
            packet = sock.recv(size - len(data))
            if not packet:
                raise socket.error("Connection closed by peer")
            data += packet
        except socket.timeout:
            raise socket.error("Receive timeout")
        except socket.error as e:
            raise e
    return data

def send_image_and_receive_result(rgb_image):
    """發送圖像並接收結果，包含錯誤處理"""
    global sock, connection_established
    
    try:
        # 編碼圖像 - 降低品質以減少傳輸量
        _, img_encoded = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_bytes = img_encoded.tobytes()
        
        # 發送圖像大小和數據
        sock.sendall(struct.pack('>I', len(img_bytes)))
        sock.sendall(img_bytes)
        rospy.logdebug("Image sent to host (%d bytes)", len(img_bytes))
        
        # 接收結果大小
        size_data = safe_recv(sock, 4)
        data_len = struct.unpack('>I', size_data)[0]
        
        # 接收結果數據
        result_data = safe_recv(sock, data_len)
        result = pickle.loads(result_data)
        
        rospy.logdebug("Received result from host (%d bytes)", data_len)
        return result
        
    except socket.error as e:
        rospy.logerr("Socket error during communication: %s", e)
        connection_established = False
        raise e
    except Exception as e:
        rospy.logerr("Error during image processing: %s", e)
        raise e

def callback(rgb_msg, depth_msg, depth_info_msg):
    global processing, connection_established, last_process_time, frame_count, skip_count
    
    frame_count += 1
    
    if processing:
        skip_count += 1
        return
    
    # 頻率控制：檢查是否到了處理時間
    current_time = time.time()
    if current_time - last_process_time < process_interval:
        skip_count += 1
        return  # 還沒到處理時間，跳過這一幀
    
    processing = True
    last_process_time = current_time
    
    # 每 100 幀報告一次統計信息
    if frame_count % 100 == 0:
        rospy.loginfo(f"Frame statistics: processed={frame_count-skip_count}, skipped={skip_count}, total={frame_count}")
    
    try:
        start_time = time.time()

        # 檢查連接狀態
        if not connection_established:
            rospy.logwarn("No connection to host, attempting to reconnect...")
            if not connect_to_host():
                rospy.logwarn("Cannot connect to host, skipping frame")
                return

        # 1. 解析相機參數
        cam_model.fromCameraInfo(depth_info_msg)
        fx = cam_model.fx()
        fy = cam_model.fy()
        cx = cam_model.cx()
        cy = cam_model.cy()

        # 2. 轉換 ROS 影像為 OpenCV 格式
        try:
            rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            depth_image_meters = depth_image_raw.astype(np.float32) / 1000.0
            h, w = depth_image_meters.shape
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return

        # 3. 發送圖像並接收結果
        try:
            result_data = send_image_and_receive_result(rgb_image)
            image = result_data["image"]
            boxes = result_data["boxes"]
        except Exception as e:
            rospy.logerr("Failed to process image with host: %s", e)
            # 嘗試重新連接
            if not connect_to_host():
                rospy.logwarn("Cannot reconnect, skipping frame")
            return
    
        # 4. 使用 numpy 優化的方式生成點雲
        # 先生成整個場景的點雲 (使用適當的採樣間隔)
        step = 4  # 增加採樣間隔，進一步減少計算量
        v_indices, u_indices = np.mgrid[0:h:step, 0:w:step]
        v_indices = v_indices.flatten()
        u_indices = u_indices.flatten()
        
        depths = depth_image_meters[v_indices, u_indices]
        valid_mask = (depths > 0.01) & (depths < 20.0) & (~np.isnan(depths))
        
        v_valid = v_indices[valid_mask]
        u_valid = u_indices[valid_mask]
        z_valid = depths[valid_mask]
        
        # 計算3D座標
        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy
        
        # 獲取顏色
        colors = rgb_image[v_valid, u_valid]
        
        # 轉換為列表格式
        scene_points = np.vstack((x_valid, y_valid, z_valid)).T
        scene_colors = colors[:, [2,1,0]]  # BGR轉RGB
        
        # 創建一個索引陣列，用於標記點屬於哪個物體
        point_labels = np.zeros(len(scene_points), dtype=np.int32) - 1  # -1表示不屬於任何物體
        
        # 5. 為每個物體框單獨處理
        for i, box in enumerate(boxes):
            cls_id = int(box['cls'])
            cls_name = COCO_CLASSES[cls_id]
            conf = float(box['conf'])
            xyxy = box['xyxy']
            
            # 使用優化的方式標記物體框內的點
            u_min, v_min, u_max, v_max = map(int, [max(0, xyxy[0]), max(0, xyxy[1]), 
                                              min(w-1, xyxy[2]), min(h-1, xyxy[3])])
            
            # 找出屬於該物體框的點
            in_box_mask = (u_valid >= u_min) & (u_valid <= u_max) & (v_valid >= v_min) & (v_valid <= v_max)
            if np.any(in_box_mask):
                # 找出該物體框內的最近點
                box_depths = z_valid[in_box_mask]
                min_depth = np.min(box_depths)
                
                # 標記屬於該物體的點
                point_labels[in_box_mask] = i
                
                rospy.loginfo(f"物體 {cls_name} (置信度: {conf:.2f}): 最近距離 {min_depth:.2f} 米")
        
        # 為物體點著色 (用於點雲)
        scene_colors_copy = scene_colors.copy()
        for i, box in enumerate(boxes):
            # 找出屬於該物體的點
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                # 獲取該物體的深度範圍
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)  # 避免除以零
                
                # 根據深度為物體點著色
                normalized_depths = (obj_depths - min_depth) / depth_range
                r = (255 * (1 - normalized_depths)).astype(np.uint8)
                g = np.ones_like(r) * 128
                b = (255 * normalized_depths).astype(np.uint8)
                
                # 更新顏色
                scene_colors_copy[obj_mask, 0] = r
                scene_colors_copy[obj_mask, 1] = g
                scene_colors_copy[obj_mask, 2] = b
        
        # 6. 發布整個場景的點雲
        if len(scene_points) > 0:
            # 定義 PointCloud2 字段
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1),
            ]
            
            # 創建 header
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "camera_depth_optical_frame"
            
            # 打包點和顏色
            packed_points = []
            for i in range(len(scene_points)):
                pt = scene_points[i]
                color = scene_colors_copy[i]
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                rgb_packed = (r << 16) | (g << 8) | b
                packed_points.append([pt[0], pt[1], pt[2], rgb_packed])
            
            # 創建 PointCloud2 消息
            scene_cloud_msg = pc2.create_cloud(header, fields, packed_points)
            scene_pointcloud_pub.publish(scene_cloud_msg)
            rospy.loginfo(f"發布了包含 {len(scene_points)} 個點的場景點雲。")
        
        # 7. 在原始偵測結果上添加深度點和深度信息
        for i, box in enumerate(boxes):
            # 找出屬於該物體的點
            obj_mask = (point_labels == i)
            if np.any(obj_mask):
                # 獲取該物體的深度範圍
                obj_depths = z_valid[obj_mask]
                min_depth = np.min(obj_depths)
                max_depth = np.max(obj_depths)
                depth_range = max(max_depth - min_depth, 0.1)
                
                # 獲取點的座標和深度
                obj_u = u_valid[obj_mask]
                obj_v = v_valid[obj_mask]
                normalized_depths = (obj_depths - min_depth) / depth_range
                
                # 每隔幾個點繪製一次，減少密度但保持視覺效果
                skip = 4  # 增加跳過的點數
                for j in range(0, len(obj_u), skip):
                    u, v = int(obj_u[j]), int(obj_v[j])
                    depth_val = normalized_depths[j]
                    
                    # 根據深度生成顏色 (紅色表示近，藍色表示遠)
                    r = int(255 * (1 - depth_val))
                    g = 0  # 移除綠色成分，增強紅藍對比
                    b = int(255 * depth_val)
                    
                    # 直接在偵測結果上繪製點
                    cv2.circle(image, (u, v), 2, (b, g, r), -1)  # 縮小點的大小
                
                # 添加深度信息文字
                xyxy = box['xyxy']
                u_min, v_min = map(int, [xyxy[0], xyxy[1]])
                cls_name = COCO_CLASSES[int(box['cls'])]
                depth_text = f"depth: {min_depth:.2f}m"
                
                right_top_x = u_min  
                right_top_y = v_min - 25  
                cv2.putText(image, depth_text, (right_top_x, right_top_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # 使用黃色
        
        # 8. 發布帶有深度信息的偵測結果
        det_image_pub.publish(bridge.cv2_to_imgmsg(image, encoding="bgr8"))

        process_time = time.time() - start_time
        rospy.loginfo(f"檢測與點雲處理完成。處理時間: {process_time:.3f}秒")

    except Exception as e:
        rospy.logerr(f"處理圖像時出錯: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
        # 連接可能已斷開，標記為未連接
        connection_established = False
    finally:
        processing = False

def cleanup():
    """清理資源"""
    global sock
    if sock:
        try:
            sock.close()
        except:
            pass
    rospy.loginfo("Resources cleaned up")

# 初始連接
if not connect_to_host():
    rospy.logfatal("Cannot establish initial connection to host. Exiting.")
    exit(1)

# 註冊清理函數
rospy.on_shutdown(cleanup)

# 使用 message_filters 同步 RGB、深度影像和相機參數
# 這邊要改成要接收的 rostopic
rgb_topic = "/camera/color/image_raw"
depth_topic = "/camera/depth/image_rect_raw"
depth_info_topic = "/camera/depth/camera_info"

rgb_sub = message_filters.Subscriber(rgb_topic, Image)
depth_sub = message_filters.Subscriber(depth_topic, Image)
depth_info_sub = message_filters.Subscriber(depth_info_topic, CameraInfo)

# 使用 ApproximateTimeSynchronizer 實現時間同步
ts = message_filters.ApproximateTimeSynchronizer(
    [rgb_sub, depth_sub, depth_info_sub],
    queue_size=10,
    slop=0.1
)
ts.registerCallback(callback)

rospy.loginfo(f"YOLO檢測器已啟動，處理頻率: {target_fps} FPS")
rospy.loginfo(f"訂閱: RGB: {rgb_topic}, 深度: {depth_topic}, 相機參數: {depth_info_topic}")
rospy.loginfo(f"發布標註影像到: /yolo/detection/image")
rospy.loginfo(f"發布場景點雲到: /yolo/scene/pointcloud")

rospy.spin()