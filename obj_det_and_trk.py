import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, 
                            check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                            scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

#---------------Object Tracking---------------
import skimage
from sort import *


#-----------Object Blurring-------------------
blurratio = 40

#-----------pandas-------------
import pandas as pd

#-----------self-made----------
from Lib.transform_coordinate import Transform_Coordinate


"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, obj_cls, identities=None, categories=None, 
                names=None, color_box=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)
        obj = int(obj_cls[i])

        # Green=0, Orange=1, Red=2
        if obj == 0:
            color = (0, 255, 0)
        elif obj == 1:
            color = (0, 128, 255)
        else:
            color = (0, 0, 255)

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
    return img
#..............................................................................

@torch.no_grad()
class YOLOv5_Tracking:
    def __init__(
        self,
        weights=ROOT / 'yolov5n.pt',
        source=ROOT / 'yolov5/data/images', 
        data=ROOT / 'yolov5/data/coco128.yaml',  
        imgsz=(640, 640),conf_thres=0.25,iou_thres=0.45,  
        max_det=1000, device='cpu',  view_img=False,  
        save_txt=True, save_conf=False, save_crop=False, 
        nosave=False, classes=None,  agnostic_nms=False,  
        augment=False, visualize=False,  update=False,  
        project=ROOT / 'runs/detect',  name='exp',  
        exist_ok=False, line_thickness=2,hide_labels=False,  
        hide_conf=False,half=False,dnn=False,display_labels=False,
        blur_obj=False,color_box = False,
        interval=5,
        depth_dir=''
    ):
        self.weights = weights
        self.source = source
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.display_labels = display_labels
        self.blur_obj = blur_obj
        self.color_box = color_box
        self.interval=interval
        self.depth_dir=depth_dir
    
    def run(self):

        save_img = not self.nosave and not self.source.endswith('.txt') 
    
        #.... Initialize SORT .... 
        sort_max_age = 10
        sort_min_hits = 8
        sort_iou_thresh = 0.2
        sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh) 
        track_color_id = 0
        #......................... 

        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  

        set_logging()
        device = select_device(self.device)
        self.half &= device.type != 'cpu'

        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(self.imgsz, s=stride)  

        self.half &= (pt or jit or onnx or engine) and device.type != 'cpu'  
        if pt or jit:
            model.model.half() if self.half else model.model.float()

        if webcam:
            cudnn.benchmark = True  
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset) 
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1 
        vid_path, vid_writer = [None] * bs, [None] * bs
        
        t0 = time.time()
        
        dt, seen = [0.0, 0.0, 0.0], 0

        for path, im, im0s, vid_cap, s in dataset:
        
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            distance = 0
            for i, det in enumerate(pred):  # per image

                distance=seen*self.interval
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                imc = im0.copy() if self.save_crop else im0
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    # # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    #     if self.blur_obj:
                    #         crop_obj = im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                    #         blur = cv2.blur(crop_obj,(blurratio,blurratio))
                    #         im0[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
                    #     else:
                    #         continue

                    #..................USE TRACK FUNCTION....................
                    #pass an empty array to sort
                    dets_to_sort = np.empty((0,6))
                    
                    # NOTE: We send in detected object class too
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, 
                                                np.array([x1, y1, x2, y2, 
                                                            conf, detclass])))

                    
                    # Run SORT
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()                    

                    # #loop over tracks
                    # for track in tracks:
                    #     if color_box:
                    #         color = compute_color_for_labels(track_color_id)
                    #         [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                    #                 (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                    #                 color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
                    #                 if i < len(track.centroidarr)-1 ] 
                    #         track_color_id = track_color_id+1
                    #     else:
                    #         [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                    #                 (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                    #                 (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                    #                 if i < len(track.centroidarr)-1 ] 
                    
                    # draw boxes for visualization
                    if len(tracked_dets)>0:
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        obj_cls = tracked_dets[:,4:5]
                        draw_boxes(im0, bbox_xyxy, obj_cls, identities, categories, names, self.color_box)
 
                        for i in range(len(tracked_dets)):
                            cx = int((tracked_dets[i][0] + tracked_dets[i][2])/2)
                            cy = int((tracked_dets[i][1] + tracked_dets[i][3])/2)
                            w_ = int(tracked_dets[i][2] - tracked_dets[i][0])
                            h_ = int(tracked_dets[i][3] - tracked_dets[i][1])
                            obj_cls_ = int(tracked_dets[i][4])
                            id_ = int(tracked_dets[i][8])

                            df = pd.read_csv((self.depth_dir+'depth'+p.name.replace('.jpg', '')).replace('color_img', '')+'.csv', header=None)
                            cz = df[cx][cy]*0.1

                            # print(distance)
                            world_coordinate = world.transformation_w([[cx, cy, cz]])
                            world_coordinate = '('+str(round(world_coordinate[0][0], 1))+' '+str(abs(round(world_coordinate[0][1]-distance, 1)))+' '+str(round(world_coordinate[0][2], 1))+')'

                            text = str(obj_cls_) + ','+ str(id_) + ',' +str(cx) + ',' + str(cy) + ',' + str(w_) + ',' + str(h_) + ',' + world_coordinate +  '\n'                     
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(text)
                        f.close()

                else:
                    with open(f'{txt_path}.txt', 'a') as f:
                            f.write('')
                    f.close()


                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1) 
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path: 
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  
                            if vid_cap: 
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
            print("Processing")
        print("Success")

        if self.update:
            strip_optimizer(self.weights)
        
        if vid_cap:
            vid_cap.release()

depth_dir = './depthmap/'
robot_pos = np.array([0.0, 0.0, 0.0]) # ロボット座標系（基準となる座標系）固定
camera_w_pos = np.array([-15.25, 106.0, -4.0]) # 全体カメラ座標系　固定
camera_w_rotation_matlix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #全体カメラの回転行列
R_w = np.array([618.6954956054688, 618.8485717773438, 320.0, 240.0])




# intervalの単位は[cm]で，ロボットの進む間隔．進む向きによって正負が異なる．
yolov5 = YOLOv5_Tracking(weights='./best-7-16-2.pt', source='./imgs2', conf_thres=0.80, interval=5, depth_dir=depth_dir)
world = Transform_Coordinate(robot_pos, camera_w_pos, camera_w_rotation_matlix, R_w)
# yolov5 = YOLOv5_Tracking(weights='./best-tom.pt', source='tom', conf_thres=0.60, save_txt=True)
yolov5.run()