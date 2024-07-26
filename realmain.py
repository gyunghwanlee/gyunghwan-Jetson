# 필요한 라이브러리 임포트
import pyrealsense2 as rs
import cv2
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import numpy as np

# RealSense Depth Camera 초기화 함수
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    return pipeline, profile

# 깊이 정보 추출 함수
def get_depth_info(depth_frame, bbox):
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_center = int((bbox[1] + bbox[3]) / 2)
    depth = depth_frame.get_distance(x_center, y_center)
    return x_center, y_center, depth

# YOLOv8 기반 객체 탐지 및 깊이 정보 추출 클래스
class DepthDetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0, depth_frame = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            for det in results:
                if len(det) == 0:
                    continue
                annotator = self.get_annotator(im0)
                for *bbox, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.model.names[c] if self.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                    annotator.box_label(bbox, label, color=colors(c, True))
                    x_center, y_center, depth = get_depth_info(depth_frame, bbox)
                    log_string += f'{label} at (x: {x_center}, y: {y_center}, depth: {depth:.2f}m)\n'
                im0 = annotator.result()

        print(log_string)

# 메인 함수
def main():
    # RealSense 초기화
    pipeline, profile = init_realsense()

    try:
        while True:
            # RealSense 프레임 가져오기
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # YOLOv8 객체 탐지
            predictor = DepthDetectionPredictor()
            predictor.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            predictor.args = argparse.Namespace(line_thickness=2, conf=0.25, iou=0.45, agnostic_nms=False, max_det=1000, classes=None)
            predictor.source_type = type('source_type', (), {})()
            predictor.source_type.webcam = True
            predictor.source_type.from_img = False

            preds = predictor(color_image)
            predictor.write_results(0, preds, (None, color_image, color_image, depth_frame))

            # 결과 표시
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
