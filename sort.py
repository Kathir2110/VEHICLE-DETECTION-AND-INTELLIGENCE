import numpy as np
from filterpy.kalman import KalmanFilter


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two arrays of boxes.
    Both inputs must be shape (N,4)
    """
    bb_test = np.expand_dims(bb_test, 1)  # (N,1,4)
    bb_gt = np.expand_dims(bb_gt, 0)      # (1,M,4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * \
                (bb_test[..., 3] - bb_test[..., 1])

    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * \
              (bb_gt[..., 3] - bb_gt[..., 1])

    iou = wh / (area_test + area_gt - wh + 1e-6)
    return iou


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / (w + 1e-6)
    return np.array([x[0] - w / 2.,
                     x[1] - h / 2.,
                     x[0] + w / 2.,
                     x[1] + h / 2.]).reshape((1, 4))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)


class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 5))):

        results = []

        # Predict existing tracker positions
        predicted_boxes = []
        for tracker in self.trackers:
            pos = tracker.predict()[0]
            predicted_boxes.append(pos)

        predicted_boxes = np.array(predicted_boxes)

        used_trackers = set()

        for det in dets:
            det_box = det[:4].reshape(1, 4)

            best_iou = 0
            best_tracker_idx = -1

            for i, trk_box in enumerate(predicted_boxes):
                if i in used_trackers:
                    continue

                trk_box = trk_box.reshape(1, 4)
                iou = iou_batch(det_box, trk_box)[0][0]

                if iou > best_iou:
                    best_iou = iou
                    best_tracker_idx = i

            if best_iou > self.iou_threshold:
                tracker = self.trackers[best_tracker_idx]
                tracker.update(det[:4])
                used_trackers.add(best_tracker_idx)
                results.append(np.append(det[:4], tracker.id))
            else:
                new_tracker = KalmanBoxTracker(det[:4])
                self.trackers.append(new_tracker)
                results.append(np.append(det[:4], new_tracker.id))

        return np.array(results)