import cv2
import numpy as np
from torch.utils.data import Dataset

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize image to a 32-pixel-multiple rectangle
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)

class WebcamDataset(Dataset):
    def __init__(self, source=0, img_size=640, stride=32):
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), f"Failed to open camera {source}"
        self.img_size = img_size
        self.stride = stride
        self.count = 0


    def __len__(self):
        return int(1E6)  # Large integer to simulate infinite stream

    def __getitem__(self, index):

        ret_val, frame = self.cap.read()
        attempts = 0
        while not ret_val and attempts < 5:
            ret_val, frame = self.cap.read()
            attempts += 1
        if not ret_val:
            raise RuntimeError(f"Failed to grab frame {index} from camera")

        frame = cv2.flip(frame, 1)  # flip left-right
        img, _, _ = letterbox(frame, (self.img_size, self.img_size), stride=self.stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img_path = "tmp"
        self.count += 1
        return img_path, img, frame, self.count

    def close(self):
        self.cap.release()
