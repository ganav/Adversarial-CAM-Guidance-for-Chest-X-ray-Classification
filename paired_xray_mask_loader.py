# paired_xray_mask_loader.py  (only the pairing functions need to be this way)
from pathlib import Path
import numpy as np
import cv2
from keras.utils import Sequence

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def build_pairs_data_proposed(data_root, split):
    """
    data_root: path to data_proposed
    split: 'train' or 'validation' or 'test'

    Structure:
      data_root/split/ClassName/images/*.png
      data_root/split/ClassName/masks/*.png

    Pairing: same filename under images/ and masks/
    Label: ClassName folder
    """
    data_root = Path(data_root)
    split_dir = data_root / split
    if not split_dir.exists():
        raise ValueError("Split folder not found: {}".format(split_dir))

    # classes are folders inside split
    class_names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if len(class_names) == 0:
        raise ValueError("No class folders found in: {}".format(split_dir))

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    pairs = []

    for c in class_names:
        img_dir = split_dir / c / "images"
        msk_dir = split_dir / c / "masks"

        if not img_dir.exists():
            raise ValueError("Missing images folder: {}".format(img_dir))
        if not msk_dir.exists():
            raise ValueError("Missing masks folder: {}".format(msk_dir))

        for img_path in img_dir.iterdir():
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            mask_path = msk_dir / img_path.name  # same filename
            if not mask_path.exists():
                # skip if mask missing
                continue

            pairs.append((str(img_path), str(mask_path), class_to_idx[c]))

    if len(pairs) == 0:
        raise ValueError("No paired (image,mask) found in {} (check filenames).".format(split_dir))

    return pairs, class_to_idx




def _load_grayscale(path, target_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read image: {}".format(path))
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (H,W,1)
    return img

def _load_mask(path, target_size, thresh=0.5):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Failed to read mask: {}".format(path))
    m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
    m = m.astype("float32") / 255.0
    m = (m >= thresh).astype("float32")
    m = np.expand_dims(m, axis=-1)  # (H,W,1)
    return m

class PairedXrayMaskSequence(Sequence):
    """
    Returns x_batch, y_batch, m_batch for training.
    """
    def __init__(self, pairs, img_size=299, batch_size=4, shuffle=True, x_channels=1, seed=123):
        self.pairs = list(pairs)
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.x_channels = int(x_channels)
        self.rng = np.random.RandomState(seed)
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pairs) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.pairs))
        batch_ids = self.indexes[start:end]

        H = self.img_size
        W = self.img_size
        target_size = (W, H)

        xs, ys, ms = [], [], []
        for j in batch_ids:
            img_path, mask_path, y = self.pairs[j]
            x = _load_grayscale(img_path, target_size=target_size)  # (H,W,1)
            m = _load_mask(mask_path, target_size=target_size)      # (H,W,1)

            if self.x_channels == 3:
                x = np.concatenate([x, x, x], axis=-1)              # (H,W,3)

            xs.append(x); ys.append(y); ms.append(m)

        x_batch = np.stack(xs, axis=0).astype("float32")
        y_batch = np.array(ys, dtype="int64")
        m_batch = np.stack(ms, axis=0).astype("float32")
        return x_batch, y_batch, m_batch

