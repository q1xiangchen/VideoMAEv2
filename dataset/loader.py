import io
import os

import cv2
import numpy as np
from decord import VideoReader, cpu
import h5py
from PIL import Image
import json

try:
    from petrel_client.client import Client
    petrel_backend_imported = True
except (ImportError, ModuleNotFoundError):
    petrel_backend_imported = False

def load_h5_file(dataset_path, path):
    """
    Load a file from an hdf5 dataset
    
    Args:
        dataset_path: str, path to hdf5 dataset
        path: str, path to file within hdf5 dataset
    """
    root = path.split('/')[0].replace("_", " ")
    rest = path.split('/')[1:]
    path = os.path.join(root, *rest)
    
    with h5py.File(dataset_path, 'r') as hf:
        if path.endswith('.jpg') or path.endswith('.png') or path.endswith('.gif'):
            # saved the image as raw binary, need to convert to image
            rtn = Image.open(io.BytesIO(np.array(hf[path])))
        elif path.endswith('.json'):
            # saved as a dataset string, need to convert to json dict
            rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
        elif path.endswith('.txt'):
            rtn = np.array(hf[path]).tobytes().decode('utf-8')
        elif path.endswith('.csv'):
            rtn = np.array(hf[path]).tobytes().decode('utf-8')
        elif path.endswith('.mp4'):
            rtn = np.array(hf[path])
        elif path.endswith('.avi'):
            rtn = np.array(hf[path])
        else:
            raise ValueError('Unknown file type: {}'.format(path))
        return rtn


def get_video_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None,
                     data_root: str = ""):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    if data_root:
        def _loader(video_path):
            # if _client is not None and 's3:' in video_path:
            #     video_path = io.BytesIO(_client.get(video_path))

            #NOTE: read video from h5 file
            # find the only h5 file in the directory
            dataset_path = None
            for file in os.listdir(data_root):
                if file.endswith(".h5"):
                    dataset_path = os.path.join(data_root, file)
            if dataset_path is None:
                video_path = io.BytesIO(_client.get(video_path))
            elif ".h5" in dataset_path:
                video_binary = load_h5_file(dataset_path, video_path)
                video_path = io.BytesIO(video_binary)
            else:
                pass

            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            return vr
    else:
        def _loader(video_path):
            if _client is not None and 's3:' in video_path:
                video_path = io.BytesIO(_client.get(video_path))

            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            return vr

    return _loader


# def get_video_loader(use_petrel_backend: bool = True,
#                      enable_mc: bool = True,
#                      conf_path: str = None,
#                      data_root: str = "data/ucf101_1"):
#     if petrel_backend_imported and use_petrel_backend:
#         _client = Client(conf_path=conf_path, enable_mc=enable_mc)
#     else:
#         _client = None

#     def _loader(video_path):
#         # if _client is not None and 's3:' in video_path:
#         #     video_path = io.BytesIO(_client.get(video_path))

#         #NOTE: read video from h5 file
#         # find the only h5 file in the directory
#         dataset_path = None
#         for file in os.listdir(data_root):
#             if file.endswith(".h5"):
#                 dataset_path = os.path.join(data_root, file)
#         if dataset_path is None:
#             video_path = io.BytesIO(_client.get(video_path))
#         else:
#             video_binary = load_h5_file(dataset_path, video_path)
#             video_path = io.BytesIO(video_binary)

#         vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
#         return vr

#     return _loader


def get_image_loader(use_petrel_backend: bool = True,
                     enable_mc: bool = True,
                     conf_path: str = None,
                     data_root: str = ""):
    if petrel_backend_imported and use_petrel_backend:
        _client = Client(conf_path=conf_path, enable_mc=enable_mc)
    else:
        _client = None

    if data_root:
        def _loader(frame_path):
            # if _client is not None and 's3:' in frame_path:
            #     img_bytes = _client.get(frame_path)
            # else:
            #     with open(frame_path, 'rb') as f:
            #         img_bytes = f.read()

            #NOTE: read image from h5 file
            # find the only h5 file in the directory
            dataset_path = None
            for file in os.listdir(data_root):
                if file.endswith(".h5"):
                    dataset_path = os.path.join(data_root, file)
                    break
            if dataset_path is None:
                with open(frame_path, 'rb') as f:
                    img_bytes = f.read()
            elif ".h5" in dataset_path:
                video_binary = load_h5_file(dataset_path, frame_path)
                img_bytes = io.BytesIO(video_binary)

            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img

        return _loader
    else:
        def _loader(frame_path):
            if _client is not None and 's3:' in frame_path:
                img_bytes = _client.get(frame_path)
            else:
                with open(frame_path, 'rb') as f:
                    img_bytes = f.read()

            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img

        return _loader


# def get_image_loader(use_petrel_backend: bool = True,
#                      enable_mc: bool = True,
#                      conf_path: str = None,
#                      data_root: str = "data/sthsthv2"):
#     if petrel_backend_imported and use_petrel_backend:
#         _client = Client(conf_path=conf_path, enable_mc=enable_mc)
#     else:
#         _client = None

#     def _loader(frame_path):
#         # if _client is not None and 's3:' in frame_path:
#         #     img_bytes = _client.get(frame_path)
#         # else:
#         #     with open(frame_path, 'rb') as f:
#         #         img_bytes = f.read()

#         #NOTE: read image from h5 file
#         # find the only h5 file in the directory
#         dataset_path = None
#         for file in os.listdir(data_root):
#             if file.endswith(".h5"):
#                 dataset_path = os.path.join(data_root, file)
#                 break
#         if dataset_path is None:
#             raise ValueError("No h5 file found in the directory")
        
#         video_binary = load_h5_file(dataset_path, frame_path)
#         img_bytes = io.BytesIO(video_binary)

#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
#         return img

#     return _loader
