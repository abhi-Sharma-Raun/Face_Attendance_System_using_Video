from insightface.app import FaceAnalysis
import cv2
import numpy as np
import torch
import traceback
from .config import settings



is_gpu = torch.cuda.is_available()
providers = None
ctx_id = None
if is_gpu:
  ctx_id = 0
  providers=["CUDAExecutionProvider"]
else:
  ctx_id = -1
  providers=["CPUExecutionProvider"]

print("Imported utils module")

_det_rec_model = None
_det_model = None
_clahe_instance = None
def get_clahe():
    global _clahe_instance
    if _clahe_instance is None:
        _clahe_instance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe_instance

def get_det_rec_model():
  global _det_rec_model
  if _det_rec_model is None:
    _det_rec_model = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"], root="/models")
    _det_rec_model.prepare(ctx_id=ctx_id, det_size=(320, 256), det_thresh=0.5)
  return _det_rec_model

def get_det_model():
  global _det_model
  if _det_model is None:
    _det_model = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection"], root="/models")
    _det_model.prepare(ctx_id=ctx_id, det_size=(320, 256), det_thresh=0.5)
  return _det_model
    



def get_face_register_embedding(ud_raw_prof_bytes):
  det_rec_model = get_det_rec_model()
  try:
    ud_profile = np.frombuffer(ud_raw_prof_bytes, dtype=np.uint8)
    profile = cv2.imdecode(ud_profile, cv2.IMREAD_COLOR)
  except:
    raise Exception("Unable video bytes")
  try:
    faces = det_rec_model.get(profile)
  except:
    raise Exception("Could not extract recognition embeddings")
  if len(faces) == 0:
    return None
  elif len(faces) > 1:
    return None
  else:
    return faces[0].embedding/np.linalg.norm(faces[0].embedding)

    
def lighting_check(cropped_face) -> float:  #try to do it for entire frame instead of doing it for each face one by one

  tot_pixels = cropped_face.size
  glare_pixels = np.sum(cropped_face>=250)
  shadow_pixels = np.sum(cropped_face<=15)
  glare_ratio = glare_pixels/tot_pixels
  shadow_ratio = shadow_pixels/tot_pixels
  
  if glare_ratio > 0.15 or shadow_ratio > 0.15:
    return 0.01
  else:
    return 1.0
  


def evaluate_frame(frame) -> tuple[int, float]:
  
  det_model=None
  tot_face_detected = 0
  frame_score = 0.0
  try:
    det_model = get_det_model()
  except Exception as e:
    traceback.print_exc()
    raise Exception(f"Could not initialize model: {e}")
  
  light_score = lighting_check(frame)
  if light_score == 0.01:
    return tot_face_detected
  else:
    pass
  
  try:
    faces = det_model.get(frame)
  except Exception as e:
    print(e)
    raise Exception("Could not extract detection embeddings")

  if len(faces)>0:
    tot_face_detected = np.sum([eval_face(face, frame) for face in faces])
    
  return tot_face_detected


def select_nbd_frame(point_frame: int, vid_object: cv2.VideoCapture, tot_frames: int) -> np.array:
    
  start_frame = max(1, point_frame - settings.num_eval_nbd_frames)
  end_frame = min(tot_frames, point_frame + settings.num_eval_nbd_frames)
  frames = []
  
  for i in range(start_frame, end_frame + 1, 2):
    vid_object.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
    ret, img = vid_object.read()
    if ret == True:
      img = cv2.resize(img, (320, 256))
      frames.append(img)
  
  tot_face_per_frame=[]
  frame_score_per_frame=[]
  tot_face_per_frame = [evaluate_frame(frame) for frame in frames]
  best_frame_idx = np.argmax(tot_face_per_frame)
  
  return frames[best_frame_idx], tot_face_per_frame[best_frame_idx]

def eval_face(face, frame) -> bool:
  
  frame_score = 0.0
  is_face = False
  
  confidence = face.det_score
  bbox = face.bbox.astype(int)
  x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
  face_crop = frame[y1:y2, x1:x2]
  if confidence<0.6 or face_crop.size <= settings.face_size_threshold:
    is_face = False
    return is_face
  '''
  hsv_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)  # laplacian edge detection to check if the face is clearly visible
  h, s, v = cv2.split(hsv_img)
  v = clahe.apply(v)
  equalized_gray_img = cv2.cvtColor(np.dstack((h,s,v)), cv2.COLOR_HSV2BGR)
  equalized_gray_img = cv2.cvtColor(equalized_gray_img, cv2.COLOR_BGR2GRAY)
  sharpness = cv2.Laplacian(equalized_gray_img, cv2.CV_64F).var()
  '''
  
  gray = cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
  sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
  
  #Pose Alignment
  landmarks = face.kps
  left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
  dist_left = np.linalg.norm(nose - left_eye)
  dist_right = np.linalg.norm(nose - right_eye)
  pose_symmetry = min(dist_left, dist_right) / max(dist_left, dist_right, 1e-5)
  
  if sharpness < settings.sharpness_threshold or pose_symmetry < settings.pose_symmetry_threshold:
    frame_score += 0.0
    is_face = False
  else:
    is_face = True
    frame_score = np.sum(np.log(np.array([confidence, sharpness/1000, pose_symmetry])))
  
  return  is_face