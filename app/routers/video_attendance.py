from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from typing import List
import cv2
import shutil
import tempfile
from .. import utils, schemas
import numpy as np
from ..chroma_db_setup import collection
import time


router = APIRouter(
    prefix="/video_attendance",
    tags=["video_attendance"]
)


FACE_SIMILARITY_THRESHOLD=0.55

@router.post("", response_model = schemas.Video_AttendanceResponse)
async def mark_attendance(video: UploadFile = File(...)):
    det_rec_model = utils.get_det_rec_model()
    s_time = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(video.file, tmp)
            video_path = tmp.name
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Problem with video file")
    
    # Frame selection
    vid = cv2.VideoCapture(video_path)
    tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    k_fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    N = 3*k_fps/4
    
    best_frames = []
    tot_best_faces = 0
    target_frames = [f for f in range(int(N), tot_frames + 1, int(N))]
    try:
        for frame_num in target_frames:
            best_frame, tot_face_detected = utils.select_nbd_frame(frame_num, vid, tot_frames)
            tot_best_faces += tot_face_detected
            best_frames.append(best_frame)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No problem from you side. Unable to select frame")
    
    max_extra_faces = 40
    all_pres_stud_embs = np.empty((tot_best_faces + max_extra_faces, 512), dtype=np.float32)
    
    curr_face_ind = 0
    for frame in best_frames:
        try:
            faces = det_rec_model.get(frame)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to get recognition embeddings")
        for face in faces:
            embeding = face.embedding/np.linalg.norm(face.embedding)
            all_pres_stud_embs[curr_face_ind] = embeding
            curr_face_ind += 1
            if curr_face_ind >= tot_best_faces + max_extra_faces:
                break
    all_pres_stud_embs = all_pres_stud_embs[0:curr_face_ind]
    try:
        db_response = collection.get(
            where = {"course": "B.Tech"},
            include = ["embeddings", "metadatas"]
        )
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Problem from our side")
    
    raw_metadata = db_response["metadatas"]
    stored_embs = np.array(db_response["embeddings"], dtype=np.float32)    
    if stored_embs.size == 0 or all_pres_stud_embs.size==0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either no students in the class or the sent video has no good frame")
    cosine_product = np.dot(stored_embs, all_pres_stud_embs.T)
    
    max_indices = np.argmax(cosine_product, axis=0)
    max_vals = np.max(cosine_product, axis=0)
    filtered_studs_indices = max_indices[max_vals > FACE_SIMILARITY_THRESHOLD]
    filtered_studs_indices = list(set(filtered_studs_indices))
    matching_studs_metadata = [raw_metadata[x] for x in filtered_studs_indices]
    
    roll_num = []
    present_studs_details =[]
    for metadata in matching_studs_metadata:
        if metadata["roll_num"] not in roll_num:
            roll_num.append(metadata["roll_num"])
            stud = schemas.PresentStudent(name = metadata["name"], roll_num = metadata["roll_num"])
            present_studs_details.append(stud)
    
    print(f"total time: {time.time() - s_time}")    
    return schemas.Video_AttendanceResponse(students = present_studs_details)