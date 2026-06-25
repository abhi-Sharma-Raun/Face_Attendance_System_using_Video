from pydantic import BaseModel
from typing import List


class PresentStudent(BaseModel):
    name: str
    roll_num: int
    
    
class Video_AttendanceResponse(BaseModel):
    students: List[PresentStudent]
    
class Face_registrationResponse(BaseModel):
    name: str
    roll_num: int
    