from pydantic import BaseModel

class Data(BaseModel):
    genres: str
    tags: str
    description: str