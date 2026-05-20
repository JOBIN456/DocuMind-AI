from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    pdf_name:str