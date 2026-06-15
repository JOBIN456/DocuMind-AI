from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os
from ingest import run_ingest 
from schema import ChatRequest
from fetch import run_rag
templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/pdf", tags=["pdf entry"])
router_two = APIRouter(prefix="/chat", tags=["chat"])
router_frontend = APIRouter()


@router_frontend.get("/")
def home_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "title": "PDF Chat AI"
        }
    )

@router_frontend.get("/chatpagehome")
def chat_page(request: Request, pdf: str = ""):
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "title": "PDF Chat AI",
            "pdf_name": pdf       # ← pass to template
        }
    )
@router.get("/")
def read_root():
    return {"Hello": "World"}


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    collection_name = file.filename.replace(".pdf", "").replace(" ", "_").lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        count = run_ingest(tmp_path, collection_name)  
        return JSONResponse({
            "status":           "success",
            "filename":         file.filename,
            "collection":       collection_name,      
            "chunks_inserted":  count,
        })
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
        

@router_two.post("/fetch")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty ask questions")

    answer = run_rag(req.question,req.pdf_name)   

    return JSONResponse({
        "question": req.question,
        "answer":   answer,
    })