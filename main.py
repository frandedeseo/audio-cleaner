from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.EvaluationService import EvaluationService

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluation_service = EvaluationService()

@app.post("/evaluar-lectura")
async def evaluar_lectura(text: str = Form(...), audio: UploadFile = File(...)):
    try:
        result = await evaluation_service.handle(text, audio)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/")
def read_root():    
    return {"message": "Hello from FastAPI on Vercel"}