import os
from pymongo import MongoClient
import gridfs
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["lecturas_db"]
c_fs = gridfs.GridFS(db)
collection = db["evaluaciones"]

class Load:
    def save(self, text: str, filename: str, audio_bytes: bytes, record: dict) -> bool:
        audio_id = c_fs.put(audio_bytes, filename=filename)
        doc = {
            "texto_original": text,
            "transcripcion_detectada": record.get("transcripcion"),
            "similaridad": record.get("similaridad"),
            "evaluacion": record.get("evaluacion"),
            "palabras_por_minuto": record.get("palabras_por_minuto"),
            "filename": filename,
            "audio_file_id": audio_id,
            "fecha": datetime.now(),
        }
        collection.insert_one(doc)
        return True
