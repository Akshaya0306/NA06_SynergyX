from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

class InferResponse(BaseModel):
    diagnosis: str
    disease_id: str
    confidence: float
    recommendations: list

@app.post("/infer", response_model=InferResponse)
async def infer(image: UploadFile = File(...)):
    return {
        "diagnosis": "Early Blight",
        "disease_id": "tomato_early_blight",
        "confidence": 0.92,
        "recommendations": [
            {
                "product": "Chlorothalonil 75 WP",
                "dosage": "2 g per litre",
                "notes": "Wear gloves and mask"
            }
        ]
    }
