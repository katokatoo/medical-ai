from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import shutil
import os
import logging

from src.inference.predictor import Predictor


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# lifespan（起動・終了処理）
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading model...")

    # Predictorインスタンスを作成
    app.state.predictor = Predictor()

    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down API...")


# FastAPIインスタンス生成
app = FastAPI(lifespan=lifespan)

# templatesディレクトリのパスを設定
current_file_path = os.path.abspath(__file__) # /app/src/api/main.py
api_dir = os.path.dirname(current_file_path)  # /app/src/api
src_dir = os.path.dirname(api_dir)           # /app/src
template_path = os.path.join(src_dir, "templates")

# templates
templates = Jinja2Templates(directory=template_path)

# root（HTML表示）
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )


# prediction（HTML返却）
@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):

    try:

        # ファイル形式チェック
        if not file.filename.endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400,
                detail="Only image files are allowed"
            )

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # ファイル保存
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File received: {file.filename}")

        # Predictorインスタンスを取得
        predictor = request.app.state.predictor

        # 推論
        prob, pred = predictor.predict(file_path)

        label = "Malignant" if pred == 1 else "Benign"

        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "request": request,
                "prediction": label,
                "probability": round(prob, 4)
            }
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )