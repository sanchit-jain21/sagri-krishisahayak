from pathlib import Path
import base64
import binascii
import io
import json
import logging
import random
import time

import joblib
import pandas as pd
import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# --- API Keys ---
FAST2SMS_API_KEY = "e8ngMPO4d1ctybvp2IomN3iTkUYuZVWzaFKfE6LqRBXDSQjwx0TNE8wlHfO15rnBIuLzcvdhX439VZmG"

app = FastAPI(title="Sagri ML Backend API")
logger = logging.getLogger("sagri.backend")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load Models ---
# New trained model with soil_type feature
CROP_PRED_DIR = BASE_DIR / "crop prediction"
CROP_MODEL_PATH = CROP_PRED_DIR / "model.pkl"
CROP_ENCODER_PATH = CROP_PRED_DIR / "crop_encoder.pkl"
SOIL_ENCODER_PATH = CROP_PRED_DIR / "soil_encoder.pkl"

crop_model = None
crop_encoder = None
soil_encoder = None

if CROP_MODEL_PATH.exists() and CROP_ENCODER_PATH.exists() and SOIL_ENCODER_PATH.exists():
    try:
        crop_model = joblib.load(CROP_MODEL_PATH)
        crop_encoder = joblib.load(CROP_ENCODER_PATH)
        soil_encoder = joblib.load(SOIL_ENCODER_PATH)
        logger.info("Loaded custom crop model with crop_encoder and soil_encoder.")
    except Exception as e:
        logger.exception("Failed to load crop model: %s", e)
else:
    logger.warning("No crop model found in 'crop prediction' directory. API will use a fallback mock response.")

DISEASE_MODEL_PATH = BASE_DIR / "diseasedetection" / "models" / "final_model.keras"
DISEASE_CLASSES_PATH = BASE_DIR / "diseasedetection" / "models" / "labels.json"
DISEASE_IMAGE_SIZE = (224, 224)
disease_model = None
disease_classes = []

if DISEASE_MODEL_PATH.exists() and DISEASE_CLASSES_PATH.exists():
    try:
        import tensorflow as tf

        disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        with open(DISEASE_CLASSES_PATH, "r", encoding="utf-8") as f:
            if DISEASE_CLASSES_PATH.suffix == ".json":
                labels_dict = json.load(f)
                disease_classes = [None] * len(labels_dict)
                for k, v in labels_dict.items():
                    disease_classes[int(v)] = k
            else:
                disease_classes = [line.strip() for line in f if line.strip()]
        logger.info("Loaded trained Disease Detection model.")
    except ImportError:
        logger.warning("Found disease model, but TensorFlow is not installed.")
    except Exception as e:
        logger.exception("Could not load disease model: %s", e)
else:
    logger.warning("No disease model found.")

PRICE_MODEL_PATH = BASE_DIR / "price prediction/model.pkl"
PRICE_COLUMNS_PATH = BASE_DIR / "price prediction/columns.pkl"
price_model = None
price_columns = None

if PRICE_MODEL_PATH.exists() and PRICE_COLUMNS_PATH.exists():
    try:
        price_model = joblib.load(PRICE_MODEL_PATH)
        price_columns = joblib.load(PRICE_COLUMNS_PATH)
        logger.info("Loaded trained custom XGBoost price model & columns.")
    except Exception as e:
        logger.exception("Could not load price model: %s", e)
else:
    logger.warning("No price prediction model found. Ensure model.pkl and columns.pkl are inside 'price prediction' folder.")

RISK_MODEL_PATH = MODEL_DIR / "crop_risk_model.pkl"
risk_model_data = None

if RISK_MODEL_PATH.exists():
    logger.info("Loaded trained Random Forest crop risk model.")
    risk_model_data = joblib.load(RISK_MODEL_PATH)
else:
    logger.warning("No trained crop risk model found.")


# --- 2. In-Memory Stores ---
# OTP Store: { identifier(phone/email): { "otp": "123456", "expires": <unix_timestamp> } }
_otp_store: dict = {}
# Community Store: List of post objects
_community_posts: list = [
    {
        "id": 1,
        "author": "Rajesh Kumar",
        "location": "Ludhiana, Punjab",
        "time": "2 hours ago",
        "content": "Just harvested my wheat crop! Got excellent yield this season. Happy to share my experience with anyone interested. Used organic fertilizers and proper irrigation timing.",
        "likes": 24,
        "comments": 8,
        "image": True,
    },
    {
        "id": 2,
        "author": "Suresh Patel",
        "location": "Amritsar, Punjab",
        "time": "5 hours ago",
        "content": "Anyone facing issues with leaf blight in wheat? Need advice on treatment. Noticed some brown spots on leaves yesterday.",
        "likes": 12,
        "comments": 15,
        "image": False,
    },
]
OTP_TTL_SECONDS = 300  # 5 minutes


def _generate_otp() -> str:
    return str(random.randint(100000, 999999))


def _store_otp(identifier: str, otp: str) -> None:
    _otp_store[identifier] = {
        "otp": otp,
        "expires": time.time() + OTP_TTL_SECONDS,
    }


def _verify_stored_otp(identifier: str, otp: str) -> tuple[bool, str]:
    entry = _otp_store.get(identifier)
    if not entry:
        return False, "No OTP found. Please request a new one."
    if time.time() > entry["expires"]:
        _otp_store.pop(identifier, None)
        return False, "OTP has expired. Please request a new one."
    if entry["otp"] != otp:
        return False, "Invalid OTP. Please try again."
    _otp_store.pop(identifier, None)  # One-time use — delete after success
    return True, "OTP verified successfully."


def _decode_disease_image(image_data: str):
    import numpy as np
    from PIL import Image

    if not image_data:
        raise HTTPException(status_code=400, detail="No image data received.")

    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        img_bytes = base64.b64decode(encoded.strip(), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format. Please upload a valid JPG or PNG.")

    pil_img = pil_img.resize(DISEASE_IMAGE_SIZE)
    img_array = np.array(pil_img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


# --- 3. Request Models ---
class CropInput(BaseModel):
    soil_type: str
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class PriceForecastInput(BaseModel):
    crop_name: str
    region: str
    days_ahead: int = 30


class RiskInput(BaseModel):
    cropType: str
    soilPh: float
    rainfall: float
    temperature: float
    humidity: float


class DiseaseInput(BaseModel):
    image_data: str


class SendSmsOtpInput(BaseModel):
    phone: str  # 10-digit Indian number


class SendEmailOtpInput(BaseModel):
    email: str


class VerifyOtpInput(BaseModel):
    identifier: str  # phone or email
    otp: str


# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "Sagri ML Backend API is running!"}


# ------ OTP Endpoints ------

@app.post("/api/send-sms-otp")
def send_sms_otp(data: SendSmsOtpInput):
    phone = data.phone.strip()
    if len(phone) != 10 or not phone.isdigit():
        raise HTTPException(status_code=400, detail="Invalid phone number. Must be exactly 10 digits.")

    api_key = FAST2SMS_API_KEY
    if not api_key:
        raise HTTPException(status_code=503, detail="Fast2SMS API key not set.")

    otp = _generate_otp()
    _store_otp(phone, otp)

    headers = {"authorization": api_key}
    payload = {
        "message": f"Your SAGRI login OTP is {otp}. Valid for 5 minutes. Do not share this code with anyone.",
        "route": "q",
        "numbers": phone,
        "flash": 0,
    }

    try:
        resp = http_requests.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers=headers,
            json=payload,
            timeout=10,
        )
        result = resp.json()
        if not result.get("return", False):
            _otp_store.pop(phone, None)
            raise HTTPException(
                status_code=502,
                detail=f"Fast2SMS error: {result.get('message', 'Unknown error')}",
            )
        return {"success": True, "message": f"OTP sent to +91{phone}"}
    except http_requests.RequestException as e:
        _otp_store.pop(phone, None)
        raise HTTPException(status_code=502, detail=f"SMS service unreachable: {str(e)}")


@app.post("/api/verify-sms-otp")
def verify_sms_otp(data: VerifyOtpInput):
    ok, msg = _verify_stored_otp(data.identifier.strip(), data.otp.strip())
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/send-email-otp")
def send_email_otp(data: SendEmailOtpInput):
    email = data.email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")

    otp = _generate_otp()
    _store_otp(email, otp)
    # Return OTP to the frontend — EmailJS will deliver the email from the browser
    return {"success": True, "otp": otp, "message": f"OTP generated for {email}"}


@app.post("/api/verify-email-otp")
def verify_email_otp(data: VerifyOtpInput):
    ok, msg = _verify_stored_otp(data.identifier.strip().lower(), data.otp.strip())
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/predict_crop")
def predict_crop(data: CropInput):
    if crop_model and soil_encoder:
        try:
            import numpy as np
            
            # Map frontend soil type to encoder classes
            soil_mapping = {
                "clay": "Clay",
                "sandy": "Sandy",
                "loamy": "Loamy",
                "silt": "Silt",
                "red": "Red Soil",
                "black": "Black Soil",
            }
            mapped_soil = soil_mapping.get(data.soil_type.lower().strip(), "Clay")
            
            try:
                encoded_soil = soil_encoder.transform([mapped_soil])[0]
            except ValueError:
                encoded_soil = soil_encoder.transform(["Clay"])[0]

            features = np.array([[
                encoded_soil,
                data.N,
                data.P,
                data.K,
                data.temperature,
                data.humidity,
                data.ph,
                data.rainfall,
            ]])
            
            raw_pred = crop_model.predict(features)
            
            if crop_encoder:
                crop_name = crop_encoder.inverse_transform(raw_pred)[0]
            else:
                crop_name = str(raw_pred[0])
            return {"recommended_crop": crop_name, "is_mock": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    mock_crops = ["wheat", "mungbean", "rice", "maize", "cotton", "coffee"]
    return {
        "recommended_crop": random.choice(mock_crops),
        "is_mock": True,
        "warning": "No crop model found. Upload model.pkl, crop_encoder.pkl and soil_encoder.pkl.",
    }


@app.post("/api/forecast_price")
def forecast_price(data: PriceForecastInput):
    import datetime

    is_mock = True
    base_price = 1500
    
    if price_model is not None and price_columns is not None:
        try:
            crop_col = f"crop_{data.crop_name.title()}"
            region_col = f"region_{data.region.title()}"
            
            # Specific mappings for the options in UI to match columns in output exactly
            if data.region.lower() == "up":
                region_col = "region_Uttar Pradesh"
            elif data.region.lower() == "mp":
                region_col = "region_Madhya Pradesh"
                
            df = pd.DataFrame(0, index=[0], columns=price_columns)
            if crop_col in price_columns:
                df[crop_col] = 1
            if region_col in price_columns:
                df[region_col] = 1
                
            prediction = price_model.predict(df)[0]
            base_price = float(prediction)
            is_mock = False
        except Exception as e:
            logger.exception("Price prediction error: %s", e)

    forecast = []
    today = datetime.date.today()
    for i in range(data.days_ahead):
        target_date = today + datetime.timedelta(days=i)
        projected_price = base_price + (i * 1.5) + random.uniform(-15, 15)
        forecast.append(
            {
                "date": target_date.isoformat(),
                "predicted_price": round(projected_price, 2),
            }
        )

    return {"crop": data.crop_name, "forecast": forecast, "is_mock": is_mock}


@app.post("/api/predict_risk")
def predict_risk(data: RiskInput):
    if risk_model_data:
        try:
            import numpy as np

            model = risk_model_data["model"]
            features = risk_model_data["features"]

            input_dict = data.model_dump()
            crop = input_dict.pop("cropType").lower()

            # Keep all feature columns numeric float to avoid dtype cast errors for decimal inputs.
            df = pd.DataFrame(np.zeros((1, len(features)), dtype=float), columns=features)
            df.loc[0, "soilPh"] = input_dict["soilPh"]
            df.loc[0, "rainfall"] = input_dict["rainfall"]
            df.loc[0, "temperature"] = input_dict["temperature"]
            df.loc[0, "humidity"] = input_dict["humidity"]

            crop_col = f"cropType_{crop}"
            if crop_col in features:
                df.loc[0, crop_col] = 1

            prediction = model.predict(df)[0]
            return {"riskLevel": round(prediction, 1), "is_mock": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"riskLevel": random.randint(10, 90), "is_mock": True}


@app.get("/api/disease/status")
def disease_status():
    return {
        "loaded": disease_model is not None and bool(disease_classes),
        "model": "diseasedetection/models/final_model.keras",
        "classes": len(disease_classes),
        "input_size": list(DISEASE_IMAGE_SIZE),
    }


@app.post("/api/detect_disease")
def detect_disease(data: DiseaseInput):
    if not disease_model or not disease_classes:
        logger.error("Disease model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Disease model not loaded. Ensure final_model.keras is in backend/diseasedetection/models/ and TensorFlow is installed.",
        )

    # Disease-specific treatment lookup
    DISEASE_TREATMENTS: dict[str, list[str]] = {
        "Apple___Apple_scab": [
            "Apply fungicides such as myclobutanil or captan every 7-10 days.",
            "Remove and destroy infected leaves and fruit.",
            "Prune trees to improve air circulation.",
            "Avoid overhead irrigation to reduce leaf wetness.",
        ],
        "Apple___Black_rot": [
            "Prune and destroy infected branches, mummified fruit, and cankers.",
            "Apply copper-based fungicides at bud break.",
            "Spray thiophanate-methyl or captan during the growing season.",
            "Remove all crop debris after harvest.",
        ],
        "Apple___Cedar_apple_rust": [
            "Apply fungicides (myclobutanil, mancozeb) starting at pink bud stage.",
            "Remove nearby juniper/cedar trees if possible.",
            "Plant rust-resistant apple varieties.",
            "Spray at 7-10 day intervals during wet spring weather.",
        ],
        "Corn_(maize)___Common_rust_": [
            "Apply fungicides (azoxystrobin, propiconazole) early when pustules appear.",
            "Plant rust-resistant corn hybrids.",
            "Monitor fields regularly for early detection.",
            "Avoid late planting to reduce exposure to high-humidity periods.",
        ],
        "Corn_(maize)___Northern_Leaf_Blight": [
            "Apply foliar fungicides (propiconazole, azoxystrobin) at first sign.",
            "Use resistant hybrids for future planting.",
            "Rotate crops to reduce inoculum buildup in soil.",
            "Manage crop residue by tillage or decomposition.",
        ],
        "Grape___Black_rot": [
            "Apply myclobutanil or mancozeb from early shoot growth.",
            "Remove mummified berries and infected canes before bud break.",
            "Improve canopy airflow through pruning and training.",
            "Spray every 7 days during wet weather.",
        ],
        "Grape___Esca_(Black_Measles)": [
            "Prune and destroy infected wood during dry weather.",
            "Paint pruning wounds with fungicide-based sealant.",
            "Avoid large pruning wounds; use double pruning techniques.",
            "There is no curative treatment; focus on prevention.",
        ],
        "Orange___Haunglongbing_(Citrus_greening)": [
            "Remove and destroy infected trees immediately to prevent spread.",
            "Control the Asian citrus psyllid vector with insecticides.",
            "Use certified disease-free planting material.",
            "There is currently no cure; management focuses on prevention.",
        ],
        "Peach___Bacterial_spot": [
            "Apply copper-based bactericides at petal fall and repeat every 10-14 days.",
            "Plant resistant peach varieties.",
            "Avoid overhead irrigation to minimize water splash.",
            "Remove and destroy heavily infected plant material.",
        ],
        "Pepper,_bell___Bacterial_spot": [
            "Apply copper hydroxide or copper octanoate sprays preventively.",
            "Use certified disease-free seed and transplants.",
            "Rotate crops with non-host plants for at least 2 years.",
            "Avoid working in the field when foliage is wet.",
        ],
        "Potato___Early_blight": [
            "Apply chlorothalonil or mancozeb at first symptom appearance.",
            "Ensure adequate plant nutrition, especially nitrogen and potassium.",
            "Destroy infected plant debris after harvest.",
            "Use certified, disease-free seed tubers.",
        ],
        "Potato___Late_blight": [
            "Apply metalaxyl or cymoxanil fungicides preventively before infection.",
            "Destroy all infected plant material immediately.",
            "Avoid overhead irrigation; water at the base of plants.",
            "Plant resistant potato varieties.",
        ],
        "Strawberry___Leaf_scorch": [
            "Apply captan or myclobutanil fungicides early in the season.",
            "Remove and destroy infected leaves promptly.",
            "Improve air circulation by appropriate plant spacing.",
            "Avoid wetting foliage during irrigation.",
        ],
        "Tomato___Early_blight": [
            "Spray mancozeb or chlorothalonil at first sign of disease.",
            "Remove lower infected leaves to slow spread.",
            "Stake or cage tomatoes to improve air circulation.",
            "Apply mulch to prevent soil splash onto leaves.",
        ],
        "Tomato___Late_blight": [
            "Apply copper fungicides or metalaxyl-based products immediately.",
            "Remove and destroy all infected plant parts.",
            "Avoid overhead watering; water at soil level.",
            "Grow resistant tomato varieties in future seasons.",
        ],
        "Tomato___Leaf_Mold": [
            "Apply fungicides such as chlorothalonil or mancozeb preventively.",
            "Improve greenhouse ventilation to reduce humidity.",
            "Remove and destroy infected leaves.",
            "Space plants adequately for good air circulation.",
        ],
        "Tomato___Septoria_leaf_spot": [
            "Spray mancozeb or copper-based fungicides at first appearance.",
            "Remove and destroy infected lower leaves.",
            "Mulch around plants to prevent soil splash.",
            "Rotate crops annually to reduce soilborne inoculum.",
        ],
        "Tomato___Target_Spot": [
            "Apply azoxystrobin or chlorothalonil fungicide sprays.",
            "Remove and dispose of heavily infected leaves.",
            "Ensure good plant spacing and pruning for air circulation.",
            "Avoid overhead irrigation.",
        ],
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
            "Control whitefly vectors with insecticides (imidacloprid, thiamethoxam).",
            "Use yellow sticky traps to monitor whitefly populations.",
            "Plant resistant tomato varieties.",
            "Remove and destroy infected plants promptly.",
        ],
        "Tomato___Tomato_mosaic_virus": [
            "Remove and destroy infected plants immediately.",
            "Disinfect tools with 10% bleach solution between each plant.",
            "Wash hands thoroughly before handling plants.",
            "Control aphid vectors and use virus-free certified seeds.",
        ],
    }

    DEFAULT_TREATMENT = [
        "Isolate the affected plants immediately.",
        "Consult a local agricultural expert.",
        "Consider appropriate fungicide/pesticide treatment.",
        "Remove and destroy severely affected leaves.",
    ]

    try:
        import numpy as np

        logger.info("Received image for disease detection (payload length: %s)", len(data.image_data))
        img_array = _decode_disease_image(data.image_data)
        logger.info("Preprocessed image array shape: %s", img_array.shape)

        predictions = disease_model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(predictions[0]))
        if class_idx >= len(disease_classes) or disease_classes[class_idx] is None:
            raise HTTPException(status_code=500, detail="Disease model returned an unknown class index.")

        confidence = float(predictions[0][class_idx]) * 100
        disease_name = disease_classes[class_idx]
        is_healthy = "healthy" in disease_name.lower()

        clean_disease_name = disease_name.replace("___", " - ").replace("_", " ")
        treatment_steps = [] if is_healthy else DISEASE_TREATMENTS.get(disease_name, DEFAULT_TREATMENT)
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "disease": disease_classes[int(idx)].replace("___", " - ").replace("_", " "),
                "confidence": round(float(predictions[0][int(idx)]) * 100, 1),
            }
            for idx in top_indices
            if int(idx) < len(disease_classes) and disease_classes[int(idx)] is not None
        ]
        second_best = top_predictions[1]["confidence"] if len(top_predictions) > 1 else 0.0
        confidence_gap = round(confidence - second_best, 1)
        is_uncertain = (confidence < 55.0) or (confidence_gap < 12.0)

        logger.info("Disease prediction: %s (%.2f%%)", disease_name, confidence)

        if is_uncertain:
            return {
                "disease": "Uncertain result",
                "confidence": round(confidence, 1),
                "severity": "Needs Review",
                "recommendation": "Image confidence is low. Upload a clearer close-up of the affected leaf in good lighting.",
                "treatment": DEFAULT_TREATMENT,
                "color": "orange",
                "top_predictions": top_predictions,
            }

        return {
            "disease": clean_disease_name,
            "confidence": round(confidence, 1),
            "severity": "None" if is_healthy else ("High" if confidence > 80 else "Medium"),
            "recommendation": (
                "Crop looks great! No disease detected."
                if is_healthy
                else f"AI identified possible {clean_disease_name}. Please take action."
            ),
            "treatment": treatment_steps,
            "color": "green" if is_healthy else "red",
            "top_predictions": top_predictions,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Backend error during disease detection: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
