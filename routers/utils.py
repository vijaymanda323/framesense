import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

logger = logging.getLogger("framesense.utils")

router = APIRouter()

@router.post("/utils/base64", summary="Convert uploaded image to base64 data URL")
async def get_base64_url(file: UploadFile = File(...)):
    """
    Accepts an image file upload and returns its base64 encoded data URL.
    Useful for quick visualization or passing images between systems.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        encoded = base64.b64encode(contents).decode("utf-8")
        data_url = f"data:{file.content_type};base64,{encoded}"
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "base64_url": data_url
        }
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during base64 encoding.")
    finally:
        await file.close()
