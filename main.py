import uvicorn

from app.config import Settings
from app.server import create_app

if __name__ == "__main__":
    settings = Settings()
    uvicorn.run(
        create_app(),
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
