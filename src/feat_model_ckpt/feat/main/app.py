import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uvicorn
from server import app, AppConfig  # noqa: F401


if __name__ == '__main__':
    uvicorn.run(
        app='app:app',
        host=AppConfig.app_host,
        port=AppConfig.app_port,
        root_path=AppConfig.app_root_path,
        reload=AppConfig.app_reload,
    )