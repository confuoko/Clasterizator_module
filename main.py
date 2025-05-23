from fastapi import FastAPI
from views import api_router

app = FastAPI(
    title="My API",
    description="Простое API с маршрутом /ping",
    version="1.0.0"
)

# Подключаем все маршруты
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

