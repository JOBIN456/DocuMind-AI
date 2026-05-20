from fastapi import FastAPI
import router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(router.router)
app.include_router(router.router_two)
app.include_router(router.router_frontend)