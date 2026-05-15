import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import NOMBRES_CLASES, ConfigEntrenamiento
cfg = ConfigEntrenamiento()

from src.predict import cargar_modelo, predecir as clasificar_senal

_modelo = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _modelo
    _modelo = cargar_modelo()
    yield
    _modelo = None


app = FastAPI(
    title="Clasificador de Latidos ECG",
    description=f"Clasifica un único latido ECG en {cfg.num_clases} categorías de arritmia (MIT-BIH).",
    version="1.0.0",
    lifespan=lifespan,
)


class PeticionLatido(BaseModel):
    senal: list[float] = Field(..., description=f"{cfg.tamano_entrada} valores de amplitud normalizados correspondientes a un latido")

    @field_validator("senal")
    @classmethod
    def verificar_longitud(cls, v):
        if len(v) != 187:
            raise ValueError(
                f"La señal debe tener exactamente {cfg.tamano_entrada} valores, se recibieron {len(v)}"
            )
        return v


class RespuestaLatido(BaseModel):
    clase_predicha: int
    etiqueta: str
    probabilidades: dict[str, float]


@app.get("/", tags=["estado"])
def raiz():
    return {"estado": "ok", "servicio": "Clasificador de Latidos ECG"}


@app.get("/health", tags=["estado"])
def health():
    return {"estado": "ok"}


@app.get("/clases", tags=["informacion"])
def obtener_clases():
    return NOMBRES_CLASES


@app.post("/predecir", response_model=RespuestaLatido, tags=["inferencia"])
def predecir(peticion: PeticionLatido):
    if _modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    resultado = clasificar_senal(peticion.senal, modelo=_modelo)
    return resultado


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
