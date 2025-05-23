import uvicorn
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import mlflow
from agent import AgentRequest, run_agent, reset_agent

mlflow.pydantic_ai.autolog()
mlflow.set_tracking_uri("/home/coder/smart-home/mlflow")
_ = mlflow.set_experiment(f"{os.getenv("MLFLOW_EXPERIMENT_NAME", "Smart Home Agent")}")

app = FastAPI()

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
BUILD_DIR = Path(SCRIPT_DIR / "whisper.cpp/build-em/bin/command.wasm").resolve()
COI_WORKER_FILE = SCRIPT_DIR / "whisper.cpp/examples/coi-serviceworker.js"


# Serve all .js, .worker.js files from build directory
@app.get(f"/whisper/{{file_path:path}}")
async def serve_static_files(_: Request, file_path: str):
    full_path = BUILD_DIR / file_path

    if file_path == "":
        return RedirectResponse(url=f"index.html")

    if not full_path.resolve().parent == BUILD_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    if full_path.exists() and full_path.is_file():
        return FileResponse(
            full_path,
            headers={
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Access-Control-Allow-Origin": "*",
            },
        )
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/")
async def redirect_root():
    return RedirectResponse(url=f"whisper/")


@app.get("/coi-serviceworker.js")
async def serve_coi_worker():
    if COI_WORKER_FILE.exists():
        return FileResponse(
            COI_WORKER_FILE,
            media_type="application/javascript",
            headers={
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Access-Control-Allow-Origin": "*",
            },
        )
    raise HTTPException(status_code=404, detail="coi-serviceworker.js not found")


@app.post(path="/converse")
async def converse(request: AgentRequest) -> JSONResponse:
    response = await run_agent(request)
    return JSONResponse(response)


@app.post(path="/reset")
async def reset() -> JSONResponse:
    await reset_agent()
    return JSONResponse("{'status': 'ok'}")


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, log_level="debug", port=int(os.getenv("UVICORN_PORT", "8912")))
