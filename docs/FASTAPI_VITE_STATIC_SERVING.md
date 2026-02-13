# FastAPI + Vite 빌드 프론트엔드 정적 서빙 (Production)

## 왜 이전 설정이 실패했는가

**증상:** `Failed to load module script: Expected a JavaScript module but the server responded with a MIME type of "text/html"`

**원인:**  
브라우저가 `/assets/index-xxxx.js` 를 요청했는데, 서버가 **index.html**을 내려보냄.  
즉, **JS/CSS 정적 요청이 SPA 폴백으로 처리**되어 HTML이 반환되고, 브라우저는 JS로 파싱하려다 MIME 오류 발생.

**구체적 원인:**  
- FastAPI에서 **catch-all 라우트** `GET /{path:path}` 로 “나머지 모든 경로 → index.html” 을 구현한 경우.
- Starlette/FastAPI 라우팅에서 **경로 매칭 순서**에 따라 `/{path:path}` 가 **`/assets/xxx` 보다 먼저** 매칭되거나, mount 보다 나중에 등록된 catch-all 이 **`/assets` 마운트보다 우선** 처리되는 경우가 있음.
- 그 결과 `/assets/index-xxxx.js` 요청이 catch-all 에 걸려 **index.html**이 반환됨 → MIME 오류.

**해결 방향:**  
- **`/assets/*` 는 반드시 StaticFiles 마운트만 처리** 하도록 하고,  
- **SPA 폴백은 “실제로 404가 났을 때만”** index.html 을 주는 방식으로 구현.

---

## 올바른 마운트/라우트 순서

1. **API 라우터** (`/api/*`)  
2. **기타 고정 경로** (예: `/health`, `/docs` 는 FastAPI 기본)  
3. **백엔드 정적 파일** (예: `/static` → 생성 이미지 등)  
4. **프론트엔드 정적 자산** → **`/assets` 마운트** (반드시 여기서만 `/assets/*` 처리)  
5. **루트 문서** → **`GET /`** → `index.html`  
6. **SPA 폴백** → **404 예외 핸들러**에서 GET 요청일 때만 `index.html` 반환 (catch-all 라우트 사용 안 함)

이 순서로 두면:

- `/assets/index-xxxx.js` → **4번** StaticFiles → JS 파일 + `Content-Type: application/javascript`
- `/` → **5번** → index.html
- `/some-spa-route` → 어떤 라우트/마운트에도 안 걸림 → 404 → **6번** 핸들러 → index.html

---

## 프로젝트 구조 (가정)

```
project/
├── app/
│   └── main.py
├── frontend/
│   ├── src/
│   ├── dist/
│   │   ├── index.html
│   │   └── assets/
│   │       ├── index-xxxx.js
│   │       └── index-xxxx.css
│   └── vite.config.ts
```

---

## Production-ready FastAPI `main.py` 예시

아래는 **프론트는 `frontend/dist` 에만** 두고, **API/health/static 은 기존처럼** 두는 경우의 최소 예시다.  
(기존 API, lifespan, CORS 등은 그대로 두고, “정적 서빙 + SPA 폴백” 부분만 이 패턴으로 맞추면 된다.)

```python
# app/main.py (관련 부분만)

from pathlib import Path
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 1) 프로젝트 루트 = app 의 부모
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

def setup_frontend(app: FastAPI) -> None:
    """Vite 빌드 결과(frontend/dist)를 / 와 /assets 에서 서빙. SPA 폴백은 404 핸들러로."""
    index_path = FRONTEND_DIST / "index.html"
    if not index_path.exists():
        return

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        # 반드시 /assets 는 StaticFiles 만 처리 → JS/CSS 가 올바른 MIME 로 내려감
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend_assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path, media_type="text/html")

    # SPA 폴백: 404 일 때만 index.html. catch-all 라우트 사용 안 함 → /assets/* 가 HTML 로 안 나감
    app.state._frontend_index_path = index_path

    @app.exception_handler(StarletteHTTPException)
    async def spa_fallback(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404 and request.method == "GET":
            path = getattr(app.state, "_frontend_index_path", None)
            if path is not None:
                return FileResponse(path, media_type="text/html")
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
```

- **API, /health, /static** 등은 `setup_frontend` **이전에** 이미 등록되어 있다고 가정.
- **등록 순서:** API → /health → /static → **setup_frontend()** (여기서 `/assets` 마운트 → `GET /` → 404 핸들러).

---

## Vite 설정 (`vite.config.ts`)

빌드 결과가 **루트 기준** `/assets/...` 로 요청되도록 base 를 `"/"` 로 두면, FastAPI 에서 `/assets` 마운트와 그대로 맞다.

```ts
// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
});
```

---

## 동작 정리

| 요청 | 처리 위치 | 응답 |
|------|-----------|------|
| `GET /` | `GET /` 라우트 | `frontend/dist/index.html` (text/html) |
| `GET /assets/index-xxxx.js` | `/assets` StaticFiles | JS 파일 (application/javascript) |
| `GET /assets/index-xxxx.css` | `/assets` StaticFiles | CSS 파일 (text/css) |
| `GET /api/...` | API 라우터 | JSON 등 API 응답 |
| `GET /any-other-path` | 매칭 없음 → 404 | 404 핸들러 → index.html (SPA 라우팅) |

**JS/CSS 요청이 절대 index.html 로 폴백되지 않으므로** MIME 오류가 나지 않는다.
