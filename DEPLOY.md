# 배포 시 MIME 오류 (main.tsx / text/html) 해결

브라우저에 **"Expected a JavaScript module but server responded with text/html"** 가 나오면,  
서버가 **개발용 index.html**을 내려보내고 있다는 뜻입니다. (개발용은 `/src/main.tsx`를 로드해서 JS 요청에도 HTML이 가면 오류 발생)

## 해결: 반드시 **빌드 결과물**만 배포

- ✅ 사용할 것: **`frontend/dist` 폴더 안의 내용** (빌드 후 생성됨)
- ❌ 사용하면 안 되는 것: `frontend` 폴더 전체, 또는 `frontend/index.html` 만 복사

### 1) 이 프로젝트 백엔드(FastAPI)로 서빙할 때

```bash
# 프로젝트 루트 (zimage_webapp)
./scripts/build_and_serve.sh
cd backend && ./run.sh
```

- `build_and_serve.sh` 가 `frontend/dist` 내용을 `backend/static_frontend` 로 복사합니다.
- 백엔드는 `static_frontend` 안의 index.html 이 **/assets/** 를 참조할 때만 프론트를 서빙합니다.  
  (dev용 index면 서빙하지 않고 안내 페이지만 띄움)

### 2) nginx 등 다른 웹서버로 배포할 때

- **배포할 디렉터리**: `frontend/dist` **안의 내용**을 그대로 사용 (dist 자체가 아니라 dist 안의 index.html, assets/ 등)
- **중요**: `location /` 에서 `try_files $uri $uri/ /index.html` 을 쓰면,  
  **실제 파일이 있는 요청**(예: `/assets/index-xxx.js`) 은 반드시 **그 파일**로 응답해야 하고,  
  **없는 경로**일 때만 `index.html` 로 넘겨야 합니다.  
  그래야 JS 요청에 HTML이 안 가서 MIME 오류가 나지 않습니다.

nginx 예시 (백엔드 8000 포트로 API 프록시). **이미지 생성·LLM 채팅**이 오래 걸리므로 타임아웃을 넉넉히 두어야 "invalid response from upstream" / "Network Error" 가 나지 않습니다. **LLM 채팅**은 1~5분 걸릴 수 있으므로 `/api/llm/` 은 600초(10분) 권장.

```nginx
server {
    listen 80;
    root /path/to/frontend/dist;
    index index.html;

    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 30s;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    location /static/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
    location /health {
        proxy_pass http://127.0.0.1:8000;
    }
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- `root` 를 **dist** 로 두면 `/assets/index-xxx.js` 요청은 `dist/assets/index-xxx.js` 파일로 응답되고,  
  없는 경로만 `index.html` 로 갑니다.

### 3) 확인 방법

- 배포한 서버에서 `curl -I http://서버/assets/아무거나.js` (실제 있는 파일명으로) 호출 시  
  `Content-Type: application/javascript` (또는 `text/javascript`) 가 나와야 합니다.  
  `text/html` 이 나오면 아직 HTML이 내려가고 있는 것이므로, 위처럼 **dist 내용**과 **try_files** 동작을 다시 확인하면 됩니다.
