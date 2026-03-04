# GET /static/generated/xxx.png 404 — 근본 원인과 해결

## 근본 원인

이미지는 **해당 요청을 처리한 Pod/프로세스의 로컬 디스크**에만 저장됩니다.

- `POST /api/generate` → Pod A가 처리 → 파일 저장: **Pod A의** `backend_dir/static/generated/xxx.png`
- 클라이언트가 응답의 `generated_url`(/static/generated/xxx.png)로 **GET** 요청
- 로드밸런서가 **Pod B**로 GET 요청 전달 → Pod B의 디스크에는 해당 파일 없음 → **404 Not Found**

즉, **멀티 replica(여러 Pod) 환경**에서 저장 위치와 조회 요청이 서로 다른 인스턴스로 가면 404가 납니다. 단일 Pod여도 재시작/스케일 아웃 후에는 같은 파일이 없을 수 있습니다.

## 적용된 해결 (코드)

1. **응답에 base64 포함**  
   `POST /api/generate` 응답에 `generated_image_base64` 필드를 추가했습니다.  
   프론트에서는 이 값을 사용해 **두 번째 GET 없이** 바로 이미지를 표시할 수 있습니다. (멀티 Pod에서도 동일하게 동작)

2. **시작 시 경로 로그**  
   서버 기동 시 `Static directory ready`, `Root /static mount` 로그에 **실제 사용 중인 static/generated 절대 경로**가 출력됩니다.  
   - 단일 Pod에서도 404가 나면 이 로그로 “저장 경로와 서빙 경로가 같은 디렉터리인지” 확인할 수 있습니다.

3. **zimage 프론트엔드**  
   `ResultViewer`에서 `generated_image_base64`가 있으면 URL 대신 이 값을 사용해 표시·다운로드합니다.

## 인프라로 해결하려면

- **공유 스토리지**: 모든 replica가 같은 디렉터리를 바라보게 합니다.  
  - Kubernetes: **ReadWriteMany PVC**를 `static/generated`에 마운트 (예: NFS, EFS 등).  
  - docker-compose: `generated_data` 볼륨으로 `/app/static/generated` 공유 (이미 설정됨).
- **replica 1개**: Pod를 1개만 두면 GET도 같은 Pod로 갈 수 있어, 로컬 디스크만 써도 404가 나지 않을 수 있습니다. (재시작 시에는 사라짐.)
- **Sticky session**: 같은 클라이언트를 항상 같은 Pod로 보내도록 설정. 인프라 설정이 필요하고, Pod 재시작 시 파일이 없으면 여전히 404 가능.

정리하면, **멀티 Pod에서 안정적으로 쓰려면**  
- 응답의 **base64**로 표시/다운로드하거나  
- **공유 볼륨(PVC 등)**으로 `static/generated`를 모든 Pod가 같이 쓰도록 하면 됩니다.
