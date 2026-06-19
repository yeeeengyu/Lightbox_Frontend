# LightBox Dashboard

졸음운전 감지 서버와 연결하기 위한 간단한 React 대시보드입니다.

## 실행

```bash
npm install
npm run dev
```

## 웹앱 배포

```bash
npm run build
npm run preview
```

HTTPS로 배포하면 iPhone Safari에서 카메라 권한을 사용할 수 있고, 공유 메뉴의 “홈 화면에 추가”로 앱처럼 실행할 수 있습니다. PWA manifest, iOS 메타 태그, 서비스 워커가 포함되어 있습니다.

## 화면 구성

- 실시간 카메라 미리보기
- 현재 위험도 및 감지 이벤트 요약
- 졸음 감지 빈도 그래프
- 최근 알림 로그
- 토스트 알림
- 홈 화면 추가용 웹앱 설정

현재 버전은 브라우저에서 좌/우 눈 포인트를 추출하고, WebSocket 서버로 전송한 뒤 서버 판정 응답을 화면에 반영합니다.

## WebSocket 연동

눈 감김 감지는 공개 서버의 `/ws/keypoints` 경로를 사용합니다.

```text
wss://spoti.ingyuc.click/ws/keypoints
```

하품 감지는 `/ws/yawn` 경로를 사용합니다.

```text
wss://spoti.ingyuc.click/ws/yawn
```

로컬 서버나 다른 주소를 쓰려면 `.env`에 설정합니다.

```bash
VITE_DROWSINESS_WS_URL=ws://localhost:8001/ws/keypoints
VITE_YAWN_WS_URL=ws://localhost:8001/ws/yawn
VITE_YAWN_FRAME_INTERVAL_MS=1500
```

프론트는 좌우 눈 6개 포인트를 추출해서 약 250ms 간격으로 보냅니다.
하품 감지 WebSocket에는 카메라 프레임을 긴 변 480px 이하의 JPEG 바이너리로 약 1500ms 간격으로 보냅니다.

```json
{
  "rightEye": [
    { "x": 0.1, "y": 0.2 },
    { "x": 0.2, "y": 0.2 },
    { "x": 0.3, "y": 0.2 },
    { "x": 0.4, "y": 0.2 },
    { "x": 0.3, "y": 0.3 },
    { "x": 0.2, "y": 0.3 }
  ],
  "leftEye": [
    { "x": 0.5, "y": 0.2 },
    { "x": 0.6, "y": 0.2 },
    { "x": 0.7, "y": 0.2 },
    { "x": 0.8, "y": 0.2 },
    { "x": 0.7, "y": 0.3 },
    { "x": 0.6, "y": 0.3 }
  ]
}
```

눈 포인트 인덱스는 다음 순서입니다.

```text
leftEye:  33, 160, 158, 133, 153, 144
rightEye: 362, 385, 387, 263, 373, 380
```

서버 응답은 아래 형태를 기준으로 대시보드에 반영합니다.

```json
{
  "ok": true,
  "ear": 0.23,
  "ear_smooth": 0.22,
  "status": "OPEN"
}
```

`status`가 `CLOSED` 또는 `CLOSE`로 오면 화면 하단에 “위웅위웅! 눈 감김 감지” 알람이 뜨고, 가능한 브라우저에서는 짧은 사이렌 톤을 재생합니다.

위험도는 서버가 `risk`를 직접 주면 그 값을 사용하고, 없으면 `status`와 EAR로 계산합니다. `OPEN`일 때도 `ear_smooth` 또는 `ear`가 낮아질수록 0~70% 사이에서 점진적으로 올라가며, `CLOSED`는 기본 90%에서 최대 100%까지 올라갑니다.

프론트는 하품 감지 WebSocket에 연결한 뒤 서버 응답을 수신합니다. 서버가 아래 응답을 보내면 노란색 주의 토스트와 로그로 표시합니다.

```json
{
  "ok": true,
  "yawning": true,
  "yawn_confidence": 0.87,
  "detections": [
    {
      "label": "yawning",
      "confidence": 0.87,
      "box": [120.0, 80.0, 260.0, 220.0]
    }
  ]
}
```
