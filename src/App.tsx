import {
  AlertTriangle,
  Bell,
  Camera,
  CheckCircle2,
  CircleGauge,
  Clock3,
  Moon,
  Radio,
  ScanFace,
  ShieldAlert,
  Sun,
  Video,
  Volume2,
  Wifi,
  WifiOff
} from "lucide-react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { ReactNode, RefObject } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type AlertLevel = "normal" | "warning" | "danger";
type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

type EyePoint = {
  x: number;
  y: number;
};

type EyePayload = {
  leftEye: EyePoint[];
  rightEye: EyePoint[];
};

type DetectionEvent = {
  id: number;
  time: string;
  label: string;
  level: AlertLevel;
  risk: number;
};

type FrequencyPoint = {
  label: string;
  count: number;
};

type ServerDecision = {
  isDrowsy: boolean;
  level: AlertLevel;
  risk: number;
  message: string;
  ear?: number;
  earSmooth?: number;
  status?: string;
  yawnConfidence?: number;
};

const DROWSINESS_WS_URL =
  import.meta.env.VITE_DROWSINESS_WS_URL ?? "wss://spoti.ingyuc.click/ws/keypoints";
const YAWN_WS_URL = import.meta.env.VITE_YAWN_WS_URL ?? "wss://spoti.ingyuc.click/ws/yawn";
const MEDIAPIPE_WASM_URL =
  import.meta.env.VITE_MEDIAPIPE_WASM_URL ?? "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const FACE_LANDMARKER_MODEL_URL =
  import.meta.env.VITE_FACE_LANDMARKER_MODEL_URL ??
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const OPEN_EAR_SAFE = 0.34;
const OPEN_EAR_WARNING = 0.22;
const OPEN_RISK_MAX = 70;
const CLOSED_RISK_BASE = 90;
const CLOSED_EAR_REFERENCE = 0.22;
const CLOSED_EAR_MIN = 0.12;

const emptyFrequency: FrequencyPoint[] = Array.from({ length: 10 }, (_, index) => ({
  label: `-${9 - index}`,
  count: 0
}));

const levelText: Record<AlertLevel, string> = {
  normal: "정상",
  warning: "주의",
  danger: "위험"
};

const connectionText: Record<ConnectionStatus, string> = {
  connecting: "서버 연결 중",
  connected: "서버 연결됨",
  disconnected: "서버 연결 끊김",
  error: "서버 연결 오류"
};

const levelDescription: Record<AlertLevel, string> = {
  normal: "운전자 상태가 안정적입니다.",
  warning: "졸음 징후가 감지되고 있습니다.",
  danger: "즉시 경고가 필요한 상태입니다."
};

function normalizeWsUrl(url: string) {
  if (url.startsWith("https://")) {
    return url.replace("https://", "wss://");
  }

  if (url.startsWith("http://")) {
    return url.replace("http://", "ws://");
  }

  return url;
}

function formatTime(date: Date) {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(date);
}

function formatMinute(date: Date) {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  }).format(date);
}

function clampRisk(value: unknown) {
  const numberValue = Number(value);

  if (!Number.isFinite(numberValue)) {
    return 0;
  }

  if (numberValue <= 1) {
    return Math.round(numberValue * 100);
  }

  return Math.max(0, Math.min(100, Math.round(numberValue)));
}

function normalizeLevel(value: unknown, risk: number, isDrowsy: boolean): AlertLevel {
  if (typeof value === "string") {
    const normalized = value.toLowerCase();

    if (["danger", "critical", "drowsy", "closed", "close", "위험"].includes(normalized)) {
      return "danger";
    }

    if (["warning", "warn", "caution", "yawn", "yawning", "주의", "하품"].includes(normalized)) {
      return "warning";
    }

    if (["normal", "safe", "ok", "open", "정상"].includes(normalized)) {
      return "normal";
    }
  }

  if (risk >= 76 || isDrowsy) {
    return "danger";
  }

  if (risk >= 50) {
    return "warning";
  }

  return "normal";
}

function isClosedEyeStatus(status?: string) {
  return status === "CLOSED" || status === "CLOSE";
}

function isYawningStatus(status?: string) {
  return status === "YAWN" || status === "YAWNING";
}

function calculateStatusRisk(status: string | undefined, ear: number | undefined, earSmooth: number | undefined): number {
  const earValue = earSmooth ?? ear;

  if (isYawningStatus(status)) {
    return 55;
  }

  if (status === "OPEN") {
    if (earValue === undefined) {
      return 0;
    }

    const closingRatio = Math.max(0, Math.min(1, (OPEN_EAR_SAFE - earValue) / (OPEN_EAR_SAFE - OPEN_EAR_WARNING)));
    return Math.round(closingRatio * OPEN_RISK_MAX);
  }

  if (!isClosedEyeStatus(status)) {
    return earValue === undefined ? 0 : calculateStatusRisk("OPEN", ear, earSmooth);
  }

  if (earValue === undefined) {
    return CLOSED_RISK_BASE;
  }

  const closedDepth = Math.max(0, Math.min(1, (CLOSED_EAR_REFERENCE - earValue) / (CLOSED_EAR_REFERENCE - CLOSED_EAR_MIN)));
  return Math.round(CLOSED_RISK_BASE + closedDepth * (100 - CLOSED_RISK_BASE));
}

function parseServerDecision(raw: unknown): ServerDecision | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const data = raw as Record<string, unknown>;
  const statusValue = data.yawning === true ? "YAWNING" : data.status ?? data.state ?? data.label ?? data.event ?? data.class;
  const status = typeof statusValue === "string" ? statusValue.toUpperCase() : undefined;
  const ear = typeof data.ear === "number" ? data.ear : undefined;
  const earSmooth = typeof data.ear_smooth === "number" ? data.ear_smooth : undefined;
  const yawnConfidence = typeof data.yawn_confidence === "number" ? data.yawn_confidence : undefined;
  const isClosed = isClosedEyeStatus(status);
  const isYawning = isYawningStatus(status);
  const risk = clampRisk(
    data.risk ??
      data.score ??
      data.probability ??
      data.drowsiness_score ??
      (isYawning && yawnConfidence !== undefined ? yawnConfidence : calculateStatusRisk(status, ear, earSmooth))
  );
  const isDrowsy = Boolean(
    data.is_drowsy ??
      data.isDrowsy ??
      data.drowsy ??
      data.sleepy ??
      data.alarm ??
      (status ? status !== "OPEN" : false)
  );
  const level = normalizeLevel(data.level ?? data.state ?? status, risk, isDrowsy);
  const fallbackMessage = status === "OPEN"
    ? "눈 뜸 상태"
    : isClosed
      ? "위웅위웅! 눈 감김 감지"
      : isYawning
        ? "하품 감지"
      : status
        ? `눈 상태 ${status}`
        : level === "normal"
          ? "정상 상태"
          : "졸음 징후 감지";

  return {
    isDrowsy: isDrowsy || level !== "normal",
    level,
    risk,
    message: String(data.message ?? data.reason ?? (isYawning ? fallbackMessage : data.label) ?? fallbackMessage),
    ear,
    earSmooth,
    status,
    yawnConfidence
  };
}

function useCameraPreview() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [cameraStatus, setCameraStatus] = useState("카메라 대기 중");

  useEffect(() => {
    let stream: MediaStream | null = null;
    let isMounted = true;

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus("브라우저 카메라 미지원");
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
          audio: false
        });

        if (isMounted && videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraStatus("카메라 연결됨");
        }
      } catch {
        if (isMounted) {
          setCameraStatus("카메라 권한 필요");
        }
      }
    }

    startCamera();

    return () => {
      isMounted = false;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, cameraStatus };
}

function useDrowsinessSocket(onDecision: (decision: ServerDecision) => void) {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const wsUrl = useMemo(() => normalizeWsUrl(DROWSINESS_WS_URL), []);

  useEffect(() => {
    let isClosedByEffect = false;

    function connect() {
      setConnectionStatus("connecting");

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnectionStatus("connected");
      };

      socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(String(event.data));
          const decision = parseServerDecision(parsed);

          if (decision) {
            onDecision(decision);
          }
        } catch {
          setConnectionStatus("error");
        }
      };

      socket.onerror = () => {
        setConnectionStatus("error");
      };

      socket.onclose = () => {
        if (isClosedByEffect) {
          return;
        }

        setConnectionStatus("disconnected");
        reconnectTimerRef.current = window.setTimeout(connect, 2500);
      };
    }

    connect();

    return () => {
      isClosedByEffect = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, [onDecision, wsUrl]);

  const sendEyePayload = useCallback((payload: EyePayload) => {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    socket.send(JSON.stringify(payload));
    return true;
  }, []);

  return { connectionStatus, sendEyePayload, wsUrl };
}

function useYawnSocket(onDecision: (decision: ServerDecision) => void) {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const wsUrl = useMemo(() => normalizeWsUrl(YAWN_WS_URL), []);

  useEffect(() => {
    let isClosedByEffect = false;

    function connect() {
      setConnectionStatus("connecting");

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnectionStatus("connected");
      };

      socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(String(event.data));
          const decision = parseServerDecision(parsed);

          if (decision?.status && isYawningStatus(decision.status)) {
            onDecision({
              ...decision,
              level: "warning",
              message: decision.message || "하품 감지"
            });
          }
        } catch {
          setConnectionStatus("error");
        }
      };

      socket.onerror = () => {
        setConnectionStatus("error");
      };

      socket.onclose = () => {
        if (isClosedByEffect) {
          return;
        }

        setConnectionStatus("disconnected");
        reconnectTimerRef.current = window.setTimeout(connect, 2500);
      };
    }

    connect();

    return () => {
      isClosedByEffect = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, [onDecision, wsUrl]);

  return { connectionStatus, wsUrl };
}

function useEyeLandmarks(
  videoRef: RefObject<HTMLVideoElement | null>,
  sendEyePayload: (payload: EyePayload) => boolean
) {
  const lastVideoTimeRef = useRef(-1);
  const lastSentAtRef = useRef(0);
  const [mediapipeStatus, setMediapipeStatus] = useState("눈 포인트 준비 중");
  const [eyePreview, setEyePreview] = useState<EyePayload | null>(null);

  useEffect(() => {
    let landmarker: FaceLandmarker | null = null;
    let animationFrame = 0;
    let isCancelled = false;

    async function setup() {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
        landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: FACE_LANDMARKER_MODEL_URL,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1
        });

        if (!isCancelled) {
          setMediapipeStatus("눈 포인트 감지 중");
          detectLoop();
        }
      } catch {
        if (!isCancelled) {
          setMediapipeStatus("눈 포인트 로드 실패");
        }
      }
    }

    function extractEyePayload(video: HTMLVideoElement, now: number): EyePayload | null {
      if (!landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return null;
      }

      if (video.currentTime === lastVideoTimeRef.current) {
        return null;
      }

      lastVideoTimeRef.current = video.currentTime;
      const result = landmarker.detectForVideo(video, now);
      const faceLandmarks = result.faceLandmarks[0];

      if (!faceLandmarks) {
        return null;
      }

      const toEyePoints = (indices: number[]) =>
        indices.map((index) => {
          const point = faceLandmarks[index];
          return {
            x: point.x,
            y: point.y
          };
        });

      return {
        leftEye: toEyePoints(LEFT_EYE_INDICES),
        rightEye: toEyePoints(RIGHT_EYE_INDICES)
      };
    }

    function detectLoop(now = performance.now()) {
      if (isCancelled) {
        return;
      }

      const video = videoRef.current;
      const payload = video ? extractEyePayload(video, now) : null;

      if (payload) {
        setEyePreview(payload);

        if (now - lastSentAtRef.current > 250) {
          sendEyePayload(payload);
          lastSentAtRef.current = now;
        }
      }

      animationFrame = window.requestAnimationFrame(detectLoop);
    }

    setup();

    return () => {
      isCancelled = true;
      window.cancelAnimationFrame(animationFrame);
      landmarker?.close();
    };
  }, [sendEyePayload, videoRef]);

  return { mediapipeStatus, eyePreview };
}

function useSirenAlarm(active: boolean) {
  const audioContextRef = useRef<AudioContext | null>(null);
  const intervalRef = useRef<number | null>(null);

  const playPulse = useCallback(() => {
    try {
      const audioWindow = window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext };
      const AudioContextCtor = window.AudioContext ?? audioWindow.webkitAudioContext;

      if (!AudioContextCtor) {
        return;
      }

      const context = audioContextRef.current ?? new AudioContextCtor();
      audioContextRef.current = context;

      if (context.state === "suspended") {
        context.resume().catch(() => undefined);
      }

      const now = context.currentTime;
      const masterGain = context.createGain();
      const lowTone = context.createOscillator();
      const highTone = context.createOscillator();

      lowTone.type = "square";
      highTone.type = "sawtooth";

      lowTone.frequency.setValueAtTime(410, now);
      lowTone.frequency.linearRampToValueAtTime(270, now + 0.18);
      lowTone.frequency.linearRampToValueAtTime(520, now + 0.38);
      lowTone.frequency.linearRampToValueAtTime(310, now + 0.62);

      highTone.frequency.setValueAtTime(920, now);
      highTone.frequency.linearRampToValueAtTime(1260, now + 0.18);
      highTone.frequency.linearRampToValueAtTime(720, now + 0.38);
      highTone.frequency.linearRampToValueAtTime(1180, now + 0.62);

      masterGain.gain.setValueAtTime(0.0001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.18, now + 0.04);
      masterGain.gain.setValueAtTime(0.18, now + 0.18);
      masterGain.gain.exponentialRampToValueAtTime(0.04, now + 0.26);
      masterGain.gain.exponentialRampToValueAtTime(0.2, now + 0.34);
      masterGain.gain.setValueAtTime(0.2, now + 0.56);
      masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.74);

      lowTone.connect(masterGain);
      highTone.connect(masterGain);
      masterGain.connect(context.destination);
      lowTone.start(now);
      highTone.start(now);
      lowTone.stop(now + 0.76);
      highTone.stop(now + 0.76);
    } catch {
      // Browsers may block audio until the page receives a user gesture.
    }
  }, []);

  useEffect(() => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (!active) {
      return;
    }

    playPulse();
    intervalRef.current = window.setInterval(playPulse, 820);

    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [active, playPulse]);
}

function useThemeMode() {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }

    return window.localStorage.getItem("lightbox-theme") === "dark";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = isDarkMode ? "dark" : "light";
    window.localStorage.setItem("lightbox-theme", isDarkMode ? "dark" : "light");
  }, [isDarkMode]);

  return { isDarkMode, setIsDarkMode };
}

function App() {
  const { videoRef, cameraStatus } = useCameraPreview();
  const { isDarkMode, setIsDarkMode } = useThemeMode();
  const showDiagnostics = import.meta.env.DEV;
  const lastAlertAtRef = useRef(0);
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [frequency, setFrequency] = useState<FrequencyPoint[]>(emptyFrequency);
  const [toast, setToast] = useState<DetectionEvent | null>(null);
  const [currentDecision, setCurrentDecision] = useState<ServerDecision>({
    isDrowsy: false,
    level: "normal",
    risk: 0,
    message: "서버 판정 대기 중",
    status: "WAITING"
  });

  const handleDecision = useCallback((decision: ServerDecision) => {
    setCurrentDecision(decision);

    if (!decision.isDrowsy && decision.level === "normal") {
      return;
    }

    const now = Date.now();
    if (now - lastAlertAtRef.current < 3000) {
      return;
    }

    lastAlertAtRef.current = now;
    const event: DetectionEvent = {
      id: now,
      time: formatTime(new Date(now)),
      label: decision.message,
      level: decision.level,
      risk: decision.risk
    };

    setEvents((current) => [event, ...current].slice(0, 8));
    setToast(event);
    setFrequency((current) => {
      const label = formatMinute(new Date(now));
      const next = [...current];
      const last = next[next.length - 1];

      if (last?.label === label) {
        next[next.length - 1] = { ...last, count: last.count + 1 };
        return next;
      }

      return [...next.slice(1), { label, count: 1 }];
    });
  }, []);

  const { connectionStatus, sendEyePayload, wsUrl } = useDrowsinessSocket(handleDecision);
  const { connectionStatus: yawnConnectionStatus, wsUrl: yawnWsUrl } = useYawnSocket(handleDecision);
  const { mediapipeStatus, eyePreview } = useEyeLandmarks(videoRef, sendEyePayload);

  const totalEvents = frequency.reduce((sum, point) => sum + point.count, 0);
  const currentLevel = currentDecision.level;
  const currentRisk = currentDecision.risk;
  const isClosedAlert = isClosedEyeStatus(currentDecision.status);
  useSirenAlarm(isClosedAlert);

  const averageFrequency = useMemo(() => {
    if (frequency.length === 0) {
      return 0;
    }

    return Math.round((totalEvents / frequency.length) * 10) / 10;
  }, [frequency, totalEvents]);

  useEffect(() => {
    if (!toast) {
      return;
    }

    const timeout = window.setTimeout(() => setToast(null), 4200);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  return (
    <main className="dashboard">
      <section className="topbar" aria-label="대시보드 요약">
        <div>
          <p className="eyebrow">LightBox Driver Monitor</p>
          <h1>졸음운전 감지 대시보드</h1>
        </div>
        <div className="top-actions">
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setIsDarkMode((current) => !current)}
            aria-label={isDarkMode ? "라이트 모드로 전환" : "다크 모드로 전환"}
            title={isDarkMode ? "라이트 모드" : "다크 모드"}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <div className={`status-pill ${currentLevel}`}>
            <Radio size={18} />
            <span>{levelText[currentLevel]}</span>
          </div>
        </div>
      </section>

      <section className="summary-grid" aria-label="실시간 지표">
        <MetricCard icon={<CircleGauge />} label="현재 위험도" value={`${currentRisk}%`} tone={currentLevel} />
        <MetricCard icon={<Bell />} label="감지 이벤트" value={`${totalEvents}회`} tone="neutral" />
        <MetricCard icon={<Clock3 />} label="평균 빈도" value={`${averageFrequency}회/구간`} tone="neutral" />
        <MetricCard
          icon={<CircleGauge />}
          label="EAR"
          value={currentDecision.earSmooth?.toFixed(3) ?? currentDecision.ear?.toFixed(3) ?? "-"}
          tone="neutral"
        />
        {showDiagnostics && (
          <>
            <MetricCard icon={<Camera />} label="카메라 상태" value={cameraStatus} tone="neutral" />
            <MetricCard
              icon={connectionStatus === "connected" ? <Wifi /> : <WifiOff />}
              label="WebSocket"
              value={connectionText[connectionStatus]}
              tone={connectionStatus === "connected" ? "normal" : connectionStatus === "error" ? "danger" : "neutral"}
            />
            <MetricCard
              icon={yawnConnectionStatus === "connected" ? <Wifi /> : <WifiOff />}
              label="하품 소켓"
              value={connectionText[yawnConnectionStatus]}
              tone={yawnConnectionStatus === "connected" ? "normal" : yawnConnectionStatus === "error" ? "danger" : "neutral"}
            />
            <MetricCard icon={<ScanFace />} label="눈 포인트" value={mediapipeStatus} tone="neutral" />
          </>
        )}
      </section>

      <section className="content-grid">
        <div className="camera-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Live Camera</p>
              <h2>운전자 화면</h2>
            </div>
            <span className={`camera-badge ${cameraStatus === "카메라 연결됨" ? "active" : ""}`}>
              <Video size={16} />
              {cameraStatus}
            </span>
          </div>

          <div className="video-shell">
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="scan-frame" aria-hidden="true">
              <span />
              <span />
              <span />
              <span />
            </div>
            <div className="risk-overlay">
              <span>위험도</span>
              <strong>{currentRisk}%</strong>
            </div>
          </div>
        </div>

        <aside className="alert-panel" aria-label="알림 로그">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Alerts</p>
              <h2>최근 알림</h2>
            </div>
          </div>

          <div className="event-list">
            {events.length === 0 && (
              <article className="event-empty">
                <CheckCircle2 size={20} />
                <span>{currentDecision.message}</span>
              </article>
            )}

            {events.map((event) => (
              <article className={`event-item ${event.level}`} key={event.id}>
                <div className="event-icon">
                  {event.level === "danger" ? <ShieldAlert size={18} /> : <AlertTriangle size={18} />}
                </div>
                <div>
                  <strong>{event.label}</strong>
                  <span>{event.time} · 위험도 {event.risk}%</span>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </section>

      <section className="chart-panel" aria-label="졸음 감지 빈도 그래프">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Detection Frequency</p>
            <h2>졸음 감지 빈도</h2>
          </div>
          <span className="chart-summary">최근 {frequency.length}개 구간</span>
        </div>
        <FrequencyChart data={frequency} />
      </section>

      {showDiagnostics && (
        <section className="payload-panel" aria-label="서버 송신 정보">
          <div>
            <p className="eyebrow">Server Endpoint</p>
            <strong>{wsUrl}</strong>
          </div>
          <div>
            <p className="eyebrow">Yawn Endpoint</p>
            <strong>{yawnWsUrl}</strong>
          </div>
          <div>
            <p className="eyebrow">Eye Points</p>
            <strong>좌/우 눈 각 6포인트 · {eyePreview ? "송신 중" : "얼굴 대기 중"}</strong>
          </div>
          <div>
            <p className="eyebrow">Server Status</p>
            <strong>{currentDecision.status ?? "WAITING"}</strong>
          </div>
        </section>
      )}

      {isClosedAlert && (
        <div className="closed-alarm" role="alert" aria-live="assertive">
          <div className="closed-alarm-icon">
            <Volume2 size={24} />
          </div>
          <div>
            <strong>위웅위웅! 눈 감김 감지</strong>
            <span>운전자 졸음 상태가 감지되었습니다.</span>
          </div>
        </div>
      )}

      {toast && (
        <div className={`toast ${toast.level}`} role="status" aria-live="polite">
          <div className="toast-icon">
            {toast.level === "normal" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          </div>
          <div>
            <strong>{toast.label}</strong>
            <span>{levelDescription[toast.level]}</span>
          </div>
        </div>
      )}
    </main>
  );
}

function MetricCard({
  icon,
  label,
  value,
  tone
}: {
  icon: ReactNode;
  label: string;
  value: string;
  tone: AlertLevel | "neutral";
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </article>
  );
}

function FrequencyChart({ data }: { data: FrequencyPoint[] }) {
  const max = Math.max(...data.map((point) => point.count), 1);
  const width = 900;
  const height = 260;
  const padding = { top: 24, right: 24, bottom: 42, left: 42 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const points = data.map((point, index) => {
    const x = padding.left + (chartWidth / Math.max(data.length - 1, 1)) * index;
    const y = padding.top + chartHeight - (point.count / max) * chartHeight;
    return { ...point, x, y };
  });
  const areaPath = [
    `M ${padding.left} ${padding.top + chartHeight}`,
    ...points.map((point) => `L ${point.x} ${point.y}`),
    `L ${padding.left + chartWidth} ${padding.top + chartHeight}`,
    "Z"
  ].join(" ");
  const linePath = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="시간대별 졸음 감지 빈도 선 그래프">
        <defs>
          <linearGradient id="frequencyFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.42" />
            <stop offset="100%" stopColor="#14b8a6" stopOpacity="0.08" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3, 4].map((tick) => {
          const y = padding.top + (chartHeight / 4) * tick;
          return <line className="grid-line" key={tick} x1={padding.left} x2={padding.left + chartWidth} y1={y} y2={y} />;
        })}
        <path className="area-path" d={areaPath} />
        <path className="line-path" d={linePath} />
        {points.map((point) => (
          <g key={`${point.label}-${point.x}`}>
            <circle className="point-dot" cx={point.x} cy={point.y} r="5" />
            <text className="x-label" x={point.x} y={height - 14} textAnchor="middle">
              {point.label}
            </text>
          </g>
        ))}
        {[0, Math.ceil(max / 2), max].map((tick) => {
          const y = padding.top + chartHeight - (tick / max) * chartHeight;
          return (
            <text className="y-label" key={tick} x={18} y={y + 4}>
              {tick}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

export { App };
