import {
  AlertTriangle,
  Bell,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  CircleGauge,
  Clock3,
  LogIn,
  LogOut,
  Moon,
  Radio,
  ShieldCheck,
  Sun,
  UserPlus,
  Video,
  Volume2,
  Wifi,
  WifiOff,
} from "lucide-react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { FormEvent, ReactNode, RefObject } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import logoUrl from "../logo.png";

type AlertLevel = "normal" | "warning" | "danger";
type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";
type AuthMode = "login" | "signup";
type AuthStatus = "checking" | "signedOut" | "signedIn";

type EyePoint = {
  x: number;
  y: number;
};

type EyeKeypoints = {
  leftEye: EyePoint[];
  rightEye: EyePoint[];
};

type DeviceType = "phone" | "mobile" | "laptop" | "desktop";

type EyePayload = {
  device: DeviceType;
  browser: string;
  userAgent: string;
  keypoints: EyeKeypoints;
  leftEye: EyePoint[];
  rightEye: EyePoint[];
};

type DetectionEvent = {
  id: number;
  label: string;
  level: AlertLevel;
  risk: number;
  status?: string;
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

type AuthUser = {
  id: number;
  username: string;
  nickname: string;
};

type AuthSession = {
  accessToken: string;
  tokenType: string;
  expiresAt: string;
  user: AuthUser;
};

type LoginResponse = {
  ok: boolean;
  access_token: string;
  token_type: string;
  expires_at: string;
  user: AuthUser;
};

type SignupResponse = {
  ok: boolean;
  user: AuthUser;
};

type MeResponse = {
  ok: boolean;
  user: AuthUser;
};

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "" : "https://spoti.ingyuc.click")
).replace(/\/$/, "");
const USERNAME_PATTERN = /^[A-Za-z0-9_.-]{3,32}$/;
const DROWSINESS_WS_URL =
  import.meta.env.VITE_DROWSINESS_WS_URL ?? "wss://spoti.ingyuc.click/ws/keypoints";
const YAWN_WS_URL = import.meta.env.VITE_YAWN_WS_URL ?? "wss://spoti.ingyuc.click/ws/yawn";
const YAWN_FRAME_INTERVAL_MS = Math.max(1000, Number(import.meta.env.VITE_YAWN_FRAME_INTERVAL_MS) || 1500);
const YAWN_FRAME_MAX_WIDTH = 480;
const YAWN_FRAME_QUALITY = 0.75;
const MEDIAPIPE_WASM_URL =
  import.meta.env.VITE_MEDIAPIPE_WASM_URL ?? "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const FACE_LANDMARKER_MODEL_URL =
  import.meta.env.VITE_FACE_LANDMARKER_MODEL_URL ??
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const AUTH_STORAGE_KEY = "lightbox-auth-session";

const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const OPEN_EAR_SAFE = 0.34;
const OPEN_EAR_WARNING = 0.22;
const OPEN_RISK_MAX = 70;
const CLOSED_RISK_BASE = 90;
const CLOSED_EAR_REFERENCE = 0.22;
const CLOSED_EAR_MIN = 0.12;

const levelText: Record<AlertLevel, string> = {
  normal: "정상",
  warning: "주의",
  danger: "위험"
};

const levelDescription: Record<AlertLevel, string> = {
  normal: "운전자 상태가 안정적입니다.",
  warning: "졸음 징후가 감지되고 있습니다.",
  danger: "즉시 경고가 필요한 상태입니다."
};

const compactConnectionText: Record<ConnectionStatus, string> = {
  connecting: "연결 중",
  connected: "연결됨",
  disconnected: "끊김",
  error: "오류"
};

type ApiRequestOptions = RequestInit & {
  token?: string;
};

function readStoredAuthSession(): AuthSession | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const rawSession = window.localStorage.getItem(AUTH_STORAGE_KEY);

    if (!rawSession) {
      return null;
    }

    const parsed = JSON.parse(rawSession) as Partial<AuthSession>;

    if (
      typeof parsed.accessToken !== "string" ||
      typeof parsed.tokenType !== "string" ||
      typeof parsed.expiresAt !== "string" ||
      !parsed.user
    ) {
      return null;
    }

    return parsed as AuthSession;
  } catch {
    return null;
  }
}

function persistAuthSession(session: AuthSession | null) {
  if (typeof window === "undefined") {
    return;
  }

  if (!session) {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }

  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
}

function createAuthSession(response: LoginResponse): AuthSession {
  return {
    accessToken: response.access_token,
    tokenType: response.token_type,
    expiresAt: response.expires_at,
    user: response.user
  };
}

function toFriendlyApiError(message: string) {
  const normalizedMessage = message.toLowerCase();

  if (normalizedMessage.includes("username must be")) {
    return "아이디는 영문, 숫자, 점, 하이픈, 밑줄만 사용해 3-32자로 입력해 주세요.";
  }

  if (normalizedMessage.includes("nickname must be")) {
    return "닉네임은 2-30자로 입력해 주세요.";
  }

  if (normalizedMessage.includes("password must be")) {
    return "비밀번호는 8자 이상으로 입력해 주세요.";
  }

  if (normalizedMessage.includes("password confirmation")) {
    return "비밀번호 확인이 일치하지 않습니다.";
  }

  return message;
}

async function readApiError(response: Response) {
  try {
    const data = (await response.json()) as { detail?: unknown; message?: unknown };

    if (Array.isArray(data.detail)) {
      const firstError = data.detail.find(
        (item): item is { msg: string } => Boolean(item) && typeof item === "object" && "msg" in item && typeof item.msg === "string"
      );

      if (firstError) {
        return toFriendlyApiError(firstError.msg);
      }
    }

    if (typeof data.detail === "string") {
      return toFriendlyApiError(data.detail);
    }

    if (typeof data.message === "string") {
      return toFriendlyApiError(data.message);
    }
  } catch {
    // Fall back to status based messages below.
  }

  if (response.status === 401) {
    return "아이디 또는 비밀번호를 확인해 주세요.";
  }

  if (response.status === 409) {
    return "이미 존재하는 아이디입니다.";
  }

  return "요청 처리 중 문제가 발생했습니다.";
}

async function apiRequest<T>(path: string, options: ApiRequestOptions = {}) {
  const { token, headers, body, ...init } = options;
  const requestHeaders = new Headers(headers);

  if (body && !requestHeaders.has("Content-Type")) {
    requestHeaders.set("Content-Type", "application/json");
  }

  if (token) {
    requestHeaders.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    body,
    headers: requestHeaders
  });

  if (!response.ok) {
    throw new Error(await readApiError(response));
  }

  return (await response.json()) as T;
}

function useAuthSession() {
  const [session, setSessionState] = useState<AuthSession | null>(() => readStoredAuthSession());
  const [authStatus, setAuthStatus] = useState<AuthStatus>(() => (readStoredAuthSession() ? "checking" : "signedOut"));

  const setSession = useCallback((nextSession: AuthSession | null) => {
    persistAuthSession(nextSession);
    setSessionState(nextSession);
    setAuthStatus(nextSession ? "signedIn" : "signedOut");
  }, []);

  useEffect(() => {
    const token = session?.accessToken;

    if (!token) {
      setAuthStatus("signedOut");
      return;
    }

    let isCancelled = false;
    setAuthStatus("checking");

    apiRequest<MeResponse>("/auth/me", { token })
      .then((response) => {
        if (isCancelled) {
          return;
        }

        setSessionState((current) => {
          if (!current) {
            return current;
          }

          const nextSession = { ...current, user: response.user };
          persistAuthSession(nextSession);
          return nextSession;
        });
        setAuthStatus("signedIn");
      })
      .catch(() => {
        if (!isCancelled) {
          setSession(null);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [session?.accessToken, setSession]);

  const login = useCallback(
    async (username: string, password: string) => {
      const response = await apiRequest<LoginResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password })
      });

      setSession(createAuthSession(response));
    },
    [setSession]
  );

  const signup = useCallback(
    async (username: string, nickname: string, password: string, passwordConfirm: string) => {
      await apiRequest<SignupResponse>("/auth/signup", {
        method: "POST",
        body: JSON.stringify({ username, nickname, password, passwordConfirm })
      });
      await login(username, password);
    },
    [login]
  );

  const logout = useCallback(async () => {
    const token = session?.accessToken;
    setSession(null);

    if (!token) {
      return;
    }

    try {
      await apiRequest<{ ok: boolean }>("/auth/logout", {
        method: "POST",
        token
      });
    } catch {
      // The local session is already cleared, so a stale server token should not block logout.
    }
  }, [session?.accessToken, setSession]);

  return { authStatus, login, logout, session, signup };
}

function normalizeWsUrl(url: string) {
  if (url.startsWith("https://")) {
    return url.replace("https://", "wss://");
  }

  if (url.startsWith("http://")) {
    return url.replace("http://", "ws://");
  }

  return url;
}

function getBrowserName(userAgent: string) {
  if (/Edg\//.test(userAgent)) {
    return "Edge";
  }

  if (/SamsungBrowser\//.test(userAgent)) {
    return "Samsung Internet";
  }

  if (/CriOS\/|Chrome\//.test(userAgent) && !/Edg\//.test(userAgent)) {
    return "Chrome";
  }

  if (/FxiOS\/|Firefox\//.test(userAgent)) {
    return "Firefox";
  }

  if (/Safari\//.test(userAgent) && /Version\//.test(userAgent)) {
    return "Safari";
  }

  return "Unknown";
}

function getDeviceType(userAgent: string): DeviceType {
  const normalized = userAgent.toLowerCase();
  const touchPoints = navigator.maxTouchPoints ?? 0;

  if (/iphone|android.*mobile|mobile|phone|galaxy|pixel/.test(normalized)) {
    return "phone";
  }

  if (/ipad|tablet|ios|android/.test(normalized) || (normalized.includes("macintosh") && touchPoints > 1)) {
    return "mobile";
  }

  if (/macintosh|mac os|windows|linux/.test(normalized)) {
    return "laptop";
  }

  return "desktop";
}

function createEyePayload(keypoints: EyeKeypoints): EyePayload {
  const userAgent = navigator.userAgent;

  return {
    device: getDeviceType(userAgent),
    browser: getBrowserName(userAgent),
    userAgent,
    keypoints,
    leftEye: keypoints.leftEye,
    rightEye: keypoints.rightEye
  };
}

function readNumber(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const parsed = Number(value);

      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return undefined;
}

function formatMinute(date: Date) {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  }).format(date);
}

function createEmptyFrequency(now = new Date()): FrequencyPoint[] {
  return Array.from({ length: 10 }, (_, index) => {
    const date = new Date(now);
    date.setSeconds(0, 0);
    date.setMinutes(date.getMinutes() - (9 - index));

    return {
      label: formatMinute(date),
      count: 0
    };
  });
}

let sharedAudioContext: AudioContext | null = null;

function getSharedAudioContext() {
  const audioWindow = window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext };
  const AudioContextCtor = window.AudioContext ?? audioWindow.webkitAudioContext;

  if (!AudioContextCtor) {
    return null;
  }

  sharedAudioContext = sharedAudioContext ?? new AudioContextCtor();
  return sharedAudioContext;
}

function unlockSharedAudio() {
  const context = getSharedAudioContext();

  if (!context) {
    return;
  }

  if (context.state === "suspended") {
    context.resume().catch(() => undefined);
  }

  const now = context.currentTime;
  const gain = context.createGain();
  const oscillator = context.createOscillator();

  gain.gain.setValueAtTime(0.0001, now);
  oscillator.frequency.setValueAtTime(440, now);
  oscillator.connect(gain);
  gain.connect(context.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.03);
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

function isYawningLabel(value: unknown) {
  if (typeof value !== "string") {
    return false;
  }

  const normalized = value.trim().toUpperCase();
  return isYawningStatus(normalized) || normalized === "하품";
}

function findYawnDetection(value: unknown) {
  if (!Array.isArray(value)) {
    return null;
  }

  for (const detection of value) {
    if (!detection || typeof detection !== "object") {
      continue;
    }

    const data = detection as Record<string, unknown>;
    const label = data.label ?? data.class ?? data.name ?? data.event;

    if (isYawningLabel(label)) {
      return {
        label: String(label),
        confidence: data.confidence ?? data.score ?? data.probability
      };
    }
  }

  return null;
}

function captureVideoJpeg(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  if (
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    video.videoWidth === 0 ||
    video.videoHeight === 0
  ) {
    return Promise.resolve(null);
  }

  const scale = Math.min(1, YAWN_FRAME_MAX_WIDTH / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);

  const context = canvas.getContext("2d");
  if (!context) {
    return Promise.resolve(null);
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", YAWN_FRAME_QUALITY);
  });
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
  const yawnDetection = findYawnDetection(data.detections);
  const hasYawnBoolean =
    data.yawning === true || data.is_yawning === true || data.isYawning === true || data.yawn === true;
  const statusValue =
    hasYawnBoolean || yawnDetection
      ? "YAWNING"
      : data.status ?? data.state ?? data.label ?? data.event ?? data.class;
  const status = typeof statusValue === "string" ? statusValue.toUpperCase() : undefined;
  const ear = readNumber(data.ear, data.ear_avg, data.earAverage, data.avg_ear, data.left_ear, data.ear_left, data.right_ear, data.ear_right);
  const earSmooth = readNumber(data.ear_smooth, data.earSmooth, data.smooth_ear, data.ear_smoothed);
  const rawYawnConfidence = data.yawn_confidence ?? data.yawnConfidence ?? data.confidence ?? yawnDetection?.confidence;
  const yawnConfidence = typeof rawYawnConfidence === "number" ? rawYawnConfidence : undefined;
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

  const sendEyePayload = useCallback((keypoints: EyeKeypoints) => {
    const socket = socketRef.current;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return false;
    }

    socket.send(JSON.stringify(createEyePayload(keypoints)));
    return true;
  }, []);

  return { connectionStatus, sendEyePayload, wsUrl };
}

function useYawnSocket(
  videoRef: RefObject<HTMLVideoElement | null>,
  onDecision: (decision: ServerDecision) => void
) {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const isSendingFrameRef = useRef(false);
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

  useEffect(() => {
    const canvas = document.createElement("canvas");
    let isCancelled = false;

    async function sendYawnFrame() {
      const socket = socketRef.current;
      const video = videoRef.current;

      if (
        isCancelled ||
        isSendingFrameRef.current ||
        !video ||
        !socket ||
        socket.readyState !== WebSocket.OPEN
      ) {
        return;
      }

      isSendingFrameRef.current = true;

      try {
        const frame = await captureVideoJpeg(video, canvas);

        if (!isCancelled && frame && socket.readyState === WebSocket.OPEN) {
          socket.send(frame);
        }
      } finally {
        isSendingFrameRef.current = false;
      }
    }

    const interval = window.setInterval(sendYawnFrame, YAWN_FRAME_INTERVAL_MS);
    void sendYawnFrame();

    return () => {
      isCancelled = true;
      window.clearInterval(interval);
    };
  }, [videoRef]);

  return { connectionStatus, wsUrl };
}

function useEyeLandmarks(
  videoRef: RefObject<HTMLVideoElement | null>,
  sendEyePayload: (payload: EyeKeypoints) => boolean
) {
  const lastVideoTimeRef = useRef(-1);
  const lastSentAtRef = useRef(0);
  const [mediapipeStatus, setMediapipeStatus] = useState("눈 포인트 준비 중");
  const [eyePreview, setEyePreview] = useState<EyeKeypoints | null>(null);

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

    function extractEyePayload(video: HTMLVideoElement, now: number): EyeKeypoints | null {
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
  const intervalRef = useRef<number | null>(null);

  const playPulse = useCallback(() => {
    try {
      const context = getSharedAudioContext();

      if (!context) {
        return;
      }

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

function useYawnChime(triggerId: number | null) {
  const lastPlayedIdRef = useRef<number | null>(null);

  const playChime = useCallback(() => {
    try {
      const context = getSharedAudioContext();

      if (!context) {
        return;
      }

      if (context.state === "suspended") {
        context.resume().catch(() => undefined);
      }

      const now = context.currentTime;
      const masterGain = context.createGain();
      const firstTone = context.createOscillator();
      const secondTone = context.createOscillator();

      firstTone.type = "sine";
      secondTone.type = "sine";
      firstTone.frequency.setValueAtTime(660, now);
      secondTone.frequency.setValueAtTime(880, now + 0.16);

      masterGain.gain.setValueAtTime(0.0001, now);
      masterGain.gain.exponentialRampToValueAtTime(0.075, now + 0.04);
      masterGain.gain.exponentialRampToValueAtTime(0.025, now + 0.18);
      masterGain.gain.exponentialRampToValueAtTime(0.085, now + 0.22);
      masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.58);

      firstTone.connect(masterGain);
      secondTone.connect(masterGain);
      masterGain.connect(context.destination);
      firstTone.start(now);
      firstTone.stop(now + 0.2);
      secondTone.start(now + 0.17);
      secondTone.stop(now + 0.6);
    } catch {
      // Browsers may block audio until the page receives a user gesture.
    }
  }, []);

  useEffect(() => {
    if (triggerId === null || lastPlayedIdRef.current === triggerId) {
      return;
    }

    lastPlayedIdRef.current = triggerId;
    playChime();
  }, [playChime, triggerId]);
}

function useAudioUnlock() {
  useEffect(() => {
    const unlock = () => unlockSharedAudio();
    const options = { passive: true };

    window.addEventListener("pointerdown", unlock, options);
    window.addEventListener("touchstart", unlock, options);
    window.addEventListener("keydown", unlock);

    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("touchstart", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);
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

function Dashboard({
  isDarkMode,
  onLogout,
  setIsDarkMode,
  user
}: {
  isDarkMode: boolean;
  onLogout: () => void;
  setIsDarkMode: (updater: (current: boolean) => boolean) => void;
  user: AuthUser;
}) {
  const { videoRef, cameraStatus } = useCameraPreview();
  const lastAlertAtRef = useRef(0);
  const [isSummaryOpen, setIsSummaryOpen] = useState(() => {
    if (typeof window === "undefined") {
      return true;
    }

    return window.localStorage.getItem("lightbox-summary") !== "collapsed";
  });
  const [isChartOpen, setIsChartOpen] = useState(() => {
    if (typeof window === "undefined") {
      return true;
    }

    return window.localStorage.getItem("lightbox-chart") !== "collapsed";
  });
  const [frequency, setFrequency] = useState<FrequencyPoint[]>(() => createEmptyFrequency());
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
      label: decision.message,
      level: decision.level,
      risk: decision.risk,
      status: decision.status
    };

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

  const { connectionStatus, sendEyePayload } = useDrowsinessSocket(handleDecision);
  const { connectionStatus: yawnConnectionStatus } = useYawnSocket(videoRef, handleDecision);
  useEyeLandmarks(videoRef, sendEyePayload);

  const totalEvents = frequency.reduce((sum, point) => sum + point.count, 0);
  const currentLevel = currentDecision.level;
  const currentRisk = currentDecision.risk;
  const isClosedAlert = isClosedEyeStatus(currentDecision.status);
  const yawnChimeTriggerId = toast?.status && isYawningStatus(toast.status) ? toast.id : null;
  useAudioUnlock();
  useSirenAlarm(isClosedAlert);
  useYawnChime(yawnChimeTriggerId);

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

  useEffect(() => {
    window.localStorage.setItem("lightbox-summary", isSummaryOpen ? "expanded" : "collapsed");
  }, [isSummaryOpen]);

  useEffect(() => {
    window.localStorage.setItem("lightbox-chart", isChartOpen ? "expanded" : "collapsed");
  }, [isChartOpen]);

  useEffect(() => {
    const updateFrequencyWindow = () => {
      setFrequency((current) => {
        const next = createEmptyFrequency();

        return next.map((point) => ({
          ...point,
          count: current.find((currentPoint) => currentPoint.label === point.label)?.count ?? 0
        }));
      });
    };

    const interval = window.setInterval(updateFrequencyWindow, 15000);
    updateFrequencyWindow();

    return () => window.clearInterval(interval);
  }, []);

  const isCompactDashboard = !isSummaryOpen && !isChartOpen;

  return (
    <main className={`dashboard ${isCompactDashboard ? "compact-dashboard" : ""}`}>
      <section className="topbar" aria-label="대시보드 요약">
        <div className="brand-heading">
          <h1>
            <img src={logoUrl} alt="" aria-hidden="true" />
            <span>LightBox</span>
          </h1>
        </div>
        <div className="top-actions">
          <span className="user-badge" title={`${user.nickname}님으로 로그인됨`}>
            <ShieldCheck size={15} />
            <span>{user.nickname}</span>
          </span>
          <SocketStatusBadge label="눈" status={connectionStatus} />
          <SocketStatusBadge label="하품" status={yawnConnectionStatus} />
          <button
            className="summary-toggle"
            type="button"
            onClick={() => setIsSummaryOpen((current) => !current)}
            aria-controls="summary-panel"
            aria-expanded={isSummaryOpen}
            title={isSummaryOpen ? "상단 지표 접기" : "상단 지표 펼치기"}
          >
            {isSummaryOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            <span>{isSummaryOpen ? "지표 접기" : "지표 펼치기"}</span>
          </button>
          <button
            className="theme-toggle"
            type="button"
            onClick={() => setIsDarkMode((current) => !current)}
            aria-label={isDarkMode ? "라이트 모드로 전환" : "다크 모드로 전환"}
            title={isDarkMode ? "라이트 모드" : "다크 모드"}
          >
            {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
          <button
            className="theme-toggle"
            type="button"
            onClick={onLogout}
            aria-label="로그아웃"
            title="로그아웃"
          >
            <LogOut size={18} />
          </button>
          <div className={`status-pill ${currentLevel}`}>
            <Radio size={18} />
            <span>{levelText[currentLevel]}</span>
          </div>
        </div>
      </section>

      <div
        className={`summary-collapse ${isSummaryOpen ? "open" : "closed"}`}
        id="summary-panel"
        aria-hidden={!isSummaryOpen}
      >
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
        </section>
      </div>

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

      </section>

      <section className="chart-panel" aria-label="졸음 감지 빈도 그래프">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Detection Frequency</p>
            <h2>졸음 감지 빈도</h2>
          </div>
          <div className="chart-actions">
            <span className="chart-summary">최근 {frequency.length}개 구간</span>
            <button
              className="panel-toggle"
              type="button"
              onClick={() => setIsChartOpen((current) => !current)}
              aria-controls="frequency-chart-panel"
              aria-expanded={isChartOpen}
              title={isChartOpen ? "그래프 접기" : "그래프 펼치기"}
            >
              {isChartOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              <span>{isChartOpen ? "접기" : "펼치기"}</span>
            </button>
          </div>
        </div>
        <div
          className={`chart-collapse ${isChartOpen ? "open" : "closed"}`}
          id="frequency-chart-panel"
          aria-hidden={!isChartOpen}
        >
          <FrequencyChart data={frequency} />
        </div>
      </section>

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

function AuthLoadingPage({
  isDarkMode,
  setIsDarkMode
}: {
  isDarkMode: boolean;
  setIsDarkMode: (updater: (current: boolean) => boolean) => void;
}) {
  return (
    <main className="auth-page">
      <section className="auth-topbar" aria-label="로그인 상단 메뉴">
        <div className="brand-heading">
          <h1>
            <img src={logoUrl} alt="" aria-hidden="true" />
            <span>LightBox</span>
          </h1>
        </div>
        <button
          className="theme-toggle"
          type="button"
          onClick={() => setIsDarkMode((current) => !current)}
          aria-label={isDarkMode ? "라이트 모드로 전환" : "다크 모드로 전환"}
          title={isDarkMode ? "라이트 모드" : "다크 모드"}
        >
          {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </section>
      <section className="auth-layout auth-layout-centered">
        <div className="auth-card auth-loading-card" role="status" aria-live="polite">
          <div className="auth-loading-icon">
            <ShieldCheck size={22} />
          </div>
          <div>
            <p className="eyebrow">Account</p>
            <h2>저장된 세션을 확인하는 중입니다.</h2>
          </div>
        </div>
      </section>
    </main>
  );
}

function LoginPage({
  authStatus,
  isDarkMode,
  login,
  setIsDarkMode,
  signup
}: {
  authStatus: AuthStatus;
  isDarkMode: boolean;
  login: (username: string, password: string) => Promise<void>;
  setIsDarkMode: (updater: (current: boolean) => boolean) => void;
  signup: (username: string, nickname: string, password: string, passwordConfirm: string) => Promise<void>;
}) {
  const [mode, setMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [nickname, setNickname] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isSignupMode = mode === "signup";
  const isChecking = authStatus === "checking";

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    setError("");
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError("");

    const trimmedUsername = username.trim();
    const trimmedNickname = nickname.trim();

    if (!USERNAME_PATTERN.test(trimmedUsername)) {
      setError("아이디는 영문, 숫자, 점, 하이픈, 밑줄만 사용해 3-32자로 입력해 주세요.");
      return;
    }

    if (isSignupMode && (trimmedNickname.length < 2 || trimmedNickname.length > 30)) {
      setError("닉네임은 2-30자로 입력해 주세요.");
      return;
    }

    if (isSignupMode && password !== passwordConfirm) {
      setError("비밀번호 확인이 일치하지 않습니다.");
      return;
    }

    setIsSubmitting(true);

    try {
      if (isSignupMode) {
        await signup(trimmedUsername, trimmedNickname, password, passwordConfirm);
        return;
      }

      await login(trimmedUsername, password);
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : "로그인 요청에 실패했습니다.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-topbar" aria-label="로그인 상단 메뉴">
        <div className="brand-heading">
          <h1>
            <img src={logoUrl} alt="" aria-hidden="true" />
            <span>LightBox</span>
          </h1>
        </div>
        <button
          className="theme-toggle"
          type="button"
          onClick={() => setIsDarkMode((current) => !current)}
          aria-label={isDarkMode ? "라이트 모드로 전환" : "다크 모드로 전환"}
          title={isDarkMode ? "라이트 모드" : "다크 모드"}
        >
          {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </section>

      <section className="auth-layout">
        <div className="auth-copy">
          <p className="eyebrow">Driver Safety Console</p>
          <h2>운전자 상태를 로그인 후 바로 모니터링하세요.</h2>
          <p>
            서버 인증 토큰으로 세션을 유지하고, LightBox 대시보드에서 눈 감김과 하품 감지 상태를 이어서 확인합니다.
          </p>
          <div className="auth-feature-grid" aria-label="인증 기능 요약">
            <span>
              <ShieldCheck size={16} /> 토큰 세션
            </span>
            <span>
              <Video size={16} /> 실시간 카메라
            </span>
            <span>
              <Bell size={16} /> 위험 알림
            </span>
          </div>
        </div>

        <form className="auth-card" onSubmit={handleSubmit}>
          <div className="auth-card-heading">
            <div>
              <p className="eyebrow">Account</p>
              <h2>{isSignupMode ? "회원가입" : "로그인"}</h2>
            </div>
            <div className="auth-mode-toggle" role="tablist" aria-label="인증 방식">
              <button
                type="button"
                className={mode === "login" ? "active" : ""}
                onClick={() => switchMode("login")}
                role="tab"
                aria-selected={mode === "login"}
              >
                로그인
              </button>
              <button
                type="button"
                className={mode === "signup" ? "active" : ""}
                onClick={() => switchMode("signup")}
                role="tab"
                aria-selected={mode === "signup"}
              >
                가입
              </button>
            </div>
          </div>

          {isChecking && <div className="auth-message">저장된 세션을 확인하는 중입니다.</div>}
          {error && <div className="auth-message error">{error}</div>}

          <label className="auth-field">
            <span>아이디</span>
            <input
              autoComplete="username"
              minLength={3}
              maxLength={32}
              name="username"
              onChange={(event) => setUsername(event.target.value)}
              pattern="[A-Za-z0-9_.-]{3,32}"
              placeholder="lightbox"
              required
              type="text"
              value={username}
            />
          </label>

          {isSignupMode && (
            <label className="auth-field">
              <span>닉네임</span>
              <input
                autoComplete="nickname"
                minLength={2}
                maxLength={30}
                name="nickname"
                onChange={(event) => setNickname(event.target.value)}
                placeholder="운전자"
                required
                type="text"
                value={nickname}
              />
            </label>
          )}

          <label className="auth-field">
            <span>비밀번호</span>
            <input
              autoComplete={isSignupMode ? "new-password" : "current-password"}
              minLength={8}
              name="password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="8자 이상"
              required
              type="password"
              value={password}
            />
          </label>

          {isSignupMode && (
            <label className="auth-field">
              <span>비밀번호 확인</span>
              <input
                autoComplete="new-password"
                minLength={8}
                name="passwordConfirm"
                onChange={(event) => setPasswordConfirm(event.target.value)}
                placeholder="비밀번호 재입력"
                required
                type="password"
                value={passwordConfirm}
              />
            </label>
          )}

          <button className="auth-submit" disabled={isSubmitting || isChecking} type="submit">
            {isSignupMode ? <UserPlus size={18} /> : <LogIn size={18} />}
            {isSubmitting ? "처리 중" : isSignupMode ? "가입하고 시작" : "로그인하고 시작"}
          </button>
        </form>
      </section>
    </main>
  );
}

function App() {
  const { isDarkMode, setIsDarkMode } = useThemeMode();
  const { authStatus, login, logout, session, signup } = useAuthSession();

  if (authStatus === "checking") {
    return <AuthLoadingPage isDarkMode={isDarkMode} setIsDarkMode={setIsDarkMode} />;
  }

  if (!session) {
    return (
      <LoginPage
        authStatus={authStatus}
        isDarkMode={isDarkMode}
        login={login}
        setIsDarkMode={setIsDarkMode}
        signup={signup}
      />
    );
  }

  return (
    <Dashboard
      isDarkMode={isDarkMode}
      onLogout={() => void logout()}
      setIsDarkMode={setIsDarkMode}
      user={session.user}
    />
  );
}

function SocketStatusBadge({ label, status }: { label: string; status: ConnectionStatus }) {
  const isConnected = status === "connected";

  return (
    <span className={`socket-badge ${status}`} title={`${label} 소켓 ${compactConnectionText[status]}`}>
      {isConnected ? <Wifi size={15} /> : <WifiOff size={15} />}
      <span>{label}</span>
      <strong>{compactConnectionText[status]}</strong>
    </span>
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
