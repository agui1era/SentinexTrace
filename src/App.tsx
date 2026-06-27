import VisionPanel from './lib/components/Vision/index';
import { type Settings } from './types';

const CAMERA_ENV_PREFIXES = ['VITE_RTSP_URL_', 'VITE_WEBCAM_URL_', 'VITE_CAMERA_URL_'];

const cameraUrlMap = new Map<string, string>();
for (const [key, value] of Object.entries(import.meta.env)) {
  if (typeof value !== 'string' || value.trim() === '') {
    continue;
  }

  const camId = cameraIdFromEnvKey(key);
  if (!camId) {
    continue;
  }

  cameraUrlMap.set(camId, value.trim());
}

const cameraUrls = Array.from(cameraUrlMap, ([id, url]) => ({ id, url }));
cameraUrls.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

// Fallback if no specific CAM variables are found
if (cameraUrls.length === 0 && import.meta.env.VITE_RTSP_URL) {
  cameraUrls.push({ id: 'CAM_MAIN', url: import.meta.env.VITE_RTSP_URL });
}

const HUMAN_DETECTION_THRESHOLD = envNumber(
  'VITE_OPENCV_HUMAN_THRESHOLD',
  envNumber('VITE_HUMAN_CONFIDENCE_THRESHOLD', 0.45),
);
const EMBEDDING_MATCH_THRESHOLD = envNumber(
  'VITE_EMBEDDING_MATCH_THRESHOLD',
  envNumber(
    'VITE_PATTERN_MATCH_THRESHOLD',
    envNumber('VITE_PATTERN_REUSE_THRESHOLD', envNumber('VITE_CONFIDENCE_THRESHOLD', 0.6)),
  ),
);

const DEFAULT_SETTINGS: Settings = {
  confidenceThreshold: EMBEDDING_MATCH_THRESHOLD,
  humanConfidenceThreshold: HUMAN_DETECTION_THRESHOLD,
  patternReuseThreshold: EMBEDDING_MATCH_THRESHOLD,
  sourceMode: (import.meta.env.VITE_SOURCE_MODE as 'webcam' | 'rtsp') || 'rtsp',
  rtspUrl: import.meta.env.VITE_RTSP_URL || '',
  cameras: cameraUrls,
};

function App() {
  const settings = DEFAULT_SETTINGS;

  return (
    <main className="app-container vision-only-mode">
      <VisionPanel settings={settings} />
    </main>
  );
}

function cameraIdFromEnvKey(key: string): string {
  const prefix = CAMERA_ENV_PREFIXES.find((candidate) => key.startsWith(candidate));
  if (prefix) {
    return normalizeCameraId(key.slice(prefix.length));
  }

  const suffixMatch = key.match(/^VITE_(CAM[A-Z0-9_-]+|WEBCAM[A-Z0-9_-]*)_(?:RTSP_)?URL$/i);
  if (suffixMatch) {
    return normalizeCameraId(suffixMatch[1]);
  }

  return '';
}

function normalizeCameraId(rawId: string): string {
  const cleaned = rawId.trim().replace(/[^a-z0-9_-]/gi, '').toUpperCase();
  if (!cleaned) {
    return '';
  }
  if (/^\d+$/.test(cleaned)) {
    return `CAM${cleaned}`;
  }
  if (cleaned.startsWith('CAM') || cleaned.startsWith('WEBCAM')) {
    return cleaned;
  }
  return `CAM${cleaned}`;
}

function envNumber(key: string, fallback: number): number {
  const value = Number(import.meta.env[key]);
  return Number.isFinite(value) ? value : fallback;
}

export default App;
