import React, { useState, useEffect } from 'react';
import VisionPanel from './lib/components/Vision/index';
import { type Settings } from './types';

const CAMERA_ENV_PREFIXES = ['VITE_RTSP_URL_', 'VITE_WEBCAM_URL_', 'VITE_CAMERA_URL_'];

function App() {
  const [settings, setSettings] = useState<Settings | null>(null);

  useEffect(() => {
    let active = true;

    async function loadConfig() {
      const fallbackSettings = buildSettings(import.meta.env as Record<string, unknown>);
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 1500);

      try {
        const res = await fetch('/api/vision/config', { signal: controller.signal });
        if (!res.ok) throw new Error(`Config HTTP ${res.status}`);
        const data = await res.json();
        const env = data.config || {};
        if (active) setSettings(buildSettings(env));
      } catch (e) {
        console.error('Failed to load config', e);
        if (active) setSettings(fallbackSettings);
      } finally {
        window.clearTimeout(timeout);
      }
    }

    loadConfig();
    return () => {
      active = false;
    };
  }, []);

  if (!settings) {
    return (
      <div style={{ display: 'grid', placeItems: 'center', height: '100vh', background: '#f5f8fa' }}>
        <p>Cargando configuraciones del sistema...</p>
      </div>
    );
  }

  return (
    <main className="app-container vision-only-mode">
      <VisionPanel settings={settings} />
    </main>
  );
}

function buildSettings(env: Record<string, unknown>): Settings {
  const cameraUrlMap = new Map<string, string>();
  for (const [key, value] of Object.entries(env)) {
    if (typeof value !== 'string' || value.trim() === '') continue;

    const camId = cameraIdFromEnvKey(key);
    if (camId) cameraUrlMap.set(camId, value.trim());
  }

  const cameraUrls = Array.from(cameraUrlMap, ([id, url]) => ({ id, url }));
  cameraUrls.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));

  if (cameraUrls.length === 0 && typeof env.VITE_RTSP_URL === 'string' && env.VITE_RTSP_URL.trim()) {
    cameraUrls.push({ id: 'CAM_MAIN', url: env.VITE_RTSP_URL.trim() });
  }

  const humanDetectionThreshold = envNumber(env, 'VITE_OPENCV_HUMAN_THRESHOLD', envNumber(env, 'VITE_HUMAN_CONFIDENCE_THRESHOLD', 0.45));
  const embeddingMatchThreshold = envNumber(
    env,
    'VITE_EMBEDDING_MATCH_THRESHOLD',
    envNumber(
      env,
      'VITE_PATTERN_MATCH_THRESHOLD',
      envNumber(env, 'VITE_CONFIDENCE_THRESHOLD', 0.6),
    ),
  );
  const patternReuseThreshold = envNumber(
    env,
    'VITE_PATTERN_REUSE_THRESHOLD',
    embeddingMatchThreshold,
  );

  return {
    confidenceThreshold: embeddingMatchThreshold,
    humanConfidenceThreshold: humanDetectionThreshold,
    patternReuseThreshold,
    sourceMode: env.VITE_SOURCE_MODE === 'webcam' ? 'webcam' : 'rtsp',
    rtspUrl: typeof env.VITE_RTSP_URL === 'string' ? env.VITE_RTSP_URL : '',
    cameras: cameraUrls,
    rawEnv: Object.fromEntries(
      Object.entries(env).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
    ),
  };
}

function envNumber(env: Record<string, unknown>, key: string, fallback: number): number {
  const value = Number(env[key]);
  return Number.isFinite(value) ? value : fallback;
}

function cameraIdFromEnvKey(key: string): string {
  const prefix = CAMERA_ENV_PREFIXES.find((candidate) => key.startsWith(candidate));
  if (prefix) return normalizeCameraId(key.slice(prefix.length));

  const suffixMatch = key.match(/^VITE_(CAM[A-Z0-9_-]+|WEBCAM[A-Z0-9_-]*)_(?:RTSP_)?URL$/i);
  if (suffixMatch) return normalizeCameraId(suffixMatch[1]);
  return '';
}

function normalizeCameraId(rawId: string): string {
  const cleaned = rawId.trim().replace(/[^a-z0-9_-]/gi, '').toUpperCase();
  if (!cleaned) return '';
  if (/^\d+$/.test(cleaned)) return `CAM${cleaned}`;
  if (cleaned.startsWith('CAM') || cleaned.startsWith('WEBCAM')) return cleaned;
  return `CAM${cleaned}`;
}

export default App;
