export const SETTINGS_STORAGE_KEY = 'sentinex-face-settings:v1';

export interface Settings {
  confidenceThreshold: number;
  humanConfidenceThreshold: number;
  patternReuseThreshold: number;
  sourceMode: 'webcam' | 'rtsp';
  rtspUrl: string;
  cameras: { id: string; url: string }[];
  rawEnv?: Record<string, string>;
}
