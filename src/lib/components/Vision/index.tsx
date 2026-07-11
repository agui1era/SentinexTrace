import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  CalendarDays, Camera, Check, Clock, ImageOff, ListChecks, Maximize2, MessageSquare, Send, Shield,
  SlidersHorizontal, Trash2, UserRound, X
} from 'lucide-react';
import type { Settings } from '../../../types';

type CameraConfig = { id: string; url: string };
type Detection = {
  id: string;
  identityId: string;
  name: string;
  tag?: string;
  generic?: boolean;
  cameraId?: string;
  score?: number;
  matchStatus?: string;
  timestamp?: string;
  frame?: string;
  sampleFrames?: string[];
  bbox?: number[];
};
type Identity = {
  id: string;
  name: string;
  tag?: string;
  generic?: boolean;
  cameraId?: string;
  thumbnails?: string[];
};
type PatternSummary = {
  identityId: string;
  name: string;
  tag: string;
  count: number;
  cameras: Set<string>;
  latest?: Detection;
  thumbnailUrls: string[];
  firstSeen?: string;
  lastSeen?: string;
};
type ActivityLog = {
  id: string;
  level: 'info' | 'ok' | 'warn' | 'error';
  message: string;
  time: number;
};
type BackendLog = {
  id?: string;
  level?: ActivityLog['level'];
  message?: string;
  timestamp?: string;
};
type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  detections?: number;
  range?: string;
  time: number;
};

export default function VisionPanel({ settings }: { settings: Settings }) {
  const cameraList: CameraConfig[] = useMemo(
    () => settings.cameras.filter((camera) => camera.url.trim().length > 0),
    [settings.cameras],
  );

  const [view, setView] = useState<'main' | 'history'>('main');
  const [selectedDate, setSelectedDate] = useState('');
  const [backendState, setBackendState] = useState<'idle' | 'ok' | 'error'>('idle');
  const [snapshotToken, setSnapshotToken] = useState(() => Date.now());
  const [snapshotErrors, setSnapshotErrors] = useState<Record<string, boolean>>({});
  const [detections, setDetections] = useState<Detection[]>([]);
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [tagDrafts, setTagDrafts] = useState<Record<string, { name: string; tag: string }>>({});
  const [taggingPatternIds, setTaggingPatternIds] = useState<Record<string, boolean>>({});
  const [deletingPatternIds, setDeletingPatternIds] = useState<Record<string, boolean>>({});
  const [preview, setPreview] = useState<{ src: string; label: string } | null>(null);
  const [chatFrom, setChatFrom] = useState(() => new Date().toISOString().slice(0, 10));
  const [chatTo, setChatTo] = useState(() => new Date().toISOString().slice(0, 10));
  const [chatInput, setChatInput] = useState('');
  const [chatBusy, setChatBusy] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [thresholds, setThresholds] = useState<
    { humanConfidenceThreshold: number; embeddingMatchThreshold: number; reuseThreshold: number } | null
  >(null);
  const [thresholdsBusy, setThresholdsBusy] = useState(false);
  const [activityLog, setActivityLog] = useState<ActivityLog[]>(() => [
    {
      id: 'init',
      level: 'info',
      message: 'UI lista. Esperando datos del backend Vision.',
      time: Date.now(),
    },
  ]);
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const detectionsLoading = useRef(false);
  const logSeq = useRef(0);
  const lastDetectionCount = useRef<number | null>(null);

  const addLog = useCallback((level: ActivityLog['level'], message: string) => {
    setActivityLog((prev) => {
      const now = Date.now();
      const last = prev[0];
      if (last?.message === message && now - last.time < 30000) return prev;
      return [
        {
          id: `${now}-${logSeq.current++}`,
          level,
          message,
          time: now,
        },
        ...prev,
      ].slice(0, 12);
    });
  }, []);

  const fetchBackendLogs = useCallback(async () => {
    try {
      const res = await fetch('/api/vision/logs?limit=12');
      if (!res.ok) return false;
      const logs = ((await res.json()).logs || []) as BackendLog[];
      if (logs.length === 0) return true;
      setActivityLog(logs.map((entry, index) => ({
        id: entry.id || `backend-${index}`,
        level: entry.level || 'info',
        message: entry.message || 'Evento backend',
        time: timestampValue(entry.timestamp),
      })));
      return true;
    } catch (e) {
      return false;
    }
  }, []);

  const fetchDetections = useCallback(async () => {
    if (detectionsLoading.current) return;
    detectionsLoading.current = true;
    try {
      const [detectionsRes, identitiesRes] = await Promise.all([
        fetch('/api/vision/detections?limit=160'),
        fetch('/api/vision/identities'),
      ]);
      if (!detectionsRes.ok) throw new Error(`detections HTTP ${detectionsRes.status}`);
      if (!identitiesRes.ok) throw new Error(`identities HTTP ${identitiesRes.status}`);
      const nextDetections = (await detectionsRes.json()).detections || [];
      const nextIdentities = (await identitiesRes.json()).identities || [];
      setDetections(nextDetections);
      setIdentities(nextIdentities);
      setBackendState('ok');
      const hasBackendLogs = await fetchBackendLogs();
      if (!hasBackendLogs && lastDetectionCount.current !== nextDetections.length) {
        addLog('ok', `Backend OK: ${nextDetections.length} detecciones cargadas.`);
        lastDetectionCount.current = nextDetections.length;
      }
    } catch (e) {
      setBackendState('error');
      addLog('error', 'Backend Vision sin respuesta para detecciones.');
    } finally {
      detectionsLoading.current = false;
    }
  }, [addLog, fetchBackendLogs]);

  useEffect(() => {
    fetchDetections();
    pollTimer.current = setInterval(fetchDetections, 8000);
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, [fetchDetections]);

  useEffect(() => {
    // MJPEG stream now handles live updating
  }, [view]);

  useEffect(() => {
    fetch('/api/vision/thresholds')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setThresholds(data);
      })
      .catch(() => {});
  }, []);

  const humanMode = isHumanMode(settings);
  const visibleDetections = useMemo(
    () => humanMode ? detections.filter(isHumanDetection) : detections,
    [detections, humanMode],
  );
  const untaggedPatterns = useMemo(
    () => summarizeUntaggedPatterns(visibleDetections, identities),
    [visibleDetections, identities],
  );
  const latestDetections = useMemo(() => visibleDetections.slice(0, 10), [visibleDetections]);
  const historyByDate = useMemo(() => groupDetectionsByDate(visibleDetections), [visibleDetections]);
  const historyDates = Object.keys(historyByDate);
  const activeHistoryDate = selectedDate || historyDates[0] || '';
  const activeHistoryDetections = activeHistoryDate ? historyByDate[activeHistoryDate] || [] : [];

  const handleDeleteDetection = async (detectionId: string) => {
    try {
      const res = await fetch(`/api/vision/detections/${detectionId}?limit=160`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      addLog('warn', 'Deteccion borrada.');
      fetchDetections();
    } catch (e) {
      addLog('error', 'No se pudo borrar la deteccion.');
    }
  };

  const handleTagPattern = async (pattern: PatternSummary) => {
    const draft = tagDrafts[pattern.identityId] || { name: pattern.name || '', tag: pattern.tag || '' };
    if (draft.name.trim().length < 2) {
      addLog('warn', 'Falta nombre para taguear el patron.');
      return;
    }

    setTaggingPatternIds((prev) => ({ ...prev, [pattern.identityId]: true }));
    try {
      const res = await fetch(`/api/vision/identities/${pattern.identityId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: draft.name.trim(), tag: draft.tag.trim() }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${res.status}`);
      }
      const payload = await res.json();
      const mergedCount = Array.isArray(payload.identity?.mergedIdentities)
        ? payload.identity.mergedIdentities.length
        : 0;
      const reassignedCount = Number(payload.identity?.reassignedDetections || 0);
      setTagDrafts((prev) => {
        const next = { ...prev };
        delete next[pattern.identityId];
        return next;
      });
      addLog(
        'ok',
        mergedCount > 0
          ? `Patron consolidado como ${draft.name.trim()}: ${mergedCount} duplicados, ${reassignedCount} detecciones.`
          : `Patron tagueado como ${draft.name.trim()}.`,
      );
      fetchDetections();
    } catch (e) {
      const detail = e instanceof Error ? e.message : 'error desconocido';
      addLog('error', `No se pudo nombrar el patron: ${detail}`);
    } finally {
      setTaggingPatternIds((prev) => {
        const next = { ...prev };
        delete next[pattern.identityId];
        return next;
      });
    }
  };

  const openPreview = useCallback((src: string, label: string) => {
    setPreview({ src, label });
  }, []);

  const handleSendChat = async () => {
    if (chatBusy) return;
    const message = chatInput.trim();
    if (!chatFrom || !chatTo) {
      addLog('warn', 'Elegí las fechas para analizar.');
      return;
    }
    if (!message) {
      addLog('warn', 'Escribí una pregunta para el chat.');
      return;
    }

    const userMessage: ChatMessage = {
      id: `${Date.now()}-user`,
      role: 'user',
      content: message,
      range: `${chatFrom} a ${chatTo}`,
      time: Date.now(),
    };
    const nextMessages = [...chatMessages, userMessage];
    setChatMessages(nextMessages);
    setChatInput('');
    setChatBusy(true);

    try {
      const res = await fetch('/api/vision/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fromDate: chatFrom,
          toDate: chatTo,
          message,
          history: chatMessages.slice(-8).map((entry) => ({
            role: entry.role,
            content: entry.content,
          })),
        }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setChatMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-assistant`,
          role: 'assistant',
          content: data.answer || '',
          detections: data.detections || 0,
          range: `${data.fromDate || chatFrom} a ${data.toDate || chatTo}`,
          time: Date.now(),
        },
      ]);
      addLog('ok', `Sentix AI respondio con ${data.detections || 0} detecciones de contexto.`);
    } catch (e) {
      const detail = e instanceof Error ? e.message : 'error desconocido';
      setChatMessages((prev) => prev.filter((entry) => entry.id !== userMessage.id));
      setChatInput(message);
      addLog('error', `No se pudo responder el chat: ${detail}`);
    } finally {
      setChatBusy(false);
    }
  };

  const handleSaveThresholds = async () => {
    if (!thresholds) return;
    setThresholdsBusy(true);
    try {
      const res = await fetch('/api/vision/thresholds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          humanConfidenceThreshold: thresholds.humanConfidenceThreshold,
          embeddingMatchThreshold: thresholds.embeddingMatchThreshold,
        }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setThresholds(data);
      addLog('ok', 'Umbrales actualizados en caliente.');
    } catch (e) {
      const detail = e instanceof Error ? e.message : 'error desconocido';
      addLog('error', `No se pudieron guardar los umbrales: ${detail}`);
    } finally {
      setThresholdsBusy(false);
    }
  };

  const handleDeletePattern = async (pattern: PatternSummary) => {
    const label = pattern.name || 'este patron';
    if (typeof window !== 'undefined'
      && !window.confirm(`Eliminar "${label}" y sus ${pattern.count} detecciones? No se puede deshacer.`)) {
      return;
    }
    setDeletingPatternIds((prev) => ({ ...prev, [pattern.identityId]: true }));
    try {
      const res = await fetch(`/api/vision/identities/${pattern.identityId}`, { method: 'DELETE' });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${res.status}`);
      }
      addLog('warn', `Patron eliminado: ${label}.`);
      fetchDetections();
    } catch (e) {
      const detail = e instanceof Error ? e.message : 'error desconocido';
      addLog('error', `No se pudo eliminar el patron: ${detail}`);
    } finally {
      setDeletingPatternIds((prev) => {
        const next = { ...prev };
        delete next[pattern.identityId];
        return next;
      });
    }
  };

  return (
    <div className="simple-vision-shell">
      <header className="simple-topbar">
        <div className="brand">
          <div className="brand-mark"><Shield size={24} /></div>
          <div>
            <strong>Sentinex</strong>
            <span>Vision Core</span>
          </div>
        </div>
        <div className="simple-tabs" aria-label="Vistas">
          <button className={view === 'main' ? 'active' : ''} onClick={() => setView('main')}>
            <UserRound size={16} /> Principal
          </button>
          <button className={view === 'history' ? 'active' : ''} onClick={() => setView('history')}>
            <CalendarDays size={16} /> Historial
          </button>
        </div>
      </header>

      <section className="scan-summary">
        <div>
          <span>Camaras en .env</span>
          <strong>{cameraList.length}</strong>
        </div>
        <div>
          <span>Backend</span>
          <strong>{backendState === 'ok' ? 'OK' : backendState === 'error' ? 'Error' : '...'}</strong>
        </div>
        <div>
          <span>{humanMode ? 'Humanos sin nombre' : 'Patrones sin tag'}</span>
          <strong>{untaggedPatterns.length}</strong>
        </div>
        <div>
          <span>{humanMode ? 'Detecciones humanas' : 'Detecciones'}</span>
          <strong>{visibleDetections.length}</strong>
        </div>
      </section>

      {view === 'main' ? (
        <main className="simple-main-grid">
          <section className="simple-panel camera-thumbnails-panel">
            <div className="panel-title">
              <div>
                <span>Snapshots</span>
                <strong>Miniaturas de camaras</strong>
              </div>
              <button
                className="ghost-button compact-action"
                onClick={() => {
                  setSnapshotToken(Date.now());
                  setSnapshotErrors({});
                }}
              >
                <Camera size={16} /> Actualizar
              </button>
            </div>
            <div className="camera-thumbnail-grid">
              {cameraList.map((camera) => (
                <article key={camera.id} className="camera-thumbnail-card">
                  <div className="camera-thumbnail-frame">
                    {!snapshotErrors[camera.id] ? (
                      <img
                        src={`/api/vision/${encodeURIComponent(camera.id)}/stream.mjpg`}
                        alt={`Stream ${camera.id}`}
                        onError={() => setSnapshotErrors((prev) => ({ ...prev, [camera.id]: true }))}
                      />
                    ) : (
                      <div>
                        <Camera size={22} />
                        <span>Sin frame</span>
                      </div>
                    )}
                  </div>
                  <div className="camera-thumbnail-meta">
                    <strong>{camera.id}</strong>
                    <span>{snapshotErrors[camera.id] ? 'esperando frame' : 'snapshot'}</span>
                  </div>
                </article>
              ))}
              {cameraList.length === 0 && <div className="empty-list">No hay camaras configuradas</div>}
            </div>
          </section>

          {false && (
            <section className="simple-panel activity-panel">
              <div className="panel-title">
                <div>
                  <span>Actividad</span>
                  <strong>Log de procesamiento</strong>
                </div>
                <ListChecks size={18} color="var(--muted)" />
              </div>
              <ActivityLogList logs={activityLog} />
            </section>
          )}

          <section className="simple-panel">
            <div className="panel-title">
              <div>
                <span>Acumulados</span>
                <strong>{humanMode ? 'Humanos sin nombre' : 'Patrones no tagueados'}</strong>
              </div>
              <span className="count">{untaggedPatterns.length}</span>
            </div>

            <div className="pattern-list">
              {untaggedPatterns.map((pattern) => {
                const draft = tagDrafts[pattern.identityId] || { name: pattern.name || '', tag: pattern.tag || '' };
                const isTagging = Boolean(taggingPatternIds[pattern.identityId]);
                const isDeleting = Boolean(deletingPatternIds[pattern.identityId]);

                return (
                  <article key={pattern.identityId} className="pattern-card">
                    <PatternThumbnail
                      urls={pattern.thumbnailUrls}
                      label={pattern.name}
                      onPreview={openPreview}
                    />
                    <div className="pattern-body">
                      <div className="pattern-head">
                        <div>
                          <strong>{pattern.name}</strong>
                          <span>
                            {pattern.count} detecciones · {Array.from(pattern.cameras).join(', ') || 'CAM'}
                          </span>
                        </div>
                        <small>{formatTime(pattern.lastSeen)}</small>
                      </div>
                      <div className="pattern-form">
                        <input
                          value={draft.name}
                          onChange={(event) => setTagDrafts((prev) => ({
                            ...prev,
                            [pattern.identityId]: { ...draft, name: event.target.value },
                          }))}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') handleTagPattern(pattern);
                          }}
                          placeholder="Nombre"
                        />
                        <input
                          value={draft.tag}
                          onChange={(event) => setTagDrafts((prev) => ({
                            ...prev,
                            [pattern.identityId]: { ...draft, tag: event.target.value },
                          }))}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter') handleTagPattern(pattern);
                          }}
                          placeholder="Tag"
                        />
                        <button
                          className="primary-button pattern-save-button"
                          disabled={isTagging}
                          onClick={() => handleTagPattern(pattern)}
                        >
                          <Check size={18} />
                          <span>{isTagging ? 'Guardando' : 'Nombrar'}</span>
                        </button>
                        <button
                          className="pattern-delete-button"
                          disabled={isDeleting}
                          onClick={() => handleDeletePattern(pattern)}
                          title="Eliminar patron y sus detecciones"
                        >
                          <Trash2 size={18} />
                          <span>{isDeleting ? 'Borrando' : 'Eliminar'}</span>
                        </button>
                      </div>
                    </div>
                  </article>
                );
              })}
              {untaggedPatterns.length === 0 && (
                <div className="empty-list">
                  {humanMode ? 'No hay humanos detectados' : 'No hay patrones pendientes'}
                </div>
              )}
            </div>
          </section>

          <section className="simple-panel">
            <div className="panel-title">
              <div>
                <span>Tiempo real</span>
                <strong>Ultimas detecciones</strong>
              </div>
              <Clock size={18} color="var(--muted)" />
            </div>
            <DetectionList detections={latestDetections} onDelete={handleDeleteDetection} onPreview={openPreview} />
          </section>

          {thresholds && (
            <section className="simple-panel">
              <div className="panel-title">
                <div>
                  <span>Configuración</span>
                  <strong>Umbrales</strong>
                </div>
                <SlidersHorizontal size={18} color="var(--muted)" />
              </div>
              <div className="threshold-list">
                <ThresholdSlider
                  label="Humano (YOLO)"
                  hint="mínimo para detectar persona"
                  value={thresholds.humanConfidenceThreshold}
                  onChange={(v) => setThresholds({ ...thresholds, humanConfidenceThreshold: v })}
                />
                <ThresholdSlider
                  label="Vector identidad"
                  hint="mínimo para reconocer o crear patrón nuevo"
                  value={thresholds.embeddingMatchThreshold}
                  onChange={(v) => setThresholds({ ...thresholds, embeddingMatchThreshold: v, reuseThreshold: v })}
                />
                <button className="primary-button" disabled={thresholdsBusy} onClick={handleSaveThresholds}>
                  <Check size={18} />
                  <span>{thresholdsBusy ? 'Guardando…' : 'Guardar y aplicar'}</span>
                </button>
              </div>
            </section>
          )}

          <section className="simple-panel analysis-chat-panel">
            <div className="panel-title">
              <div>
                <span>Sentix AI</span>
                <strong>Análisis por fecha</strong>
              </div>
              <MessageSquare size={18} color="var(--muted)" />
            </div>
            <div className="analysis-chat-form">
              <div className="analysis-chat-dates">
                <label>
                  <span>Desde</span>
                  <input type="date" value={chatFrom} onChange={(event) => setChatFrom(event.target.value)} />
                </label>
                <label>
                  <span>Hasta</span>
                  <input type="date" value={chatTo} onChange={(event) => setChatTo(event.target.value)} />
                </label>
              </div>
              <div className="analysis-chat-log">
                {chatMessages.map((message) => (
                  <div key={message.id} className={`analysis-chat-message ${message.role}`}>
                    <div className="analysis-chat-meta">
                      <strong>{message.role === 'user' ? 'Tú' : 'Sentix AI'}</strong>
                      <span>
                        {message.range || `${chatFrom} a ${chatTo}`}
                        {typeof message.detections === 'number' ? ` · ${message.detections} det.` : ''}
                      </span>
                    </div>
                    <p>{message.content}</p>
                  </div>
                ))}
                {chatMessages.length === 0 && (
                  <div className="empty-list">Sin preguntas en este chat</div>
                )}
              </div>
              <textarea
                className="analysis-chat-input"
                value={chatInput}
                rows={3}
                placeholder="Pregunta a Sentix AI sobre el rango elegido..."
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    handleSendChat();
                  }
                }}
              />
              <button className="primary-button" disabled={chatBusy} onClick={handleSendChat}>
                <Send size={18} />
                <span>{chatBusy ? 'Analizando…' : 'Enviar'}</span>
              </button>
            </div>
          </section>
        </main>
      ) : (
        <main className="history-view">
          <aside className="date-list">
            <div className="panel-title">
              <div>
                <span>Fechas</span>
                <strong>Historial</strong>
              </div>
            </div>
            {historyDates.map((date) => (
              <button
                key={date}
                className={activeHistoryDate === date ? 'active' : ''}
                onClick={() => setSelectedDate(date)}
              >
                <span>{date}</span>
                <strong>{historyByDate[date].length}</strong>
              </button>
            ))}
            {historyDates.length === 0 && <div className="empty-list">Sin historial</div>}
          </aside>

          <section className="simple-panel">
            <div className="panel-title">
              <div>
                <span>{activeHistoryDate || 'Sin fecha'}</span>
                <strong>Detecciones por fecha</strong>
              </div>
              <span className="count">{activeHistoryDetections.length}</span>
            </div>
            <DetectionList detections={activeHistoryDetections} onDelete={handleDeleteDetection} onPreview={openPreview} />
          </section>
        </main>
      )}
      {preview && (
        <ImagePreview src={preview.src} label={preview.label} onClose={() => setPreview(null)} />
      )}
    </div>
  );
}

function PatternThumbnail({
  urls,
  label,
  onPreview,
}: {
  urls: string[];
  label: string;
  onPreview?: (src: string, label: string) => void;
}) {
  const [failedUrls, setFailedUrls] = useState<Record<string, boolean>>({});
  const availableUrls = urls.filter((url) => url && !failedUrls[url]);
  // 1 imagen por patron: solo la miniatura de enrolamiento (primera del merge).
  const visibleUrls = availableUrls.slice(0, 1);

  if (visibleUrls.length === 0) {
    return (
      <div className="pattern-thumbnail-empty" aria-label={`Sin miniatura para ${label}`}>
        <ImageOff size={22} />
        <span>Sin miniatura</span>
      </div>
    );
  }

  return (
    <a
      href={availableUrls[0]}
      target="_blank"
      rel="noreferrer"
      className={`pattern-thumbnail-stack count-${visibleUrls.length}`}
      aria-label={`Ampliar miniatura de ${label}`}
      onClick={(event) => {
        if (onPreview) {
          event.preventDefault();
          onPreview(availableUrls[0], label);
        }
      }}
    >
      {visibleUrls.map((url) => (
        <img
          key={url}
          src={url}
          alt={label}
          onError={() => setFailedUrls((prev) => ({ ...prev, [url]: true }))}
        />
      ))}
      <span className="pattern-thumbnail-zoom"><Maximize2 size={14} /></span>
    </a>
  );
}

function ActivityLogList({ logs }: { logs: ActivityLog[] }) {
  return (
    <div className="activity-log-list">
      {logs.map((entry) => (
        <div key={entry.id} className={`activity-log-row ${entry.level}`}>
          <span>{formatTimeOnly(entry.time)}</span>
          <strong>{entry.message}</strong>
        </div>
      ))}
    </div>
  );
}

function DetectionList({
  detections,
  onDelete,
  onPreview,
}: {
  detections: Detection[];
  onDelete: (detectionId: string) => void;
  onPreview?: (src: string, label: string) => void;
}) {
  return (
    <div className="simple-detection-list">
      {detections.map((detection) => {
        const thumbUrl = detectionThumbUrl(detection);
        return (
        <article key={detection.id} className="simple-detection-row">
          <DetectionThumb
            src={thumbUrl}
            alt={detection.name}
            onClick={thumbUrl && onPreview ? () => onPreview(thumbUrl, detection.name) : undefined}
          />
          <div className="detection-info">
            <div className="detection-title">
              <strong>{detection.name}</strong>
              {detection.tag ? <span className="tag-chip">{detection.tag}</span> : null}
              <span className={`score-chip ${scoreTier(detection.matchStatus, detection.score)}`} title={detection.matchStatus || ''}>
                {scoreLabel(detection.matchStatus, detection.score)}
              </span>
            </div>
            <span>{detection.cameraId || 'CAM'} · {formatTime(detection.timestamp)}</span>
          </div>
          <button className="ghost-button danger icon-button" onClick={() => onDelete(detection.id)}>
            <Trash2 size={16} />
          </button>
        </article>
        );
      })}
      {detections.length === 0 && <div className="empty-list">Sin detecciones</div>}
    </div>
  );
}

function detectionThumbUrl(detection: Detection): string | undefined {
  const key = detection.sampleFrames?.[0] || detection.frame;
  return key ? `/api/vision/media/${key}` : undefined;
}

function DetectionThumb({ src, alt, onClick }: { src?: string; alt: string; onClick?: () => void }) {
  const [failed, setFailed] = useState(false);
  if (!src || failed) {
    return (
      <div className="detection-thumb detection-thumb-empty">
        <ImageOff size={16} />
      </div>
    );
  }
  return (
    <img
      className={`detection-thumb${onClick ? ' is-clickable' : ''}`}
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
      onClick={onClick}
    />
  );
}

function ThresholdSlider({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="threshold-row">
      <div className="threshold-head">
        <div>
          <strong>{label}</strong>
          <span>{hint}</span>
        </div>
        <span className="threshold-value">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
      />
    </div>
  );
}

function ImagePreview({ src, label, onClose }: { src: string; label: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="pattern-preview-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="pattern-preview-modal" onClick={(event) => event.stopPropagation()}>
        <div className="pattern-preview-topbar">
          <div>
            <span>Vista previa</span>
            <strong>{label}</strong>
          </div>
          <button className="ghost-button icon-button" onClick={onClose} aria-label="Cerrar">
            <X size={18} />
          </button>
        </div>
        <div className="pattern-preview-stage">
          <img src={src} alt={label} />
        </div>
      </div>
    </div>
  );
}

function scoreTier(matchStatus?: string, score?: number): string {
  if (matchStatus === 'known') return 'is-known';
  if (matchStatus === 'known_low_confidence') return 'is-low';
  if (matchStatus === 'unknown_auto_enrolled') return 'is-new';
  const value = score || 0;
  if (value >= 0.9) return 'is-known';
  if (value >= 0.88) return 'is-low';
  return 'is-new';
}

function scoreLabel(matchStatus?: string, score?: number): string {
  if (matchStatus === 'unknown_auto_enrolled') return 'NUEVO';
  return `${Math.round((score || 0) * 100)}%`;
}

function summarizeUntaggedPatterns(detections: Detection[], identities: Identity[]): PatternSummary[] {
  const byIdentity = new Map<string, PatternSummary>();
  const identitiesById = new Map(identities.map((identity) => [identity.id, identity]));

  // 1. Populate map with all generic/untagged identities from the database
  for (const identity of identities) {
    const isGeneric = Boolean(identity.generic);
    const tag = identity.tag || '';
    if (!isGeneric && tag) continue;

    byIdentity.set(identity.id, {
      identityId: identity.id,
      name: identity.name || 'Patrón',
      tag,
      count: 0,
      cameras: new Set(identity.cameraId ? [identity.cameraId] : []),
      thumbnailUrls: patternThumbnailUrls(identity, undefined),
      firstSeen: undefined,
      lastSeen: undefined,
    });
  }

  // 2. Loop through detections to update counts, cameras, and timestamps
  for (const detection of detections) {
    if (!detection.identityId) continue;

    const identity = identitiesById.get(detection.identityId);
    let existing = byIdentity.get(detection.identityId);

    if (!existing) {
      const isGeneric = Boolean(detection.generic || identity?.generic);
      const tag = detection.tag || identity?.tag || '';
      if (!isGeneric && tag) continue;

      existing = {
        identityId: detection.identityId,
        name: identity?.name || detection.name || 'Patrón',
        tag,
        count: 0,
        cameras: new Set(),
        thumbnailUrls: patternThumbnailUrls(identity, detection),
        firstSeen: undefined,
        lastSeen: undefined,
      };
      byIdentity.set(detection.identityId, existing);
    }

    existing.count += 1;
    if (detection.cameraId) existing.cameras.add(detection.cameraId);
    existing.thumbnailUrls = mergeUnique(existing.thumbnailUrls, patternThumbnailUrls(identity, detection));
    if (isAfter(detection.timestamp, existing.lastSeen)) {
      existing.latest = detection;
      existing.lastSeen = detection.timestamp;
    }
    if (isBefore(detection.timestamp, existing.firstSeen)) {
      existing.firstSeen = detection.timestamp;
    }
  }

  return Array.from(byIdentity.values()).sort((a, b) => {
    const timeA = a.lastSeen ? timestampValue(a.lastSeen) : (a.firstSeen ? timestampValue(a.firstSeen) : 0);
    const timeB = b.lastSeen ? timestampValue(b.lastSeen) : (b.firstSeen ? timestampValue(b.firstSeen) : 0);
    if (timeA === 0 && timeB === 0) {
      return b.identityId.localeCompare(a.identityId);
    }
    return timeB - timeA;
  });
}

function isHumanMode(settings: Settings) {
  const rawValue = settings.rawEnv?.VITE_REQUIRE_HUMAN || settings.rawEnv?.VITE_SCAN_REQUIRE_HUMAN || '';
  return ['1', 'true', 'yes', 'y', 'on', 'si', 'sí'].includes(rawValue.trim().toLowerCase());
}

function isHumanDetection(detection: Detection) {
  return Array.isArray(detection.bbox) && detection.bbox.length >= 4;
}

function patternThumbnailUrls(identity?: Identity, detection?: Detection) {
  return mergeUnique([
    ...(identity?.thumbnails || []),
    ...(detection?.sampleFrames || []).map(mediaUrl),
    mediaUrl(detection?.frame),
  ]);
}

function mergeUnique(...groups: string[][]) {
  const urls: string[] = [];
  for (const url of groups.flat()) {
    if (url && !urls.includes(url)) urls.push(url);
    if (urls.length >= 3) break;
  }
  return urls;
}

function groupDetectionsByDate(detections: Detection[]) {
  const grouped: Record<string, Detection[]> = {};
  for (const detection of detections) {
    const key = dateKey(detection.timestamp);
    grouped[key] = grouped[key] || [];
    grouped[key].push(detection);
  }
  return Object.fromEntries(
    Object.entries(grouped).sort(([left], [right]) => right.localeCompare(left)),
  );
}

function dateKey(value?: string) {
  if (!value) return 'Sin fecha';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Sin fecha';
  return date.toISOString().slice(0, 10);
}

function formatTime(value?: string) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleString();
}

function formatTimeOnly(value: number) {
  return new Date(value).toLocaleTimeString();
}

function mediaUrl(frame?: string) {
  if (frame?.startsWith('/api/vision/media/')) return frame;
  return frame ? `/api/vision/media/${frame}` : '';
}

function timestampValue(value?: string) {
  if (!value) return 0;
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

function isAfter(left?: string, right?: string) {
  return timestampValue(left) > timestampValue(right);
}

function isBefore(left?: string, right?: string) {
  const leftValue = timestampValue(left);
  const rightValue = timestampValue(right);
  return leftValue > 0 && (rightValue === 0 || leftValue < rightValue);
}
