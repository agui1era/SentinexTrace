import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Shield, Play, Square, Camera, UserPlus, Trash2, 
  Settings2, Activity, Wifi, Tag
} from 'lucide-react';
import type { Settings } from '../../../types';

export default function VisionPanel({ settings }: { settings: Settings }) {
  const [activeCamera, setActiveCamera] = useState<string>(settings.cameras[0]?.id || 'CAM1');
  const [streamUrl, setStreamUrl] = useState<string>(settings.cameras[0]?.url || '');
  const [globalStatus, setGlobalStatus] = useState<any>(null);
  const [camStatus, setCamStatus] = useState<any>(null);
  
  const [identities, setIdentities] = useState<any[]>([]);
  const [captures, setCaptures] = useState<any[]>([]);
  const [detections, setDetections] = useState<any[]>([]);
  
  const [selectedIdentity, setSelectedIdentity] = useState<any>(null);
  const [enrollName, setEnrollName] = useState('');
  
  const [scanThreshold, setScanThreshold] = useState(settings.confidenceThreshold);
  const [humanThreshold, setHumanThreshold] = useState(settings.humanConfidenceThreshold);
  
  const pollTimer = useRef<NodeJS.Timeout>();

  const fetchGlobalStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/vision/status');
      if (res.ok) setGlobalStatus(await res.json());
    } catch (e) {}
  }, []);

  const fetchCamStatus = useCallback(async () => {
    if (!activeCamera) return;
    try {
      const res = await fetch(`/api/vision/${activeCamera}/status`);
      if (res.ok) setCamStatus(await res.json());
    } catch (e) {}
  }, [activeCamera]);

  const fetchData = useCallback(async () => {
    try {
      const [idRes, capRes, detRes] = await Promise.all([
        fetch('/api/vision/identities'),
        fetch('/api/vision/enroll/captures'),
        fetch('/api/vision/detections?limit=10')
      ]);
      if (idRes.ok) setIdentities((await idRes.json()).identities || []);
      if (capRes.ok) setCaptures((await capRes.json()).captures || []);
      if (detRes.ok) setDetections((await detRes.json()).detections || []);
    } catch (e) {}
  }, []);

  useEffect(() => {
    fetchGlobalStatus();
    fetchData();
    pollTimer.current = setInterval(() => {
      fetchGlobalStatus();
      fetchCamStatus();
      fetchData();
    }, 2000);
    return () => clearInterval(pollTimer.current);
  }, [fetchGlobalStatus, fetchCamStatus, fetchData]);

  const handleStartStream = async () => {
    await fetch(`/api/vision/${activeCamera}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: streamUrl })
    });
    fetchCamStatus();
  };

  const handleStopStream = async () => {
    await fetch(`/api/vision/${activeCamera}/stream/stop`, { method: 'POST' });
    fetchCamStatus();
  };

  const handleStartScan = async () => {
    await fetch(`/api/vision/${activeCamera}/scan/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embeddingMatchThreshold: scanThreshold,
        humanConfidenceThreshold: humanThreshold,
        intervalSeconds: 1.0,
        requireHuman: true
      })
    });
    fetchCamStatus();
  };

  const handleStopScan = async () => {
    await fetch(`/api/vision/${activeCamera}/scan/stop`, { method: 'POST' });
    fetchCamStatus();
  };

  const handleCapture = async () => {
    await fetch(`/api/vision/${activeCamera}/enroll/capture`, { method: 'POST' });
    fetchData();
  };

  const handleEnroll = async () => {
    if (enrollName.length < 2 || captures.length < 3) return;
    await fetch('/api/vision/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: enrollName,
        tag: 'User',
        sampleIds: captures.map(c => c.id)
      })
    });
    setEnrollName('');
    fetchData();
  };

  const handleDeleteIdentity = async (id: string) => {
    await fetch(`/api/vision/identities/${id}`, { method: 'DELETE' });
    if (selectedIdentity?.id === id) setSelectedIdentity(null);
    fetchData();
  };

  const handleClearCaptures = async () => {
    await fetch('/api/vision/enroll/captures', { method: 'DELETE' });
    fetchData();
  };

  const streamRunning = camStatus?.stream?.running;
  const scanRunning = camStatus?.scanner?.running;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Shield size={24} /></div>
          <div>
            <strong>Sentinex</strong>
            <span>Vision Core</span>
          </div>
        </div>

        {globalStatus && (
          <div className={`system-pill ${globalStatus.clip?.loaded ? 'success' : 'danger'}`}>
            <Activity size={16} /> 
            {globalStatus.clip?.loaded ? 'Sistema en Linea' : 'Sistema Offline'}
          </div>
        )}

        <div className="identity-section">
          <div className="section-heading">
            <span>Identidades</span>
            <span className="count">{identities.length}</span>
          </div>
          
          <div className="identity-list">
            {identities.map(id => (
              <button 
                key={id.id} 
                className={`identity-row ${selectedIdentity?.id === id.id ? 'selected' : ''}`}
                onClick={() => setSelectedIdentity(id)}
              >
                <img src={`/api/vision/media/${id.sample_keys?.[0] || ''}`} alt={id.name} />
                <div>
                  <strong>{id.name}</strong>
                  <small>{id.tag || 'Sin Tag'}</small>
                </div>
              </button>
            ))}
            {identities.length === 0 && <div className="empty-list">No hay identidades</div>}
          </div>
        </div>

        {selectedIdentity && (
          <div className="selected-card">
            <div className="panel-title">
              <div>
                <span>Seleccionado</span>
                <strong>{selectedIdentity.name}</strong>
              </div>
              <button className="ghost-button danger" onClick={() => handleDeleteIdentity(selectedIdentity.id)}>
                <Trash2 size={16} />
              </button>
            </div>
            <div className="selected-preview">
              {selectedIdentity.sample_keys?.slice(0, 3).map((key: string, idx: number) => (
                <img key={idx} src={`/api/vision/media/${key}`} alt="sample" />
              ))}
            </div>
            <div className="meta-grid">
              <span>Tag:</span><strong>{selectedIdentity.tag || '-'}</strong>
              <span>Creado:</span><strong>{new Date(selectedIdentity.created_at * 1000).toLocaleDateString()}</strong>
            </div>
          </div>
        )}
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <p>Monitoreo en Tiempo Real</p>
            <h1>Centro de Vision</h1>
          </div>
          <div className="mode-switch">
            {settings.cameras.map(cam => (
              <button 
                key={cam.id} 
                className={activeCamera === cam.id ? 'active' : ''}
                onClick={() => {
                  setActiveCamera(cam.id);
                  setStreamUrl(cam.url);
                }}
              >
                <Wifi size={16} /> {cam.id}
              </button>
            ))}
            {settings.cameras.length === 0 && (
              <button className="active">
                <Wifi size={16} /> CAM1
              </button>
            )}
          </div>
        </header>

        <div className="main-grid">
          <div className="camera-panel">
            <div className="panel-title">
              <div>
                <span>Feed de Camara</span>
                <strong>{activeCamera}</strong>
              </div>
              {streamRunning && (
                <div className="live-dot">
                  <span /> En Vivo
                </div>
              )}
            </div>

            <div className="video-frame">
              {streamRunning ? (
                <img 
                  className="rtsp-video" 
                  src={`/api/vision/${activeCamera}/stream.mjpg?t=${Date.now()}`} 
                  alt="Live feed" 
                />
              ) : (
                <div className="empty-list" style={{border: 'none', background: 'transparent'}}>
                  <Wifi size={32} opacity={0.3} />
                  <span>Stream Apagado</span>
                </div>
              )}
              <div className="face-guide"><span /></div>
            </div>

            <div className="camera-actions">
              <div className="field" style={{flex: 1}}>
                <input 
                  type="text" 
                  value={streamUrl} 
                  onChange={e => setStreamUrl(e.target.value)} 
                  placeholder="URL RTSP o 0 para Webcam" 
                />
              </div>
              {!streamRunning ? (
                <button className="primary-button" onClick={handleStartStream}>
                  <Play size={18} /> Iniciar
                </button>
              ) : (
                <button className="primary-button stop-live" onClick={handleStopStream}>
                  <Square size={18} /> Detener
                </button>
              )}
            </div>
          </div>

          <div className="operator-panel">
            <div className="panel-title">
              <div>
                <span>Operacion</span>
                <strong>Control del Scanner</strong>
              </div>
              {scanRunning && <Activity size={18} color="var(--success)" />}
            </div>

            <div className="model-card">
              <div className="confidence-control">
                <span className="field-label">Confianza Match</span>
                <input 
                  type="number" 
                  step="0.05" 
                  min="0.1" 
                  max="0.99" 
                  value={scanThreshold} 
                  onChange={e => setScanThreshold(parseFloat(e.target.value))} 
                />
              </div>
              <div className="confidence-control">
                <span className="field-label">Confianza Humano</span>
                <input 
                  type="number" 
                  step="0.05" 
                  min="0.1" 
                  max="0.99" 
                  value={humanThreshold} 
                  onChange={e => setHumanThreshold(parseFloat(e.target.value))} 
                />
              </div>

              {!scanRunning ? (
                <button className="primary-button" onClick={handleStartScan}>
                  <Activity size={18} /> Iniciar Escaneo
                </button>
              ) : (
                <button className="primary-button stop-live" onClick={handleStopScan}>
                  <Square size={18} /> Detener Escaneo
                </button>
              )}
            </div>

            <div className="panel-title" style={{marginTop: '10px'}}>
              <div>
                <span>Captura</span>
                <strong>Enrolar Nuevo</strong>
              </div>
              <button className="ghost-button" onClick={handleClearCaptures}>
                <Trash2 size={16} /> Limpiar
              </button>
            </div>

            <button className="ghost-button" onClick={handleCapture} disabled={!streamRunning}>
              <Camera size={18} /> Capturar Frame
            </button>

            <div className="capture-strip">
              {[0, 1, 2].map(i => {
                const cap = captures[i];
                return (
                  <div key={i} className={`capture-slot ${cap ? 'filled' : ''}`}>
                    {cap ? (
                      <img src={cap.url} alt="capture" />
                    ) : (
                      i + 1
                    )}
                  </div>
                );
              })}
            </div>

            <div className="field">
              <input 
                type="text" 
                placeholder="Nombre de la persona" 
                value={enrollName} 
                onChange={e => setEnrollName(e.target.value)} 
              />
            </div>
            
            <button 
              className="enroll-button" 
              onClick={handleEnroll}
              disabled={enrollName.length < 2 || captures.length < 3}
            >
              <UserPlus size={18} /> Guardar Identidad
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
