import http from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const PORT = Number(process.env.RTSP_SERVER_PORT || 8787);
const FFMPEG_PATH = process.env.FFMPEG_PATH || 'ffmpeg';
const DEFAULT_RTSP_URL = process.env.RTSP_URL || '';
const FPS = Number(process.env.RTSP_FPS || 2);
const WIDTH = Number(process.env.RTSP_WIDTH || 960);

let rtspUrl = DEFAULT_RTSP_URL;
let ffmpegProcess = null;
let latestFrame = null;
let lastFrameAt = null;
let lastError = '';
const mjpegClients = new Set();

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`);

    if (request.method === 'OPTIONS') {
      sendCors(response);
      response.writeHead(204).end();
      return;
    }

    if (request.method === 'GET' && url.pathname === '/api/rtsp/status') {
      sendJson(response, statusPayload());
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/rtsp/config') {
      const body = await readJson(request);
      const nextUrl = typeof body.url === 'string' ? body.url.trim() : '';

      if (!nextUrl) {
        stopFfmpeg();
        rtspUrl = '';
        latestFrame = null;
        lastFrameAt = null;
        lastError = 'URL RTSP vacia';
        sendJson(response, statusPayload(), 400);
        return;
      }

      rtspUrl = nextUrl;
      startFfmpeg();
      sendJson(response, statusPayload());
      return;
    }

    if (request.method === 'POST' && url.pathname === '/api/rtsp/stop') {
      stopFfmpeg();
      sendJson(response, statusPayload());
      return;
    }

    if (request.method === 'GET' && url.pathname === '/api/rtsp/snapshot') {
      ensureStarted();

      if (!latestFrame) {
        sendJson(response, { error: lastError || 'Aun no hay frame RTSP disponible' }, 503);
        return;
      }

      sendCors(response);
      response.writeHead(200, {
        'cache-control': 'no-store',
        'content-length': latestFrame.length,
        'content-type': 'image/jpeg',
      });
      response.end(latestFrame);
      return;
    }

    if (request.method === 'GET' && url.pathname === '/api/rtsp/mjpeg') {
      ensureStarted();
      sendCors(response);
      response.writeHead(200, {
        'cache-control': 'no-store, no-cache, must-revalidate',
        connection: 'close',
        'content-type': 'multipart/x-mixed-replace; boundary=sentinexframe',
        pragma: 'no-cache',
      });

      const client = response;
      mjpegClients.add(client);
      request.on('close', () => {
        mjpegClients.delete(client);
      });

      if (latestFrame) {
        writeMjpegFrame(client, latestFrame);
      }
      return;
    }

    sendJson(response, { error: 'Not found' }, 404);
  } catch (error) {
    sendJson(response, { error: error instanceof Error ? error.message : 'RTSP server error' }, 500);
  }
});

server.listen(PORT, () => {
  console.log(`RTSP bridge listening on http://localhost:${PORT}`);
  if (rtspUrl) {
    startFfmpeg();
  }
});

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

function ensureStarted() {
  if (rtspUrl && !ffmpegProcess) {
    startFfmpeg();
  }
}

function startFfmpeg() {
  stopFfmpeg();
  latestFrame = null;
  lastFrameAt = null;
  lastError = '';

  if (!rtspUrl) {
    lastError = 'URL RTSP no configurada';
    return;
  }

  const args = [
    '-hide_banner',
    '-loglevel',
    'warning',
    '-rtsp_transport',
    'tcp',
    '-i',
    rtspUrl,
    '-an',
    '-vf',
    `fps=${FPS},scale=${WIDTH}:-1`,
    '-q:v',
    '5',
    '-f',
    'image2pipe',
    '-vcodec',
    'mjpeg',
    'pipe:1',
  ];

  ffmpegProcess = spawn(FFMPEG_PATH, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let buffer = Buffer.alloc(0);

  ffmpegProcess.stdout.on('data', (chunk) => {
    buffer = Buffer.concat([buffer, chunk]);
    const result = extractJpegFrames(buffer);
    buffer = result.rest;

    for (const frame of result.frames) {
      latestFrame = frame;
      lastFrameAt = Date.now();
      lastError = '';
      broadcastFrame(frame);
    }
  });

  ffmpegProcess.stderr.on('data', (chunk) => {
    const text = String(chunk).trim();
    if (text) {
      lastError = text.slice(-500);
    }
  });

  ffmpegProcess.on('exit', (code, signal) => {
    ffmpegProcess = null;
    if (code !== 0 && signal !== 'SIGTERM') {
      lastError = lastError || `ffmpeg salio con codigo ${code ?? signal}`;
    }
  });
}

function stopFfmpeg() {
  if (!ffmpegProcess) {
    return;
  }

  const current = ffmpegProcess;
  ffmpegProcess = null;
  current.kill('SIGTERM');
}

function extractJpegFrames(buffer) {
  const frames = [];
  let cursor = 0;

  while (cursor < buffer.length) {
    const start = buffer.indexOf(Buffer.from([0xff, 0xd8]), cursor);
    if (start === -1) {
      return { frames, rest: Buffer.alloc(0) };
    }

    const end = buffer.indexOf(Buffer.from([0xff, 0xd9]), start + 2);
    if (end === -1) {
      return { frames, rest: buffer.subarray(start) };
    }

    frames.push(buffer.subarray(start, end + 2));
    cursor = end + 2;
  }

  return { frames, rest: Buffer.alloc(0) };
}

function broadcastFrame(frame) {
  for (const client of mjpegClients) {
    writeMjpegFrame(client, frame);
  }
}

function writeMjpegFrame(response, frame) {
  response.write(`--sentinexframe\r\ncontent-type: image/jpeg\r\ncontent-length: ${frame.length}\r\n\r\n`);
  response.write(frame);
  response.write('\r\n');
}

function statusPayload() {
  return {
    clients: mjpegClients.size,
    configured: Boolean(rtspUrl),
    error: lastError,
    lastFrameAt,
    running: Boolean(ffmpegProcess),
    url: maskRtspUrl(rtspUrl),
  };
}

function maskRtspUrl(url) {
  if (!url) {
    return '';
  }

  try {
    const parsed = new URL(url);
    if (parsed.password) parsed.password = '***';
    if (parsed.username) parsed.username = '***';
    return parsed.toString();
  } catch {
    return url.replace(/\/\/([^:@/]+):([^@/]+)@/, '//***:***@');
  }
}

async function readJson(request) {
  const chunks = [];
  request.on('data', (chunk) => chunks.push(chunk));
  await once(request, 'end');
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) : {};
}

function sendJson(response, payload, statusCode = 200) {
  const body = JSON.stringify(payload);
  sendCors(response);
  response.writeHead(statusCode, {
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(body),
    'content-type': 'application/json',
  });
  response.end(body);
}

function sendCors(response) {
  response.setHeader('access-control-allow-headers', 'content-type');
  response.setHeader('access-control-allow-methods', 'GET,POST,OPTIONS');
  response.setHeader('access-control-allow-origin', '*');
}

function shutdown() {
  stopFfmpeg();
  server.close(() => process.exit(0));
}
