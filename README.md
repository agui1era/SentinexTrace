# Sentinex Face

Prototipo local de enrolamiento biometrico usando embeddings CLIP en el navegador.

## Correr

```bash
npm install
npm run dev:vision
npm run dev -- --port 5173
```

Luego abre `http://localhost:5173/` y permite el acceso a la camara.

Para el motor Python de vision primero crea el entorno local e instala la base:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-vision.txt
```

Para generar embeddings CLIP reales en Python instala el extra pesado:

```bash
.venv/bin/pip install -r requirements-clip.txt
```

## Flujo

1. En `Enrolar`, escribe el nombre de la persona.
2. Captura 3 fotos frontales con buena luz.
3. Presiona `Generar vector` para guardar el promedio normalizado del embedding.
4. En `Buscar`, toma una lectura y revisa la mejor coincidencia con ranking.
5. En `Usuario`, usa la misma busqueda pero con una pantalla simple que saluda a la persona reconocida. Activa `Modo continuo` en esa ventana para que mire, reconozca y hable.
6. En `Continuo`, activa el reconocimiento automatico y configura el saludo TTS y la confidence minima.
7. En `Ajustes`, cambia la fuente entre `Webcam` y `RTSP`.
8. En `Vision RTSP`, usa el backend Python para ver RTSP en vivo, sacar 3 o 5 fotos del stream, generar embedding CLIP con nombre/tag y guardar detecciones con timestamp. El scanner continuo detecta humanos, toma varias muestras, busca contra Chroma/JSON y crea una identidad generica si no supera el umbral.

La segunda ventana para usuario/kiosko abre directo con `http://localhost:5173/?mode=usuario` o `http://localhost:5173/?mode=continuo`. Por restricciones del navegador, presiona `Modo continuo` o `Iniciar modo continuo` en esa ventana para habilitar camara continua y TTS.

## TTS

La frase por defecto es `Hola {nombre}`. Puedes cambiarla desde `Continuo`; si la frase no incluye `{nombre}`, la app agrega el nombre reconocido al final. Tambien puedes configurar el confidence minimo para que el saludo solo se dispare con una coincidencia suficiente. La configuracion queda guardada en `localStorage`.

## RTSP

El navegador no abre `rtsp://` directo. El modo recomendado es `Vision RTSP` con Python. Tambien queda disponible el bridge Node legacy con `ffmpeg`:

```bash
npm run dev:rtsp
```

Luego en `Ajustes` selecciona `RTSP`, pega la URL `rtsp://...` y presiona `Conectar RTSP`.
El frontend muestra el stream via `/api/rtsp/mjpeg` y toma snapshots via `/api/rtsp/snapshot` para pasarlos por CLIP solo contra identidades registradas.

Tambien puedes iniciar el bridge con una URL por entorno:

```bash
RTSP_URL='rtsp://usuario:password@ip:554/stream' npm run dev:rtsp
```

Los datos quedan en `localStorage` del navegador. No hay servidor ni subida de fotos.

## Vision Python

El flujo recomendado para camaras RTSP reales es el modo `Vision RTSP`:

```bash
npm run dev:vision
npm run dev -- --port 5173
```

Abre `http://localhost:5173/?mode=vision`, pega la URL RTSP y presiona `Conectar`.
El backend Python expone:

- `/api/vision/stream.mjpg`: visor MJPEG para el navegador.
- `/api/vision/snapshot`: ultimo frame JPEG.
- `/api/vision/enroll/capture`: guarda una foto del stream.
- `/api/vision/enroll`: promedia 3 a 5 embeddings CLIP y registra nombre/tag.
- `/api/vision/scan/start`: revisa continuamente el stream, detecta humanos con OpenCV HOG, compara contra la base vectorial y guarda detecciones.
- `/api/vision/detections`: lista detecciones con timestamp, identidad y score.
- `/api/vision/identities/{id}` `PATCH`: renombra una identidad generica creada por el scanner.

El scanner usa por defecto `threshold=0.9`, `sampleCount=3` y `autoEnrollUnknown=true`: acepta como conocido solo el match sobre el umbral; si detecta humano pero no lo reconoce, guarda muestras, embedding e identidad generica `person-0001`, `person-0002`, etc.

La base vectorial local queda en Chroma si esta instalado/configurado y tambien en `data/vision/vector_db.json` como fallback. Las fotos quedan en `data/vision/`. Esa carpeta esta ignorada por git.

Variables utiles:

```bash
SENTINEX_DATA_DIR=data/vision
SENTINEX_CHROMA_DIR=data/vision/chroma_db
SENTINEX_CHROMA_COLLECTION=sentinex_face_identities
SENTINEX_MONGO_URI=mongodb://localhost:27017
SENTINEX_MONGO_DB=sentinex_face
SENTINEX_MONGO_DETECTION_COLLECTION=sentinex_face_rtsp_detections
```

Mongo registra las detecciones en `SENTINEX_MONGO_DETECTION_COLLECTION` para no mezclar este sistema con otras colecciones.

## Nota tecnica

CLIP sirve para prototipar similitud visual, pero no es el modelo ideal para biometria facial productiva. Para una version mas robusta conviene cambiar el extractor por ArcFace, FaceNet u otro modelo especializado y calibrar umbrales con datos reales.
