# Deploy en Render (Graceland Soccer Web)

## Ver el código en GitHub

1. Abre **https://github.com/Alvaromp3/ml-models**
2. Navega a la carpeta: **`graceland_soccer_model/graceland_soccer_web/`**
3. Rama: **`main`** (o la rama que hayas publicado con *Publish Branch* en Cursor).

Si acabas de usar **Publish Branch** en Cursor, en GitHub ve a **Code → branches** y elige tu rama; luego **Compare & pull request** si quieres fusionar a `main`.

## Blueprint (un solo archivo)

En la raíz del repo hay **`render.yaml`**. En Render:

1. **New +** → **Blueprint**
2. Conecta el repo **Alvaromp3/ml-models**
3. Render detecta `render.yaml` y crea **dos servicios**: API estática (frontend) + Python (backend).

### Variables importantes (Dashboard → cada servicio → Environment)

| Servicio | Variable | Descripción |
|----------|-----------|-------------|
| Backend | `ALLOWED_ORIGINS` | Orígenes del SPA, separados por coma. Ej: `https://graceland-frontend.onrender.com` |
| Backend | `DISABLE_MODEL_TRAINING` | `1` en producción (ya va en el blueprint). |
| Backend | `ENVIRONMENT` | `production` oculta `/docs` y OpenAPI. |
| Backend | `API_KEY` | Opcional. Si existe, todas las rutas (excepto `/`, `/health`) requieren cabecera `X-API-Key` o `Authorization: Bearer …`. |
| Backend | `OPEN_ROUTER_API_KEY` | Para informes con IA (opcional). |
| Frontend | `VITE_API_BASE_URL` | URL **solo del host** del backend, sin `/api`. Ej: `https://graceland-backend.onrender.com` |
| Frontend | `VITE_API_KEY` | Solo si activaste `API_KEY` en el backend **y** aceptas que la clave viaja en el bundle JS (no es segura frente a usuarios). |

Tras cambiar variables del **frontend**, hay que **volver a desplegar** el sitio estático (nuevo build).

### CORS

El backend lee `ALLOWED_ORIGINS`. Debe incluir la URL exacta del frontend (esquema + host, sin barra final), por ejemplo `https://graceland-frontend.onrender.com`.

### Autenticación

`API_KEY` protege el API frente a accesos casuales **si la clave no está en el frontend**. Si la pones en `VITE_API_KEY`, cualquiera puede verla en el navegador. Para un despliegue serio, lo habitual es **mismo origen** (API + estático detrás de un proxy) o **login real (OAuth/JWT)**.

### Datos en disco (CSV, modelos)

Los planes gratuitos de Render suelen tener **filesystem efímero**: al reiniciar el servicio se pierden datos subidos salvo que:

- Añadas un **Disk** al servicio backend y montes una ruta persistente, y
- Configures `DATA_STORE_DIR` (u otra variable que use tu `data_service`) apuntando a ese directorio.

Revisa en código dónde se guardan los CSV y alinea esa ruta con el volumen.

### HTTPS y secretos

- Render ya sirve **HTTPS** en `*.onrender.com`.
- No subas `.env` al repositorio; usa solo **Environment** en Render.

### Publicar rama desde Cursor

En Cursor, **Publish Branch** sube la rama actual a `origin`. En GitHub: **repositorio → rama (dropdown)** → selecciona la rama → **Pull requests** si quieres fusionar a `main`.

También puedes hacer desde terminal en la raíz del repo `ml-models`:

```bash
git push -u origin nombre-de-tu-rama
```
