# Alura Agente — Santos Pegasus Soluciones

Agente de IA que responde preguntas sobre la documentación interna de Santos
Pegasus Soluciones (onboarding, ingeniería back-end/front-end, protocolo de
incidentes, arquitectura de microservicios) usando RAG (Retrieval-Augmented
Generation).

**Demo en vivo:** https://alura-agente-santos-pegasus.onrender.com

## Descripción del proyecto

Santos Pegasus Soluciones es una empresa de software con varios manuales
técnicos internos (onboarding, guías de ingeniería, protocolos de
incidentes, arquitectura). En vez de que cada desarrollador tenga que abrir
y buscar manualmente entre varios PDFs, este agente permite hacer preguntas
en lenguaje natural y obtener respuestas directas, basadas únicamente en el
contenido real de esos documentos — citando de qué archivo salió cada
respuesta.

## Arquitectura

```
PDF (data/) --> ingest.py (carga + chunking)
             --> vectorstore.py (Pinecone: embeddings integrados + índice)
             --> agent.py (retrieval en Pinecone + generación con Groq) --> respuesta
             --> main.py (FastAPI: interfaz web + endpoint /ask)
             --> desplegado en Render (Web Service)
```

- **Ingesta**: `PyPDFLoader` (LangChain) + `RecursiveCharacterTextSplitter`
- **Vector store**: Pinecone (serverless, embeddings integrados con el modelo `multilingual-e5-large` — no se carga ningún modelo localmente)
- **LLM**: Groq (`llama-3.3-70b-versatile`), vía SDK de OpenAI apuntando al endpoint compatible de Groq
- **API**: FastAPI — interfaz web simple en `/` y endpoint `POST /ask`
- **Deploy**: Render (Web Service, plan free), build automático desde GitHub

### Por qué esta arquitectura (decisiones y aprendizajes)

La sugerencia inicial del challenge era usar OCI (Oracle Cloud) tanto para
el LLM como para el deploy. Por motivos personales (el parto prematuro de
mi esposa durante el desarrollo de este proyecto), no tuve el tiempo
necesario para revisar y validar correctamente el funcionamiento de OCI, así
que se optó por herramientas con las que ya tenía experiencia previa (Groq,
que ya uso en otros proyectos como Vecy y FinAI) para poder avanzar con
seguridad bajo esa limitación de tiempo.

El diseño original para el retrieval usaba embeddings locales
(`sentence-transformers` + FAISS). Al desplegar en el free tier de Render
(límite de 512MB de RAM), el proceso se quedaba sin memoria solo al cargar
el modelo de embeddings (confirmado con medición real: ~585MB). Se evaluaron
alternativas (API de inferencia de HuggingFace, `fastembed`/ONNX) y todas
tenían el mismo problema de memoria o limitaciones de cuota gratuita poco
fiables. La solución final usa Pinecone con embeddings integrados: el vector
se genera en los servidores de Pinecone, así que el servidor de Render nunca
carga un modelo pesado — solo hace llamadas HTTP livianas, igual que con
Groq.

## Tecnologías

Python · LangChain · Pinecone (RAG, embeddings integrados) · Groq (Llama 3.3 70B) · FastAPI · Render

## Instrucciones para ejecutar el proyecto

### Localmente

1. Clonar el repo y crear entorno virtual:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copiar `.env.example` a `.env` y completar con:
   - `GROQ_API_KEY` (gratis, sin tarjeta, en https://console.groq.com/keys)
   - `PINECONE_API_KEY` (gratis, sin tarjeta, en https://app.pinecone.io)
3. Los PDFs fuente ya están incluidos en `data/`.
4. Subir los documentos a Pinecone (solo la primera vez, o si cambian los PDFs):
   ```
   cd src && python vectorstore.py
   ```
5. Probar el agente por consola:
   ```
   python agent.py
   ```
6. Levantar la API localmente:
   ```
   uvicorn main:app --reload
   ```
   y abrir `http://127.0.0.1:8000` en el navegador.

### Deploy en Render

1. Conectar el repo de GitHub en [render.com](https://render.com) → New → Blueprint (detecta automáticamente `render.yaml`).
2. Configurar las variables de entorno (`GROQ_API_KEY`, `GROQ_MODEL`, `PINECONE_API_KEY`) en el dashboard de Render (Environment).
3. Deploy. Nota: el índice de Pinecone es persistente en la nube — no hace falta reconstruirlo en cada deploy, solo la primera vez desde tu máquina.
4. La primera petición tras inactividad puede tardar ~30-60s (spin-down del free tier).

## Ejemplos de preguntas y respuestas

Probado en la demo en vivo (https://alura-agente-santos-pegasus.onrender.com):

---

**Pregunta:** ¿Cuál es el protocolo de respuesta a incidentes?

**Respuesta:** El protocolo de respuesta a incidentes se establece en el documento "PROTOCOLO DE RESPUESTA A INCIDENTES Y POST-MORTEMS" de Santo Pegasus Soluciones, que incluye pasos y procedimientos para detectar, gestionar, mitigar y documentar incidentes de forma estructurada y eficiente. El protocolo cubre aspectos como la clasificación de severidad de incidentes, roles y responsabilidades, proceso de detección y alerta, respuesta al incidente, comunicación, resolución y cierre del incidente, entre otros. El documento también incluye una plantilla oficial de post-mortem y establece métricas de confiabilidad como SLI, SLO y SLA.

*Fuente: protocolo_incidencias.pdf*

---

**Pregunta:** ¿Qué tecnologías se usan en el back-end? / ¿Qué stack usa el equipo de ingeniería?

**Respuesta:** El equipo de ingeniería de Santo Pegasus Soluciones utiliza las siguientes tecnologías en el back-end: Java 17+, Spring Boot 3+, Spring Security, Spring Data JPA; bases de datos PostgreSQL, MongoDB y Redis; bases vectoriales Pinecone y Qdrant; e Inteligencia Artificial con LangChain (RAG) e integración con LLMs externos. La infraestructura cloud es AWS (RDS, SES, SQS, Secrets Manager, EC2/ECS), con Docker y Docker Compose para contenedorización, GitHub Actions para CI/CD, SLF4J/Logback, Prometheus y Datadog para observabilidad, SonarQube, JUnit 5, Mockito y Testcontainers para calidad de código, y Git con GitFlow y Conventional Commits.

*Fuentes: guia_ingenieria_backend.pdf, pegasus.pdf*

---

**Pregunta:** ¿Cómo está organizada la arquitectura de microservicios?

**Respuesta:** La arquitectura está organizada por dominio de negocio, con cada squad teniendo ownership completo de su ciclo de vida de software. Los microservicios son independientes y escalables individualmente. Se utiliza un API Gateway (Kong / AWS API GW) para routing, rate limiting, autenticación y terminación de SSL. La infraestructura se basa en AWS, con un mapa de dependencias entre servicios, patrones de comunicación definidos, estrategia de bases de datos, observabilidad distribuida, versionado de APIs y seguridad entre servicios.

*Fuente: arquitectura_microservicios.pdf*

---

**Pregunta:** ¿Qué debe hacer un nuevo desarrollador en su primer día?

**Respuesta:** Según el manual de onboarding, el primer día incluye la sección "Día 1 — Accesos y Cuentas". El checklist de onboarding (a completar junto con el Tech Lead antes de fin de la primera semana) incluye: correo corporativo (@santopegasus.com) funcionando, Slack instalado con los canales obligatorios unidos, y acceso a GitHub (organización privada) confirmado.

*Fuente: pegasus.pdf*

---
## Evidencia de deploy en Render

**URL pública:** https://alura-agente-santos-pegasus.onrender.com

