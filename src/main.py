"""
Capa 4: API HTTP.

Envuelve agent.py en un endpoint FastAPI (POST /ask) para poder desplegar
el agente en Render y que sea accesible públicamente. También sirve una
página HTML simple en "/" para poder hacer la demo visualmente.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import ask

app = FastAPI(title="Alura Agente - Santos Pegasus Soluciones")


class Question(BaseModel):
    question: str


HTML_PAGE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Alura Agente - Santos Pegasus Soluciones</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
  h1 { font-size: 1.4rem; }
  textarea { width: 100%; box-sizing: border-box; padding: 10px; font-size: 1rem; }
  button { margin-top: 10px; padding: 10px 20px; font-size: 1rem; cursor: pointer; }
  #respuesta { margin-top: 20px; padding: 15px; background: #f4f4f4; border-radius: 8px; white-space: pre-wrap; }
  #fuentes { margin-top: 10px; font-size: 0.85rem; color: #555; }
  .cargando { color: #888; }
</style>
</head>
<body>
  <h1>Agente RAG — Santos Pegasus Soluciones</h1>
  <p>Pregunta algo sobre onboarding, back-end, front-end, incidentes o arquitectura.</p>
  <textarea id="pregunta" rows="3" placeholder="Ej: ¿Cuál es el protocolo de respuesta a incidentes?"></textarea>
  <br>
  <button onclick="preguntar()">Preguntar</button>
  <div id="respuesta"></div>
  <div id="fuentes"></div>

<script>
async function preguntar() {
  const pregunta = document.getElementById("pregunta").value;
  const respuestaDiv = document.getElementById("respuesta");
  const fuentesDiv = document.getElementById("fuentes");
  if (!pregunta.trim()) return;

  respuestaDiv.textContent = "Pensando...";
  respuestaDiv.className = "cargando";
  fuentesDiv.textContent = "";

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: pregunta })
    });
    const data = await res.json();
    respuestaDiv.className = "";
    respuestaDiv.textContent = data.answer;
    fuentesDiv.textContent = "Fuentes: " + data.sources.join(", ");
  } catch (err) {
    respuestaDiv.className = "";
    respuestaDiv.textContent = "Error al conectar con el agente.";
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/ask")
def ask_agent(payload: Question):
    result = ask(payload.question)
    return result