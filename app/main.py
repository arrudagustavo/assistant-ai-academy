import os
import io
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import docx
from pptx import Presentation

# --- CONFIGURAÇÃO ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: API Key não encontrada no .env")

genai.configure(api_key=GOOGLE_API_KEY)

chroma_client = chromadb.PersistentClient(path="chroma_db")

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        model = "models/text-embedding-004"
        return [
            genai.embed_content(model=model, content=text, task_type="retrieval_document")['embedding']
            for text in input
        ]

collection = chroma_client.get_or_create_collection(
    name="manuais_ecommerce",
    embedding_function=GeminiEmbeddingFunction()
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- ROTAS DE PÁGINAS (FRONTEND) ---

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

# AQUI ESTÁ A CORREÇÃO: Criamos a rota /admin explicitamente
@app.get("/admin")
async def read_admin():
    return FileResponse('static/admin.html')


# --- FUNÇÕES DE EXTRAÇÃO ---
def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text.append(cell.text)
    return "\n".join(text)

def extract_text_from_pptx(file_bytes):
    prs = Presentation(io.BytesIO(file_bytes))
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

# --- ROTAS DA API ---

@app.get("/documents")
async def list_documents():
    try:
        data = collection.get()
        metadatas = data['metadatas'] or []
        sources = [m['source'] for m in metadatas if 'source' in m]
        counts = Counter(sources)
        doc_list = [{"name": name, "count": count} for name, count in counts.items()]
        return {"documents": doc_list}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        collection.delete(where={"source": filename})
        return {"status": "success", "message": f"Arquivo {filename} removido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    text = ""

    try:
        collection.delete(where={"source": filename})

        ext = filename.lower()
        if ext.endswith('.pdf'): text = extract_text_from_pdf(contents)
        elif ext.endswith('.docx'): text = extract_text_from_docx(contents)
        elif ext.endswith('.pptx'): text = extract_text_from_pptx(contents)
        elif ext.endswith('.md') or ext.endswith('.txt'): text = contents.decode("utf-8")
        else: raise HTTPException(status_code=400, detail="Formato não suportado.")

        if not text.strip(): raise HTTPException(status_code=400, detail="Arquivo vazio.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in range(len(chunks))]

        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        
        return {"status": "Sucesso", "filename": filename, "chunks_created": len(chunks)}
    except Exception as e:
        print(f"Erro upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatMessage):
    try:
        results = collection.query(query_texts=[chat_req.message], n_results=5)
        retrieved_texts = results['documents'][0]
        
        # Se não achou nada, tenta responder com lei zero
        if not retrieved_texts:
            context = "Nenhuma informação relevante encontrada nos manuais."
        else:
            context = "\n\n".join(retrieved_texts)

        # DEBUG NO TERMINAL (Para você ver o que ele leu)
        print("\n--- CONTEXTO RAG ---")
        print(context[:500] + "..." if len(context) > 500 else context)
        print("--------------------\n")

        # PROMPT OTIMIZADO (Síntese Permitida)
        system_instruction = """
        Você é um assistente de suporte especializado em funcionalidades da plataforma de e-commerce da CWS.
        
        DIRETRIZES:
        1. BASE: Responda usando o contexto abaixo.
        2. SÍNTESE: Se o usuário pedir um conceito (Ex: "O que é Cuponeria?") e o texto tiver apenas instruções de uso, você PODE explicar o conceito com base nas funcionalidades descritas.
        3. HONESTIDADE: Se o contexto não tiver NADA a ver com a pergunta, diga: "Puxa, parece que não encontrei detalhes suficientes sobre isso na documentação."
        4. ESTILO: Profissional, amigável e direto.
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        
        final_prompt = f"""
        {system_instruction}

        CONTEXTO DA DOCUMENTAÇÃO:
        {context}

        PERGUNTA DO CLIENTE:
        {chat_req.message}
        """
        
        response = model.generate_content(final_prompt)
        return {"response": response.text}

    except Exception as e:
        print(f"Erro chat: {e}")
        return {"response": "Erro técnico momentâneo."}