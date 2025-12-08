import os
import io
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Processamento
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import docx
from pptx import Presentation

# Banco Nuvem
from pinecone import Pinecone

# --- 1. CONFIGURAÇÃO ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("ERRO: Chaves de API não encontradas.")

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "academy-ia"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' não encontrado.")
index = pc.Index(index_name)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 2. FUNÇÕES DO MANIFESTO (LISTA MESTRA) ---
# Usamos um vetor especial com ID 'manifesto_arquivos' para guardar a lista de nomes

def get_manifest():
    """Recupera a lista de arquivos salvos"""
    try:
        # Busca o vetor especial
        result = index.fetch(ids=["manifesto_arquivos"])
        if result and "manifesto_arquivos" in result.vectors:
            metadata = result.vectors["manifesto_arquivos"].metadata
            # O Pinecone guarda listas como strings separadas se não for nativo, 
            # mas vamos usar o campo 'file_list' como string separada por ; para garantir
            files_str = metadata.get("file_list", "")
            if files_str:
                return files_str.split(";")
        return []
    except:
        return []

def update_manifest(filename, action="add"):
    """Atualiza a lista mestra (Adiciona ou Remove)"""
    current_files = get_manifest()
    
    if action == "add":
        if filename not in current_files:
            current_files.append(filename)
    elif action == "remove":
        if filename in current_files:
            current_files.remove(filename)
    
    # Cria um vetor "bobo" de zeros só para segurar o metadado (768 dimensões)
    dummy_vector = [0.0] * 768
    files_str = ";".join(current_files)
    
    # Salva no Pinecone
    index.upsert(vectors=[{
        "id": "manifesto_arquivos",
        "values": dummy_vector,
        "metadata": {"file_list": files_str, "type": "manifest"}
    }])


# --- 3. FUNÇÕES AUXILIARES ---

def get_embedding(text):
    return genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )['embedding']

def extract_text(contents, ext):
    text = ""
    try:
        if ext.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
            for page in pdf_reader.pages: text += page.extract_text() or ""
        elif ext.endswith('.docx'):
            doc = docx.Document(io.BytesIO(contents))
            text = "\n".join([p.text for p in doc.paragraphs])
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells: text.append(cell.text)
        elif ext.endswith('.pptx'):
            prs = Presentation(io.BytesIO(contents))
            full_text = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"): full_text.append(shape.text)
            text = "\n".join(full_text)
        elif ext.endswith('.md') or ext.endswith('.txt'):
            text = contents.decode("utf-8")
        return text
    except:
        return ""

# --- 4. ROTAS ---

@app.get("/")
async def read_root(): return FileResponse('static/index.html')

@app.get("/admin")
async def read_admin(): return FileResponse('static/admin.html')

@app.get("/documents")
async def list_documents():
    """Retorna a lista de nomes do manifesto"""
    files = get_manifest()
    # Formata para o frontend
    doc_list = [{"name": f} for f in files]
    return {"documents": doc_list}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        # 1. Deleta os vetores de conteúdo
        index.delete(filter={"source": filename})
        # 2. Remove do manifesto
        update_manifest(filename, "remove")
        return {"status": "success", "message": f"{filename} removido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    ext = filename.lower()
    
    text = extract_text(contents, ext)
    if not text.strip(): raise HTTPException(status_code=400, detail="Arquivo vazio.")

    # Remove anterior
    try: index.delete(filter={"source": filename})
    except: pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk)
            vectors_to_upsert.append({
                "id": f"{filename}_{i}", 
                "values": vector, 
                "metadata": {"source": filename, "text": chunk}
            })
        except: continue

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[i:i+batch_size])
    
    # ATUALIZA O MANIFESTO
    update_manifest(filename, "add")
        
    return {"status": "Sucesso", "filename": filename, "chunks": len(chunks)}

class ChatMessage(BaseModel): message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatMessage):
    try:
        q_embedding = get_embedding(chat_req.message)
        # Filtra para não pegar o vetor do manifesto na busca
        search_results = index.query(
            vector=q_embedding, top_k=20, include_metadata=True,
            filter={"source": {"$exists": True}} 
        )

        retrieved = [m['metadata']['text'] for m in search_results['matches'] if 'text' in m['metadata']]
        if not retrieved: return {"response": "Não encontrei informações nos manuais."}

        context = "\n\n".join(retrieved)
        
        sys_inst = """Você é um assistente técnico da CWS.
        Ignore linguagem informal, foque no conteúdo técnico.
        Se não souber, diga que não encontrou."""
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        final_prompt = f"{sys_inst}\n\nCONTEXTO:\n{context}\n\nPERGUNTA:\n{chat_req.message}"
        
        response = model.generate_content(final_prompt)
        return {"response": response.text}

    except Exception as e:
        return {"response": "Erro técnico."}