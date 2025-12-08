import os
import io
import time
import unicodedata
import re
from collections import Counter
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
    raise ValueError("ERRO: Chaves de API não encontradas no .env")

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "academy-ia"

# Checagem inicial
if index_name not in pc.list_indexes().names():
    raise ValueError(f"O Index '{index_name}' não foi encontrado no Pinecone. Verifique o nome.")
index = pc.Index(index_name)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- FUNÇÃO FAXINEIRA (ASCII ID) ---
def clean_filename(text):
    """Remove acentos e caracteres especiais para criar IDs compatíveis com Pinecone"""
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
    clean_text = re.sub(r'[^a-zA-Z0-9_.]', '', only_ascii)
    return clean_text


# --- 2. FUNÇÕES DO MANIFESTO ---
def get_manifest():
    try:
        result = index.fetch(ids=["manifesto_arquivos"])
        if result and "manifesto_arquivos" in result.vectors:
            metadata = result.vectors["manifesto_arquivos"].metadata
            files_str = metadata.get("file_list", "")
            if files_str:
                return files_str.split(";")
        return []
    except:
        return []

def update_manifest(filename, action="add"):
    current_files = get_manifest()
    
    if action == "add":
        if filename not in current_files:
            current_files.append(filename)
    elif action == "remove":
        if filename in current_files:
            current_files.remove(filename)
    
    # Vetor dummy não-zero para não dar erro
    dummy_vector = [0.01] * 768
    files_str = ";".join(current_files)
    
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
    except Exception as e:
        print(f"Erro na extração de texto: {e}")
        return ""

# --- 4. ROTAS ---

@app.get("/")
async def read_root(): return FileResponse('static/index.html')

@app.get("/admin")
async def read_admin(): return FileResponse('static/admin.html')

@app.get("/documents")
async def list_documents():
    files = get_manifest()
    doc_list = [{"name": f} for f in files]
    return {"documents": doc_list}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        index.delete(filter={"source": filename})
        update_manifest(filename, "remove")
        return {"status": "success", "message": f"{filename} removido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    ext = filename.lower()
    
    safe_id_name = clean_filename(filename)
    text = extract_text(contents, ext)
    
    if not text.strip(): 
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    # Remove anterior
    try: index.delete(filter={"source": filename})
    except: pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    vectors_to_upsert = []
    print(f"--- Iniciando Upload: {filename} ({len(chunks)} chunks) ---")
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk)
            chunk_id = f"{safe_id_name}_{i}"
            
            vectors_to_upsert.append({
                "id": chunk_id, 
                "values": vector, 
                "metadata": {"source": filename, "text": chunk}
            })
        except Exception as e:
            print(f"Erro vetorizando chunk {i}: {e}")
            continue

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[i:i+batch_size])
    
    update_manifest(filename, "add")
    print("--- Upload Concluído ---")
        
    return {"status": "Sucesso", "filename": filename, "chunks": len(chunks)}

class ChatMessage(BaseModel): message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatMessage):
    # --- MODO DEBUG SEM PROTEÇÃO (O SERVIDOR VAI PARAR SE DER ERRO) ---
    print("\n" + "="*30)
    print(f"DEBUG: Recebi pergunta: '{chat_req.message}'")
    
    # 1. Embedding
    print("DEBUG: Gerando embedding da pergunta...")
    q_embedding = get_embedding(chat_req.message)
    print(f"DEBUG: Embedding gerado. Tamanho: {len(q_embedding)} (Deve ser 768)")
    
    # 2. Busca
    print("DEBUG: Consultando Pinecone...")
    search_results = index.query(
        vector=q_embedding,
        top_k=20,
        include_metadata=True,
        filter={"source": {"$exists": True}} 
    )
    print(f"DEBUG: Pinecone retornou {len(search_results['matches'])} resultados.")

    # 3. Contexto
    retrieved = [m['metadata']['text'] for m in search_results['matches'] if 'text' in m['metadata']]
    
    if not retrieved:
        print("DEBUG: Nenhum texto encontrado nos metadados.")
        return {"response": "Não encontrei informações nos manuais."}

    context = "\n\n".join(retrieved)
    print(f"DEBUG: Contexto montado com {len(context)} caracteres.")
    
    # 4. Prompt
    sys_inst = """
    Você é um assistente de suporte especializado em funcionalidades da plataforma de e-commerce da CWS.
    
    IMPORTANTE: Os manuais podem conter transcrições de reuniões com linguagem informal.
    SUA TAREFA: Ignorar a "conversa fiada", extraia apenas a informação técnica e responda de forma profissional.

    DIRETRIZES:
    1. BASE NO CONTEXTO: Use as informações técnicas do contexto para formular suas respostas.
    2. SÍNTESE PERMITIDA: Se o usuário perguntar um conceito e o contexto tiver instruções de uso, explique o conceito baseando-se nas funcionalidades.
    3. SEM ALUCINAÇÃO: Não invente funcionalidades.
    4. LEI ZERO (Fallback): Se a informação for insuficiente, use a mensagem padrão: "Puxa, parece que não encontrei detalhes suficientes sobre essa funcionalidade na documentação que estou consultando. Recomendo entrar em contato com o suporte da CWS para obter informações mais detalhadas!"
    """
    
    model = genai.GenerativeModel('gemini-flash-latest')
    final_prompt = f"{sys_inst}\n\nCONTEXTO:\n{context}\n\nPERGUNTA:\n{chat_req.message}"
    
    print("DEBUG: Enviando para o Gemini...")
    response = model.generate_content(final_prompt)
    print("DEBUG: Resposta recebida!")
    print("="*30 + "\n")
    
    return {"response": response.text}