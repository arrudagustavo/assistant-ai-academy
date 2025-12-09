import os
import io
import time
import unicodedata
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Processamento de Arquivos
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import docx
from pptx import Presentation

# Banco Vetorial (Nuvem)
from pinecone import Pinecone

# --- 1. CONFIGURAÇÃO ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("ERRO CRÍTICO: Chaves de API não encontradas no arquivo .env")

genai.configure(api_key=GOOGLE_API_KEY)

# Configuração do Modelo
# Usando o 'gemini-flash-lite-latest' conforme sua lista de permissões.
# Ele é otimizado para velocidade e menor consumo de cota.
model = genai.GenerativeModel('models/gemini-flash-lite-latest')

# Configuração Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "academy-ia"

if index_name not in pc.list_indexes().names():
    raise ValueError(f"O Index '{index_name}' não foi encontrado no Pinecone.")
index = pc.Index(index_name)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- FUNÇÕES AUXILIARES ---

def clean_filename(text):
    """Remove acentos e caracteres especiais para criar IDs compatíveis com Pinecone"""
    nfkd_form = unicodedata.normalize('NFKD', text)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
    # Mantém apenas letras, números, underline e ponto
    clean_text = re.sub(r'[^a-zA-Z0-9_.]', '', only_ascii)
    return clean_text

def get_embedding(text):
    """Gera o vetor numérico (768 dimensões)"""
    return genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )['embedding']

def extract_text(contents, ext):
    """Extrai texto de vários formatos"""
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
                    for cell in row.cells: text += " " + cell.text
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
        print(f"Erro na extração: {e}")
        return ""


# --- GESTÃO DO MANIFESTO (LISTA DE ARQUIVOS) ---

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
    
    # Vetor dummy não-zero (0.01) para evitar erro do Pinecone
    dummy_vector = [0.01] * 768
    files_str = ";".join(current_files)
    
    index.upsert(vectors=[{
        "id": "manifesto_arquivos",
        "values": dummy_vector,
        "metadata": {"file_list": files_str, "type": "manifest"}
    }])


# --- ROTAS ---

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/admin")
async def read_admin():
    return FileResponse('static/admin.html')

@app.get("/documents")
async def list_documents():
    files = get_manifest()
    doc_list = [{"name": f} for f in files]
    return {"documents": doc_list}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        # Deleta vetores do arquivo
        index.delete(filter={"source": filename})
        # Atualiza lista visual
        update_manifest(filename, "remove")
        return {"status": "success", "message": f"{filename} removido."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    ext = filename.lower()
    
    # Gera ID limpo para o Pinecone
    safe_id_name = clean_filename(filename)

    text = extract_text(contents, ext)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Arquivo vazio ou ilegível.")

    # Remove versão anterior se existir
    try: index.delete(filter={"source": filename})
    except: pass

    # Chunking
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

    # Envia em lotes
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        index.upsert(vectors=vectors_to_upsert[i:i+batch_size])
    
    update_manifest(filename, "add")
    print("--- Upload Concluído ---")
        
    return {"status": "Sucesso", "filename": filename, "chunks": len(chunks)}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatMessage):
    # --- MODO DEBUG: Logs no terminal para rastreio ---
    print("\n" + "="*30)
    print(f"DEBUG: Recebi pergunta: '{chat_req.message}'")
    
    try:
        # 1. Embedding
        print("DEBUG: Gerando embedding...")
        q_embedding = get_embedding(chat_req.message)
        
        # 2. Busca no Pinecone (TOP K = 20 Mantido)
        print("DEBUG: Consultando Pinecone (Top 20)...")
        search_results = index.query(
            vector=q_embedding,
            top_k=20,
            include_metadata=True,
            filter={"source": {"$exists": True}} 
        )
        print(f"DEBUG: Pinecone retornou {len(search_results['matches'])} resultados.")

        # 3. Montagem do Contexto
        retrieved = [m['metadata']['text'] for m in search_results['matches'] if 'text' in m['metadata']]
        
        if not retrieved:
            print("DEBUG: Nenhum contexto encontrado.")
            return {"response": "Não encontrei informações suficientes nos manuais para responder sua pergunta."}

        context = "\n\n".join(retrieved)
        print(f"DEBUG: Contexto montado com {len(context)} caracteres.")
        
        # 4. Prompt do Sistema
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
        
        final_prompt = f"{sys_inst}\n\nCONTEXTO:\n{context}\n\nPERGUNTA:\n{chat_req.message}"
        
        # 5. Geração com Gemini
        print("DEBUG: Enviando para o Gemini...")
        response = model.generate_content(final_prompt)
        print("DEBUG: Resposta recebida com sucesso!")
        print("="*30 + "\n")
        
        return {"response": response.text}

    except Exception as e:
        # Captura erro real, imprime no terminal e devolve aviso no chat
        print(f"ERRO CRÍTICO NO CHAT: {e}")
        return {"response": f"Ocorreu um erro técnico ao processar sua solicitação: {str(e)}"}