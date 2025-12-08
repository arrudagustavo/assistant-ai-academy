import os
import io
import time
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

# Banco de Dados Vetorial (Nuvem)
from pinecone import Pinecone

# --- 1. CONFIGURAÇÃO ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("ERRO: GOOGLE_API_KEY não encontrada.")
if not PINECONE_API_KEY:
    raise ValueError("ERRO: PINECONE_API_KEY não encontrada.")

# Configura Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configura Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "academy-ia" # O nome exato que você criou no painel do Pinecone

# Verifica conexão com o índice
if index_name not in pc.list_indexes().names():
    raise ValueError(f"O Index '{index_name}' não foi encontrado no Pinecone.")
index = pc.Index(index_name)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- 2. FUNÇÕES AUXILIARES ---

def get_embedding(text):
    """Gera o vetor numérico (768 dimensões) usando o Google"""
    # O modelo 'text-embedding-004' é otimizado para RAG
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

def extract_text(contents, ext):
    """Extrai texto de PDF, DOCX, PPTX, MD, TXT"""
    text = ""
    try:
        if ext.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif ext.endswith('.docx'):
            doc = docx.Document(io.BytesIO(contents))
            text = "\n".join([p.text for p in doc.paragraphs])
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text.append(cell.text)
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
        print(f"Erro extração: {e}")
        return ""

# --- 3. ROTAS ---

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/admin")
async def read_admin():
    return FileResponse('static/admin.html')

@app.get("/documents")
async def list_documents():
    """
    Lista estatísticas do Pinecone.
    Nota: O Pinecone não permite listar nomes de arquivos facilmente como o banco local.
    Mostraremos o total de vetores ativos na nuvem.
    """
    try:
        stats = index.describe_index_stats()
        count = stats['total_vector_count']
        # Retorna um item genérico representando a base na nuvem
        return {"documents": [{"name": "Base de Conhecimento (Nuvem Pinecone)", "count": count}]}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Deleta vetores pelo metadata 'source'"""
    try:
        # Deleta todos os vetores que vieram desse arquivo
        index.delete(filter={"source": filename})
        return {"status": "success", "message": f"Memória de {filename} apagada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    ext = filename.lower()
    
    # 1. Extração
    text = extract_text(contents, ext)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Arquivo vazio ou ilegível.")

    # 2. Limpeza prévia (Evita duplicidade na nuvem)
    try:
        index.delete(filter={"source": filename})
    except:
        pass 

    # 3. Chunking (Picotar o texto)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # 4. Vetorização e Upload (Upsert)
    vectors_to_upsert = []
    
    for i, chunk in enumerate(chunks):
        try:
            vector = get_embedding(chunk)
            chunk_id = f"{filename}_{i}"
            # Prepara pacote para o Pinecone: ID, Vetor e Metadados (Texto original + Nome arquivo)
            vectors_to_upsert.append({
                "id": chunk_id, 
                "values": vector, 
                "metadata": {"source": filename, "text": chunk}
            })
        except Exception as e:
            print(f"Erro vetorizar chunk {i}: {e}")
            continue

    # Envia em lotes de 100 para não sobrecarregar a rede
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        index.upsert(vectors=batch)
        
    return {"status": "Sucesso", "filename": filename, "chunks_created": len(chunks)}

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatMessage):
    try:
        # 1. Transforma a pergunta do usuário em números (Vetor)
        question_embedding = get_embedding(chat_req.message)

        # 2. Busca no Pinecone (20 resultados mais relevantes)
        search_results = index.query(
            vector=question_embedding,
            top_k=20,
            include_metadata=True
        )

        # 3. Extrai o texto dos resultados
        retrieved_texts = []
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                retrieved_texts.append(match['metadata']['text'])
        
        # Se não achou nada relevante
        if not retrieved_texts:
            return {"response": "Puxa, parece que não encontrei detalhes suficientes sobre essa funcionalidade na documentação que estou consultando. Recomendo entrar em contato com o suporte da CWS para obter informações mais detalhadas!"}

        context = "\n\n".join(retrieved_texts)

        # 4. Prompt do Sistema (Suas Regras de Ouro)
        system_instruction = """
        Você é um assistente de suporte especializado em funcionalidades da plataforma de e-commerce da CWS. Seu tom é **profissional, amigável e prestativo**.

        IMPORTANTE: Os manuais podem conter transcrições de reuniões com linguagem informal ("né", "aí", "beleza").
        SUA TAREFA: Ignorar a conversa fiada, extrair apenas a informação técnica e responder de forma profissional.

        Diretrizes de Aderência e Precisão:
        1. BASE NO CONTEXTO: Use as informações técnicas do contexto para formular suas respostas.
        2. SÍNTESE PERMITIDA: Se o usuário perguntar um conceito e o contexto tiver instruções de uso, explique o conceito baseando-se nas funcionalidades.
        3. SEM ALUCINAÇÃO: Não invente funcionalidades.
        4. LEI ZERO (Fallback): Se a informação for insuficiente, use a mensagem padrão: "Puxa, parece que não encontrei detalhes suficientes sobre essa funcionalidade na documentação que estou consultando. Recomendo entrar em contato com o suporte da CWS para obter informações mais detalhadas!"

        Estilo e Estrutura:
        5. TOM: Use um tom amigável. Comece com uma saudação e termine com cortesia.
        6. LINGUAGEM: Simples e direta.
        7. CAMINHOS: Use Menu > Submenu > Opção se houver essa informação.
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        
        final_prompt = f"""
        {system_instruction}

        --- INÍCIO DO CONTEXTO RAG (DOCUMENTAÇÃO OFICIAL) ---
        {context}
        --- FIM DO CONTEXTO RAG ---

        PERGUNTA DO CLIENTE:
        {chat_req.message}
        """
        
        response = model.generate_content(final_prompt)
        return {"response": response.text}

    except Exception as e:
        print(f"Erro chat: {e}")
        return {"response": "Desculpe, estou enfrentando uma instabilidade técnica momentânea. Poderia tentar novamente em alguns instantes?"}