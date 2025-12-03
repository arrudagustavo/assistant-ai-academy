import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carrega a chave do .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print(f"--- Usando API Key: ...{api_key[-5:]} ---")
print("--- LISTA DE MODELOS DISPONÍVEIS PARA VOCÊ ---")

try:
    # Lista todos os modelos disponíveis
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
except Exception as e:
    print(f"ERRO FATAL: {e}")