"""Script para corrigir thesis_type e thesis_text do ART-001."""
import requests
import re
from pymilvus import connections, Collection

GPU_SERVER = 'http://195.26.233.70:55278'
SYSTEM_PROMPT = 'Voce e um assistente especializado em direito administrativo brasileiro. Responda sempre em portugues do Brasil, de forma direta e concisa. /no_think'

def clean_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()

def parse_thesis(text):
    """Parse thesis response extraindo tipo e resumo."""
    text = text.strip()
    text = re.sub(r'\*\*', '', text)

    tipos_validos = ['definicao', 'procedimento', 'obrigacao', 'proibicao', 'excecao', 'prazo']
    thesis_type = 'disposicao'
    thesis_text = text

    tipo_match = re.search(r'TIPO:?\s*(\w+)', text, re.IGNORECASE)
    if tipo_match:
        tipo_encontrado = tipo_match.group(1).lower()
        if tipo_encontrado in tipos_validos:
            thesis_type = tipo_encontrado

    resumo_match = re.search(r'RESUMO:?\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if resumo_match:
        thesis_text = resumo_match.group(1).strip()
    else:
        thesis_text = re.sub(r'^.*?TIPO:?\s*\w+\s*', '', text, flags=re.IGNORECASE).strip()

    return thesis_type, thesis_text

def main():
    connections.connect(host='77.37.43.160', port='19530')
    col = Collection('leis_v3')
    col.load()

    chunk = col.query(expr='chunk_id == "IN-65-2021#ART-001"', output_fields=['*'], limit=1)
    if not chunk:
        print('ART-001 nao encontrado!')
        return

    chunk = chunk[0]
    chunk_id = chunk['chunk_id']
    text = chunk['text'][:2000]
    print(f'Corrigindo: {chunk_id}')
    print(f'Texto: {text[:200]}...')

    try:
        resp = requests.post(f'{GPU_SERVER}/llm/chat', json={
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Analise este dispositivo legal:\n\n{text}\n\nResponda no formato:\nTIPO: [uma palavra: definicao/procedimento/obrigacao/proibicao/excecao/prazo]\nRESUMO: [uma frase resumindo o que o artigo determina]\n\nNao use markdown.'}
            ],
            'temperature': 0.0, 'max_tokens': 150
        }, timeout=120)

        thesis_resp = clean_response(resp.json()['content'])
        print(f'Resposta LLM: {thesis_resp}')

        thesis_type, thesis_text = parse_thesis(thesis_resp)
        print(f'thesis_type: {thesis_type}')
        print(f'thesis_text: {thesis_text}')

        thesis_embed = requests.post(f'{GPU_SERVER}/embed/hybrid', json={'texts': [thesis_text]}, timeout=60)
        thesis_vector = thesis_embed.json()['dense'][0]

        # Prepara row sem 'id'
        row = {}
        for k, v in chunk.items():
            if k != 'id':
                row[k] = v

        row['thesis_type'] = thesis_type
        row['thesis_text'] = thesis_text[:5000]
        row['thesis_vector'] = thesis_vector

        col.delete(expr=f'chunk_id == "{chunk_id}"')
        col.insert([row])
        col.flush()

        print('\nART-001 corrigido com sucesso!')

    except Exception as e:
        print(f'ERRO: {e}')

    connections.disconnect('default')

if __name__ == '__main__':
    main()
