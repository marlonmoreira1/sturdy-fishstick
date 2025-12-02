import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime
import time
import json
import re
import google.generativeai as genai
from google.cloud import storage
from google.oauth2 import service_account
import os



# ============================================
# FUNÇÕES
# ============================================

def carregar_canais(csv_path, limite=2500):
    """Carrega canais do CSV"""
    df = pd.read_csv(csv_path, sep=';')
    print(f"Total de canais no CSV: {len(df)}")
    df_teste = df.head(limite)
    print(f"Usando {len(df_teste)} canais para teste")
    return df_teste

def converter_para_playlist_id(channel_id):
    """UC... -> UU..."""
    if channel_id.startswith('UC'):
        return 'UU' + channel_id[2:]
    return channel_id

def buscar_video_ids_canal(channel_id, youtube_api_key):
    """Busca todos os video IDs de um canal"""
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    playlist_id = converter_para_playlist_id(channel_id)
    
    video_ids = []
    next_page_token = None
    
    while True:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    
    return video_ids

def buscar_metadados_videos(video_ids, youtube_api_key):
    """Busca metadados dos vídeos em batches de 50"""
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    videos_data = []
    
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(batch)
        )
        response = request.execute()
        
        for item in response['items']:
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            content_details = item.get('contentDetails', {})
            
            videos_data.append({
                'video_id': item['id'],
                'url': f"https://www.youtube.com/watch?v={item['id']}",
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'channel_id': snippet.get('channelId'),
                'channel_name': snippet.get('channelTitle'),
                'published_at': snippet.get('publishedAt'),
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url'),
                'viewCount': int(statistics.get('viewCount', 0)),
                'likeCount': int(statistics.get('likeCount', 0)),
                'commentCount': int(statistics.get('commentCount', 0)),
                'defaultAudioLanguage': snippet.get('defaultAudioLanguage'),
                'duration': content_details.get('duration'),
                'tags': snippet.get('tags', [])
            })
    
    return videos_data

def filtrar_por_data(df, data_minima='2024-06-01'):
    """Filtra vídeos de junho/2024 para cá"""
    df['published_at'] = pd.to_datetime(df['published_at'])
    df_filtrado = df[df['published_at'] >= data_minima].copy()
    print(f"Vídeos após filtro de data (>= {data_minima}): {len(df_filtrado)}")
    return df_filtrado




def contextualizar_videos_groq(df, groq_api_key, limite=100):
    """Classifica vídeos com Groq (Llama 3.1) - com limite"""    
    
    genai.configure(api_key=groq_api_key)
    
    df_para_classificar = df.copy()
    print(f"\nClassificando {len(df_para_classificar)} vídeos com Groq...")
    
    classificacoes = []
    
    for idx, row in df_para_classificar.iterrows():
        prompt = f"""Você é um contextualizador técnico avançado de vídeos educacionais de tecnologia.
Sua função é ler o título, descrição e nome do canal e produzir uma sinopse técnica limpa, eliminando todo ruído.

======================================================
OBJETIVO
======================================================
Gerar um resumo técnico confiável, eliminando completamente ruídos promocionais e elementos irrelevantes, deixando apenas os dados úteis para que modelos futuros consigam classificar corretamente qual tecnologia e operação o vídeo ensina.

======================================================
DETECÇÃO NÃO TÉCNICO
======================================================
Antes de gerar a sinopse, determine se o vídeo é realmente técnico.

O vídeo NÃO É TÉCNICO quando:

- não há demonstração, explicação ou ensino de tecnologia

- o conteúdo serve apenas para:

    - marketing, divulgação, anúncio, promoção, venda

    - sorteios, eventos, chamadas de live

    - carreira, mindset, motivação, trajetória

    - memes, humor, storytelling, dramatização

    - opinião, review, comparação de cursos/serviços

    - vlog, rotina, dia a dia

    - temas aspiracionais (ex.: “destrave sua carreira”, “oportunidade”, “novo lote”, etc.)

📌 Se o vídeo NÃO for técnico, retorne EXATAMENTE:

"O vídeo não apresenta conteúdo técnico ensinável."

E nada mais.

NÃO gere sinopse.
NÃO cite tecnologia.
NÃO tente extrair nada técnico.

======================================================
REGRAS ABSOLUTAS
======================================================
1. **Não invente tecnologias.**  
   Só cite ferramentas, bibliotecas, frameworks ou conceitos SE estiverem
   explicitamente presentes no título ou descrição (que não sejam partes de trechos a serem ignorados).

2. **IGNORE COMPLETAMENTE** qualquer trecho que não seja técnico:  
   - links  
   - redes sociais  
   - cursos  
   - pedidos de inscrição
   - dicas de carreira
   - eventos
   - review de ferramenta
   - autopromoções  
   - anúncios  
   - emojis  
   - listas genéricas de palavras-chave  
   - agradecimentos  
   - textos motivacionais  
   - conversa paralela  
   - conteúdo sem relação com tecnologia  

3. O nome do canal **NUNCA é prova** de qual tecnologia o vídeo usa.  
   Use-o apenas como reforço contextual (ex.: canal dedicado a Excel → reforça, mas não prova).

4. Nunca classifique trilha, não classifique ferramenta final, não gere JSON.

5. A sinopse deve ser 100% técnica, objetiva e orientada ao que é ENSINADO no vídeo.

6. Se houver ambiguidade, escolha SEMPRE a interpretação mais específica sustentada pelo texto.

======================================================
ENTRADAS DO VÍDEO
======================================================
Título: {row['title']}
Descrição: {row['description'] if row['description'] else 'Sem descrição'}
Nome do canal: {row['channel_name']}

======================================================
SAÍDA OBRIGATÓRIA (EM JSON)
======================================================

Produza **apenas um parágrafo de sinopse técnica**, com no máximo 8–12 linhas,
contendo:

- a ferramenta principal citada
- subferramentas / bibliotecas mencionadas
- conceitos técnicos centrais que o vídeo explica
- a operação prática demonstrada
- qualquer detalhe técnico que ajude o classificador a entender "o que está sendo ensinado"
- absolutamente nenhum ruído promocional

O texto deve parecer uma descrição de conteúdo feita por um analista técnico.

==========================================================
EXEMPLO DE ESTILO (não copie o conteúdo, apenas o formato)
===========================================================

"O vídeo demonstra como utilizar Python para consumir uma API REST, explicando como enviar requisições GET e interpretar respostas JSON. O autor utiliza a biblioteca requests e mostra como tratar códigos de status, converter dados para dicionários e exibir campos específicos. O foco está na construção do script, testes de chamada e boas práticas de organização do código."

AGORA GERE A SINOPSE TÉCNICA BASEADA NOS DADOS DO VÍDEO.
"""
        
        try:            
            model = genai.GenerativeModel('gemma-3-27b-it')
            response = model.generate_content(prompt) 
            classificacao = response.text                    
            classificacoes.append(classificacao)
            
            if (idx + 1) % 10 == 0:
                print(f"Classificados: {idx + 1}/{len(df_para_classificar)}")            
            
            
        except Exception as e:
            print(f"Erro ao classificar vídeo {row['video_id']}: {e}")
            classificacoes.append("erro")

        time.sleep(3)
    
    df_para_classificar['contexto'] = classificacoes
    return df_para_classificar






def classificar_videos_groq(df, groq_api_key, limite=100):
    """Classifica vídeos com Groq (Llama 3.1) - com limite"""    
    
    genai.configure(api_key=groq_api_key)    

    
    df_para_classificar = df.copy()
    print(f"\nClassificando {len(df_para_classificar)} vídeos com Groq...")
    
    classificacoes = []
    
    for idx, row in df_para_classificar.iterrows():
        prompt = f"""Você é um especialista em classificação de conteúdo educacional de tecnologia e programação do YouTube brasileiro.
                    Você receberá APENAS uma SINOPSE TÉCNICA PURIFICADA — um texto curto,
                    objetivo, sem ruído, descrevendo exatamente o que o vídeo ensina.
                    Essa sinopse já removeu promoções, links, tags irrelevantes e palavras-chave de SEO.

**OBJETIVO:**
Extrair a FERRAMENTA PRINCIPAL ensinada no vídeo da sinopse técnica fornecida,
seguindo exclusivamente a lista de tecnologias aceitas do sistema.

**REGRAS CRÍTICAS:**
1. Use SOMENTE o que está explícito na sinopse.
2. NÃO invente tecnologias.
3. NUNCA invente ou presuma tecnologias não mencionadas


**LISTA TECNOLOGIAS ACEITAS (use EXATAMENTE estes nomes):**
Python | Java | C | C++ | JavaScript | TypeScript | PHP | Go | Rust | Kotlin | Swift | SQL | HTML | CSS
React | Angular | Vue | Next.js | Node.js | Spring Boot | FastAPI | Express | GraphQL | Flutter | Tailwind CSS | Jetpack Compose | Vite | Pandas | dbt | Spark | TensorFlow | MLflow | Laravel
Machine Learning | Deep Learning 
MongoDB
Linux
Docker | Kubernetes | Airflow | Jenkins | GitHub Actions | Terraform
AWS | Azure Data Factory | GCP Dataflow
Excel | Power BI | Tableau | Grafana
RabbitMQ | Kafka
JWT | OAuth2
Prometheus | Grafana
Xcode | SwiftUI | React Native
Git | REST APIs | Cypress | Postman | REST APIs | Selenium | Cypress | JUnit | Espresso | JMeter

---

**VÍDEO A ANALISAR:**
Sinopse Técnica: {row['contexto']}

---

🧠 INFERÊNCIA PERMITIDA:
Use seu conhecimento prévio para identificar relações entre ferramentas e suas tecnologias base
- BullMQ → roda em Node.js → tecnologia_base: Node.js
- Pandas → biblioteca Python → tecnologia_base: Python
- DAX → linguagem do Power BI → tecnologia_base: Power BI
- nftables → comando do Linux → tecnologia_base: Linux
- Express → framework Node.js → tecnologia_base: Node.js
- VBA → roda em excel → tecnologia_base: Excel


🎯 REGRAS:
- Classifique sempre no nível da TECNOLOGIA PRINCIPAL (não o comando ou conceito).
- Evite conceitos abstratos (loops, algoritmos, ponteiros).
- Se o vídeo ensinar uma funcionalidade de uma tecnologia, classifique pela tecnologia.
- Se houver dúvida entre duas, escolha a mais abrangente.
- Utilize inferência técnica para relacionar ferramentas e ecossistemas.

**RESPONDA APENAS COM JSON (sem markdown, sem explicações):**

{{
    "ferramenta_principal": "nome_exato_da_lista_ou_invalido",
    "tecnologia_base": "tecnologia_mais_ampla_ou_ecossistema_da_lista",
    "confianca": "alta/media/baixa",
    "categoria": "linguagem/framework/sistema_operacional/banco_dados/cloud/bi_analytics/outra"
}}

Se a sinopse técnica não fornecer nenhum termo técnico
ou pista inequívoca que identifique uma ferramenta da lista,
a classificação DEVE ser "invalido".
"""
        
        try:
            model = genai.GenerativeModel('gemma-3-27b-it')
            response = model.generate_content(prompt) 
            classificacao = response.text          
            classificacoes.append(classificacao)
            
            if (idx + 1) % 10 == 0:
                print(f"Classificados: {idx + 1}/{len(df_para_classificar)}")            
            
            
        except Exception as e:
            print(f"Erro ao classificar vídeo {row['video_id']}: {e}")
            classificacoes.append("erro")

        time.sleep(3)
    
    df_para_classificar['classificacao_gemini'] = classificacoes
    return df_para_classificar





def carregar_trilhas(caminho_json="datasets/trilhas.json"):    
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados["trilhas"]





def obter_trilha(classificacao_json, trilhas_data):   
    if not classificacao_json:
        return []

    # 1. Garantir que está em dict
    if isinstance(classificacao_json, str):
        # Remove ```json e ``` do início/fim
        cleaned = re.sub(r'^```(?:json)?\s*', '', classificacao_json.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()
        try:
            classificacao = json.loads(cleaned)
        except:
            return []
    else:
        classificacao = classificacao_json

    # 2. Extrair as duas possibilidades
    ferramenta_principal = classificacao.get("ferramenta_principal", "")
    tecnologia_base = classificacao.get("tecnologia_base", "")

    # 3. Procurar trilha por ferramenta principal
    for trilha in trilhas_data:
        if trilha["ferramenta"].upper() == ferramenta_principal.upper():
            return trilha["topicos"]

    # 4. Se não encontrar → tentar tecnologia base
    for trilha in trilhas_data:
        if trilha["ferramenta"].upper() == tecnologia_base.upper():
            return trilha["topicos"]

    # 5. Não achou nada
    return []





def classificar_trilhas_groq(df, groq_api_key, coluna_classificacao='classificacao_gemini'):
        
    genai.configure(api_key=groq_api_key)
    
    # Carregar trilhas
    trilhas_data = carregar_trilhas()
    
    # Selecionar vídeos para classificar
    df_para_classificar = df.copy()
    print(f"\nClassificando {len(df_para_classificar)} vídeos nas trilhas com Groq...")
    
    topicos_classificados = []
    
    for idx, row in df_para_classificar.iterrows():
        print(f"→ Classificando trilha ({idx+1}/{len(df_para_classificar)}) ...")
        
        # Pegar a ferramenta classificada
        classificacao_json = row[coluna_classificacao]
        
        # Buscar a trilha dessa ferramenta
        trilha = obter_trilha(classificacao_json, trilhas_data)
        
        # Se não encontrou trilha, marcar como "sem_trilha"
        if not trilha:
            print(f"  ⚠ Trilha não encontrada para: {classificacao_json}")
            topicos_classificados.append("sem_trilha")
            continue
        
        print(f"  ✓ Trilha encontrada com {len(trilha)} tópicos")
        
        # Montar lista de tópicos para o prompt
        trilha_txt = "\n".join([f"- {t}" for t in trilha])
        
        # Montar prompt
        prompt = f"""Você é um CLASSIFICADOR ESPECIALISTA de vídeos educacionais de tecnologia.

Você receberá APENAS uma SINOPSE TÉCNICA PURIFICADA — um texto curto,
objetivo e 100% limpo de ruído, descrevendo o conteúdo real do vídeo.

==================================================
OBJETIVO
==================================================
Classificar o vídeo no TÓPICO MAIS ADEQUADO da trilha fornecida.


==================================================
REGRAS ABSOLUTAS (SIGA À RISCA)
==================================================
1. Classifique somente com base na sinopse.
2. Não invente tópicos.
3. A sinopse já removeu tudo que é ruído — confie nela.
4. Se dois tópicos forem possíveis, escolha sempre o mais específico.
5. Classificar quando a sinopse descreve exatamente o que o tópico aborda.
6. Classificar quando há palavras-chave técnicas explícitas compatíveis.
7. Quando a sinopse descrever uma ação, prática ou explicação que se encaixa de forma natural em um tópico (mesmo sem match literal), você DEVE classificar.
8. Só retorne "invalido" quando NÃO houver relação técnica plausível com NENHUM dos tópicos.


==================================================
DADOS DO VÍDEO
==================================================

Sinopse Técnica: {row['contexto']}

==================================================
TÓPICOS DISPONÍVEIS PARA "{classificacao_json}":
{trilha_txt}

==================================================
LEMBRETE FINAL:
- Você NÃO PODE criar novos tópicos
- Se não houver correspondência clara, responda "invalido".
- Se a sinopse for genérica demais (ex: motivacional, opinião,
   apresentação, dicas vagas), classifique como "invalido".

RESPONDA APENAS COM:
- O nome EXATO de um tópico da lista acima
- "invalido"

Sem explicações. Sem JSON.
"""
        
        # Chamar Groq
        try:
            model = genai.GenerativeModel('gemma-3-27b-it')
            response = model.generate_content(prompt) 
            topico = response.text
            topicos_classificados.append(topico)
            print(f"  → Tópico: {topico}")
            
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            topicos_classificados.append("erro")
        
        # Rate limit
        time.sleep(3)
    
    # Adicionar coluna ao DataFrame
    df_para_classificar['topico_trilha'] = topicos_classificados
    
    return df_para_classificar



def upload_df_to_gcs_raw(df, bucket_name, filename):
    
    
    creds_json = os.environ['STORAGE_KEY']

    creds_dict = json.loads(creds_json)
    
    credentials = service_account.Credentials.from_service_account_info(creds_dict)

    # Cria o cliente do Storage autenticado com as credenciais carregadas na memória
    client = storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = client.bucket(bucket_name)

    
    csv_data = df.to_csv(index=False, sep=';')

    
    blob_path = f"{filename}"
    blob = bucket.blob(blob_path)

    
    blob.upload_from_string(csv_data, content_type='text/csv')

    print(f"Arquivo '{blob_path}' enviado com sucesso para o bucket '{bucket_name}'.")

# ============================================
# PIPELINE PRINCIPAL
# ============================================

def executar_teste(csv_path, youtube_api_key, gemini_api_key):
    """Executa teste completo"""
    
    print("=" * 70)
    print("INICIANDO TESTE")
    print("=" * 70)
    
    # 1. Carregar 60 canais
    df_canais = carregar_canais(csv_path, limite=1)
    
    # 2. Buscar vídeos de todos os canais
    todos_videos = []
    
    for idx, row in df_canais.iterrows():
        channel_id = row['channel_id']
        channel_name = row['channel_title']
        
        print(f"\n[{idx+1}/60] Processando: {channel_name}")
        
        try:
            # Buscar video IDs
            video_ids = buscar_video_ids_canal(channel_id, youtube_api_key)
            print(f"  → {len(video_ids)} vídeos encontrados")
            
            # Buscar metadados
            if video_ids:
                videos_data = buscar_metadados_videos(video_ids, youtube_api_key)
                todos_videos.extend(videos_data)
                print(f"  → Metadados coletados: {len(videos_data)}")
            
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            continue
    
    # 3. Criar DataFrame
    print(f"\n{'=' * 70}")
    print(f"Total de vídeos coletados: {len(todos_videos)}")
    df_videos = pd.DataFrame(todos_videos)
    
    # 4. Filtrar por data (junho/2024+)
    df_filtrado = filtrar_por_data(df_videos, data_minima='2021-01-01')
    
    # 5. Salvar intermediário
    # df_filtrado.to_csv('videos_coletados_1000.csv', index=False, sep=';')
    # print(f"✅ Vídeos salvos: videos_coletados_terca.csv")
    
    # df_filtrado = pd.read_csv('videos_coletados_1000.csv',sep=';', encoding='utf-8')
    
    # 6. Classificar 100 vídeos
    print(f"\n{'=' * 70}")
    print("CLASSIFICAÇÃO COM GEMINI")
    print("=" * 70)

    df_contextualizado = contextualizar_videos_groq(df_filtrado, gemini_api_key, limite=100)
    
    df_classificado = classificar_videos_groq(df_contextualizado, gemini_api_key, limite=100)

    df_classificado_trilha = classificar_trilhas_groq(df_classificado,gemini_api_key)
    
    # 7. Salvar resultado final
    upload_df_to_gcs_raw(df_classificado_trilha, 'video_bruto', 'classificados.csv')
    print(f"\n✅ Resultado final salvo: videos_classificados_adonis.csv")
    
    # 8. Resumo
    print(f"\n{'=' * 70}")
    print("RESUMO")
    print("=" * 70)
    print(f"Canais processados: 60")
    # print(f"Vídeos totais: {len(df_videos)}")
    print(f"Vídeos desde jun/2024: {len(df_filtrado)}")
    print(f"Vídeos classificados: {len(df_classificado)}")
    
    return df_classificado_trilha # mudar depois para df_classificado

# ============================================
# EXECUTAR
# ============================================

if __name__ == "__main__":
    
    # Configurações
    CSV_PATH = 'datasets/canais_tech_BR.csv'
    YOUTUBE_API_KEY = os.environ['API_KEY']
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
    # GROQ_API_KEY = os.environ['GROQ_API_KEY']
    # Executar teste
    df_resultado = executar_teste(CSV_PATH, YOUTUBE_API_KEY, GEMINI_API_KEY)
    
    # Ver alguns resultados
    print("\n" + "=" * 70)
    print("AMOSTRA DOS RESULTADOS")
    print("=" * 70)

    print(df_resultado[['title', 'channel_name', 'published_at', 'viewCount']].head(10))











