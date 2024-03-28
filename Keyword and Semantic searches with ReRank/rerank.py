from openai import OpenAI
import os
import weaviate
import cohere
from utils import dense_retrieval
from utils import print_result
from utils import keyword_search


with open("/home/savitha07/.env") as env:
    for line in env:
        key, value = line.strip().split('=')
        os.environ[key] = value

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

os.environ["COHERE_API_KEY"] = os.environ.get('COHERE_API_KEY')
os.environ["WEAVIATE_API_KEY"] = os.environ.get('WEAVIATE_API_KEY')
os.environ["WEAVIATE_API_URL"] = os.environ.get('WEAVIATE_API_URL')

co = cohere.Client(os.environ['COHERE_API_KEY'])

auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ["WEAVIATE_API_KEY"])

client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ["COHERE_API_KEY"],
    }
)


query = "What is the capital of Canada?"

dense_retrieval_results = dense_retrieval(query, client)

print_result(dense_retrieval_results)

query_1 = "What is the capital of Canada?"
results = keyword_search(query_1, client,
  properties=["text", "title", "url", "views", 
        "lang", 
        "_additional {distance}"],
  num_results=3
  )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))

results = keyword_search(query_1,
   client,
   properties=["text", "title", "url", "views", 
               "lang", 
               "_additional {distance}"],
   num_results=500
   )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))

def rerank_responses(query, responses, 
         num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses



texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)


for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

query_2 = "Who is the tallest person in history?"

results = dense_retrieval(query_2,client)

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()

texts = [result.get('text') for result 
         in results]
reranked_text = rerank_responses(query_2, 
         texts)


for i, rerank_result in enumerate(
        reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()
