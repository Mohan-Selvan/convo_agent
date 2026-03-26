from backend.rag import Retriever
print(Retriever().search('What is Jodoa properties?', limit=3))