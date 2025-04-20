from openai import OpenAI  
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  
# Does this succeed or 400?
resp = client.embeddings.create(model="text-embedding-3-small", input=["foo","bar"])  
print(resp)
