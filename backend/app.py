import os

from flask import Flask, request
from flask_cors import CORS

from docsearch import EmbeddingPipeline

# Configurations
os.environ["PINECONE_API_KEY"] = [PINECONE_API_KEY]
os.environ["PINECONE_ENVIRON"] = [PINECONE_ENVIRON]
os.environ["PINECONE_INDEX_NAME"] = "rentallawnsw"
os.environ["OPENAI_API_KEY"] = [OPENAI_API_KEY]

pipeline = EmbeddingPipeline()
app = Flask(__name__)
CORS(app)
@app.route('/', methods=['POST'])
def query_rental_act():
    data = request.get_json()
    query = data.get('input')
    response = pipeline.query(query=query)
    return response

if __name__ == '__main__':
    app.run(debug=True)


    