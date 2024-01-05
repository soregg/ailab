import json, cgi, tempfile, shutil
from dotenv import load_dotenv
from http.server import BaseHTTPRequestHandler, HTTPServer
import os, pinecone
from langchain.document_loaders import PyPDFLoader

from langchain.llms import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.embeddings import NLPCloudEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings

# read api keys
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

#pinecone init
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index = pinecone.Index(PINECONE_INDEX)
# delete_response = index.delete(delete_all=True)

# embeddings
embeddings = HuggingFaceEmbeddings()
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     api_key=HF_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
# )
# load pdf to pinecone
def pdf_loader(tempfile_path):
    loader = PyPDFLoader(tempfile_path)
    pages = loader.load_and_split()
    vectorstore = Pinecone.from_documents(pages, embeddings, index_name=PINECONE_INDEX)

# chatbot
def chat(json_data):
    
    # llm
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # repo_id="microsoft/phi-2",
        # repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.5,
            "max_length": 200,
            "repetition_penalty": 1.03,
        },
    )


    # llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200}, device=0)
    # llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0, "max_length": 200})

    # get result from pinecone
    chain = load_qa_chain(llm, chain_type="stuff")
    vectorstore = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)
    query=json_data["userprompt"]
    docs = vectorstore.similarity_search(json_data["userprompt"])
    result = chain.run(input_documents=docs, question=query)
    return result.strip()

#####################################
# http request handler
# curl -X POST -F "file=@123.pdf" http://localhost:8000
# curl -X POST -H "Content-Type: application/json" -d '{"userprompt": "Please summarize the document in 100 words."}' http://localhost:8000
#####################################

# request handler
class MyRequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, status_code=200, content_type='text/plain'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        self._set_response(200, 'text/plain')
        self.wfile.write("Hello World.\n".encode('utf-8'))

    def do_POST(self):
        content_type, params = cgi.parse_header(self.headers['Content-Type'])

        isPdfOrJson = False
        if content_type == 'multipart/form-data':
            self._handle_multipart_form_data()
            isPdfOrJson = True
        if content_type == 'application/json':
            self._handle_json_data()
            isPdfOrJson = True
        if not isPdfOrJson:
            self.wfile.write("Hello World.")

    # pdf to pinecone
    def _handle_multipart_form_data(self):
        # create a temporary file to save the uploaded file
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )

        if 'file' in form:
            file_item = form['file']
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                shutil.copyfileobj(file_item.file, temp_file)

        temp_file_path = temp_file.name

        # pdf to pinecone
        pdf_loader(temp_file_path)

        # delete temporary file
        os.remove(temp_file_path)

        self._set_response(200, 'text/plain')
        self.wfile.write("pdf data stored to pinecone, temp pdf deleted.\n".encode('utf-8'))

    # chatbot
    def _handle_json_data(self):
        content_length = int(self.headers['Content-Length'])
        json_data = self.rfile.read(content_length)
        try:
            json_data = json.loads(json_data.decode('utf-8'))
            response_data=chat(json_data)
            self._set_response(200)
            self.wfile.write(response_data.encode('utf-8'))
        except json.JSONDecodeError:
            self._set_response(400)
            self.wfile.write("Invalid JSON format".encode('utf-8'))

def run(server_class=HTTPServer, handler_class=MyRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped")

if __name__ == '__main__':
    run()