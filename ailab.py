import socket, json, requests, cgi, tempfile, shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
import os, nlpcloud, pinecone
from langchain import vectorstores
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, TextLoader
from langchain.embeddings import NLPCloudEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

# read api keys
NLPCLOUD_API_KEY = os.environ.get("NLPCLOUD_API_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

def pdf_loader(tempfile_path):
    print ("pdf_loader")
    print (tempfile_path)
    
    # Other options for loaders 
    loader = PyPDFLoader(tempfile_path)
    # loader = UnstructuredPDFLoader("123.pdf")
    # loader = OnlinePDFLoader("https://www.redhat.com/rhdc/managed-files/rh-hong-kong-jockey-club-case-study-f29603pr-202109-en.pdf")

    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print (f'You have {len(data)} document(s) in your data')
    print (f'There are {len(data[0].page_content)} characters in your sample document')
    print (f'Here is a sample: {data[0].page_content[:200]}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    embeddings = NLPCloudEmbeddings()

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX)

def chat(prompt):
    embeddings = NLPCloudEmbeddings()
    query_vector = embeddings.embed_query(prompt['userprompt'])
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    result = pinecone.QueryVector(query_vector,top_k=5)
    print("=============")
    print("=============")
    print("=============")
    print("=============")
    print (result)
    print("=============")
    print("=============")
    print("=============")
    print("=============")
    return result["values"]

#####################################
# http request handler
# curl -X POST -F "file=@123.pdf" http://localhost:8000
# curl -X POST -H "Content-Type: application/json" -d '{"userprompt": "Please summarize the document in 100 words."}' http://localhost:8000
#####################################

class MyRequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, status_code=200, content_type='text/plain'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

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
            self._set_response(400)
            self.wfile.write("Unsupported content type")

    def _handle_multipart_form_data(self):
        # Create a temporary file to save the uploaded file

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

        # Perform action 1 (replace this with your specific logic)
        print(f"Action 1: Processing PDF file at {temp_file_path}")
        pdf_loader(temp_file_path)

        # Delete the temporary file
        os.remove(temp_file_path)

        self._set_response(200, 'text/plain')
        self.wfile.write("pdf data stored to pinecone, temp pdf deleted.\n".encode('utf-8'))

    def _handle_json_data(self):
        content_length = int(self.headers['Content-Length'])
        json_data = self.rfile.read(content_length)
        try:
            json_data = json.loads(json_data.decode('utf-8'))
            if 'userprompt' in json_data:
                response_data=chat(json_data)
                response_json = json.dumps(response_data)
                self._set_response(200, 'application/json')
                self.wfile.write(response_json.encode('utf-8'))
            else:
                self._set_response(400)
                self.wfile.write("Invalid JSON format. 'userprompt' field not found.".encode('utf-8'))
        except json.JSONDecodeError:
            self._set_response(400)
            self.wfile.write("Invalid JSON format".encode('utf-8'))

def run(server_class=HTTPServer, handler_class=MyRequestHandler, port=8000):
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