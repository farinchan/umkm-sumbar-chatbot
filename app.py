from flask import Flask, request, jsonify
from flask_cors import CORS
from llm import qa


app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/chatbot')
def chatbot():
    query = request.args.get('query')
    qa_response = qa.invoke(query)
    return jsonify({'response': qa_response})


if __name__ == '__main__':
    app.run()
