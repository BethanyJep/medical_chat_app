from flask import Flask, request, jsonify, render_template
import os
import openai
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_type = "azure"
openai.api_base = "https://test-onetwothree-openai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ['OPENAI_API_KEY']

data = pd.read_csv("patient_data.csv")
# Define the OpenAI embeddings and FAISS vector store
embeddings=OpenAIEmbeddings(deployment="med_embed",
                            model="text-embedding-ada-002",
                            openai_api_base="https://test-onetwothree-openai.openai.azure.com/",
                            openai_api_type="azure",
                            chunk_size=1)
                            
@app.route("/", methods=["GET"])
def home():
    # Render the index.html template
    return render_template("index.html")

# Define a route for handling incoming chat messages
@app.route('/', methods=['POST'])
def index():
    # 

    # Get the message from the request body
    message = request.form['text']


    # Call the OpenAI API to generate a response
    try: 
        response = openai.ChatCompletion.create(
        engine="newturbs", # engine = "deployment_name".
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                {"role": "user", "content": message}
            ]
        )

        # Extract the response text from the API response
        response = response['choices'][0]['message']['content']
    except:
        response = "Sorry, I'm a little tired right now. Please try again later."

    # Return the response as a JSON object
    return render_template('index.html', response=response, message=message)

if __name__ == '__main__':
    app.run(debug=True)