import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

# Setup LangChain retrieval chain
def setup_chain():
    file_path = "mental_health_knowledge.csv"
    template = """You are a supportive personal mental health trainer specialized for teenagers to people in their thirties. Focus on issues like lack of someone to listen, friendship or relationship problems, trust issues, and loneliness. Use the following context to answer the question helpfully and empathetically. 
    If you don't know the answer, say so and suggest consulting a professional. Always include: This is not a substitute for professional therapy. In crisis? Call 1-800-273-8255 (USA) or text HOME to 741741.

    Context: {context}

    Question: {question}

    Helpful Answer:"""

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    faiss_index = FAISS.from_documents(documents, embeddings)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key),
        chain_type="stuff",
        retriever=faiss_index.as_retriever(),
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

chain = setup_chain()

# Function to get a positive greeting based on time
def get_positive_quote():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning! 'The best way to predict the future is to create it.' How can I support you today?"
    elif 12 <= current_hour < 18:
        return "Good afternoon! 'You are stronger than you think.' How was your day so far?"
    else:
        return "Good evening! 'Every day is a new beginning.' How can I help you unwind?"

# Chat function
def chat_response(user_message, chat_history, user_type):
    if not user_message.strip():
        return chat_history + [{"role": "user", "content": "Please enter a message."}]
    
    personalized_message = f"User is a {user_type}: {user_message}"
    try:
        response = chain({"query": personalized_message})
        answer = response["result"]
    except Exception as e:
        answer = f"Error: {str(e)}. Please try again."
    
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})
    return chat_history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Personalized Mental Health Trainer")
    gr.Markdown(get_positive_quote())
    
    user_type = gr.Radio(choices=["Student", "Employed"], label="Select your status", value="Student")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
    clear = gr.Button("Clear Chat")

    msg.submit(chat_response, [msg, chatbot, user_type], chatbot)
    clear.click(lambda: [], None, chatbot)

    gr.Markdown("""
    **Crisis Resources**:
    - National Suicide Prevention Lifeline: 1-800-273-8255 (USA)
    - Crisis Text Line: Text HOME to 741741
    - [Crisis Text Line Website](https://www.crisistextline.org)
    """)

# Launch Gradio on a free port
demo.launch(server_name="0.0.0.0", server_port=7861)
