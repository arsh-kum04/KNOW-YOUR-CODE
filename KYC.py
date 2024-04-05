import streamlit as st
import requests
from PyPDF2 import PdfReader
from fpdf import FPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
token=os.getenv("TOKEN_BEARER")
# Global headers for GitHub API requests
headers = {
    'Authorization': f'Bearer {os.getenv("TOKEN_BEARER")}',
}
def commit_code_to_github(owner, repository, filepath, filename, content, token):
    # GitHub repository information
    owner = owner
    repository = repository
    branch = "main"

    # Headers with authorization
    headers = {
        "Authorization": "token {}".format(token),
        "Accept": "application/vnd.github.v3+json"
    }

    # Get the last commit SHA of a specific branch
    branch_url = "https://api.github.com/repos/{}/{}/branches/{}".format(owner, repository, branch)
    response = requests.get(branch_url, headers=headers)
    response.raise_for_status()
    last_commit_sha = response.json()['commit']['sha']

    # Create blobs with the file's content
    blob_data = {
        "content": content,
        "encoding": "utf-8"  # Specify the encoding type as utf-8
    }
    blob_url = "https://api.github.com/repos/{}/{}/git/blobs".format(owner, repository)
    response = requests.post(blob_url, json=blob_data, headers=headers)
    response.raise_for_status()
    blob_sha = response.json()['sha']

    # Create a tree which defines the folder structure
    tree_data = {
        "base_tree": last_commit_sha,
        "tree": [
            {
                "path": os.path.join(filepath),
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha
            }
        ]
    }
    tree_url = "https://api.github.com/repos/{}/{}/git/trees".format(owner, repository)
    response = requests.post(tree_url, json=tree_data, headers=headers)
    response.raise_for_status()
    tree_sha = response.json()['sha']

    # Create the commit
    commit_data = {
        "message": "Add documentation for {}".format(filename),
        "author": {
            "name": "Arsh Kumar",
            "email": "arshkum18@gmail.com"
        },
        "parents": [last_commit_sha],
        "tree": tree_sha
    }
    commit_url = "https://api.github.com/repos/{}/{}/git/commits".format(owner, repository)
    response = requests.post(commit_url, json=commit_data, headers=headers)
    response.raise_for_status()
    new_commit_sha = response.json()['sha']

    # Update the reference of your branch to point to the new commit SHA
    branch_url = "https://api.github.com/repos/{}/{}/git/refs/heads/{}".format(owner, repository, branch)
    branch_data = {
        "sha": new_commit_sha
    }
    response = requests.patch(branch_url, json=branch_data, headers=headers)
    response.raise_for_status()

    print("Code committed and pushed successfully.")

def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def generate_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def generate_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings,)
    vector_store.save_local("faiss_index")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def generate_documentation(codebase):

    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.0-pro",generation_config=generation_config,safety_settings=safety_settings)

    # Start a chat session
    convo = model.start_chat(history=[])

    # Set up the prompt
    prompt = f"""
    As a developer, you understand the importance of providing clear and comprehensive documentation for codebases. Your task is to use the Gemini API to analyze a codebase and generate detailed documentation for each file, including the folder name, file name, original code snippet, and a detailed explanation of the functionality line by line.

    The codebase provided consists of code files in various programming languages (e.g., .js, .jsx, .py) with comments or descriptive variable/function names to aid comprehension.

    Please analyze the codebase and generate documentation for each file. Ensure that the documentation includes as comments:

    1. Folder Name: Name of the folder containing the code files.
    2. File Name: Name of the specific code file being analyzed.
    3. Line by line documented Code: The original code snippet with line by line commented documentation in very well explained manner and precisely line by line document the functions in comments.

    Codebase:
    {codebase}
    """
    # Send the codebase text to Gemini API for documentation
    convo.send_message(prompt)

    # Get the response which contains the generated documentation
    response = convo.last.text

    return response


def generate_conversation_chain():
    prompt_template = """
    As a seasoned developer and mentor, you understand the importance of providing clear, understandable code examples along with comprehensive documentation for newcomers to grasp complex codebases effectively. Your task is to create a prompt that addresses the issue of clarity and understanding for newcomers seeking to comprehend code through examples, code snippets (present in the PDF), and well-commented explanations integrated into the code itself.

    Context:
    {context} (Provide the PDF containing the code for analysis)

    Question:
    {question}(line by line with code snippet)

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input_prompt(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = generate_conversation_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def fetch_user_repo(username):
    try:
        response = requests.get(f"https://api.github.com/users/{username}/repos", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code} occurred while fetching repositories.")
            return []
    except Exception as e:
        print('An error occurred while fetching repositories:', e)
        return []

def fetch_code_contents(url):
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code} occurred while fetching contents.")
            return []
    except Exception as e:
        print('An error occurred while fetching contents:', e)
        return []

def get_code_structure(contents, username, repo_name, current_path=''):
    result = ''
    commented_code=''
    documented_code=''
    for item in contents:
        if item['type'] == 'dir':
            result += f"Folder: {item['name']}\n"
            subdir_contents = fetch_code_contents(item['url'])
            new_path = os.path.join(current_path, item['name'])
            result += get_code_structure(subdir_contents, username, repo_name, new_path)
        else:
            filename = item['name']
            if filename.endswith(('.py', '.jsx', '.js', '.html', '.css','.kt','.cpp')):
                raw_url = item['download_url']
                code = fetch_user_code(raw_url)
                documented_code=generate_documentation(code)
                commented_code+=documented_code
                st.write(f"Documented Code: {documented_code}")
                result += f"File: {filename}\n"
                result += f"Documented code for {filename}:\n{code}\n\n"
                # Store the path for the file
                file_path = os.path.join(current_path, filename)
                commit_code_to_github(username,repo_name,file_path,filename,documented_code,token)
                print(f"File Path: {file_path}")
    generate_pdf_doc(username,commented_code)
    return result

def fetch_user_code(raw_url):
    try:
        response = requests.get(raw_url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error {response.status_code} occurred while fetching code.")
            return None
    except Exception as e:
        print('An error occurred while fetching code:', e)
        return None

def generate_pdf(username, repo_name, code):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in code.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(f"{username}_{repo_name}_code.pdf")

def generate_pdf_doc(username,code):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in code.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(f"{username}_final_documentedCode.pdf")
def generate_pdf_documentation(username, selected_repo):
    try:
        repositories = fetch_user_repo(username)
        if len(repositories) == 0:
            raise Exception('No repositories found for the given username.')

        repo = next((repo for repo in repositories if repo['html_url'] == selected_repo), None)
        if not repo:
            raise Exception('Selected repository not found.')

        contents = fetch_code_contents(repo['url'] + '/contents')
        code = get_code_structure(contents, username, repo['name'])
        generate_pdf(username, repo['name'], code)

        print(f"PDF '{username}_{repo['name']}_code_documentation.pdf' generated successfully.")
    except Exception as e:
        print('An error occurred while generating PDF documentation:', e)

def main():
    st.set_page_config("Know Your Code")
    st.header("KNOW YOUR CODE")

    page = st.sidebar.selectbox("Select Page", ["Code Documentation Generator", "Gemini Chat"])

    if page == "Code Documentation Generator":
        st.subheader("Code Documentation Generator")

        username = st.text_input("Enter GitHub Username:", "")
        if username:
            repositories = fetch_user_repo(username)
            if repositories:
                selected_repo_name = st.selectbox("Select Repository:", [repo["name"] for repo in repositories])
                selected_repo = next((repo for repo in repositories if repo["name"] == selected_repo_name), None)
                if selected_repo:
                    st.write(f"You selected: {selected_repo_name}")
                    if st.button("Fetch Repository Data and Generate PDF"):
                        repoLink = f"https://github.com/{username}/{selected_repo_name}"
                        generate_pdf_documentation(username, repoLink)
                        st.success("PDF generated successfully.")
    elif page=="Gemini Chat":
        st.title("KNOW YOUR CODE: Chat with Code")
        
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input_prompt(user_question)
    
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_text_from_pdf(pdf_docs)
                    text_chunks = generate_text_chunks(raw_text)
                    generate_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
