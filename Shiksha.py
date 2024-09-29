import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import os
import google.generativeai as genai
import asyncio
import wikipedia
from googlesearch import search
import fitz  # PyMuPDF
from transformers import BartForConditionalGeneration, BartTokenizer
from gesturecv import run_gesture_math_solver
from app import calculate_perplexity, calculate_burstiness, plot_most_common_words, plot_repeated_words
from test import calculate_perplexity, calculate_burstiness as calculate_burstiness_gpt, plot_top_repeated_words



# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt_pdf = """ Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in provided context just say, 
    "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

prompt_youtube = """You are Youtube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:  """

# Load the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def generate_pdf_summary(uploaded_file):
    # Extract text from uploaded PDF
    text = extract_text_from_pdf(uploaded_file)

    if text.strip() != "":
        # Tokenize input text
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    else:
        return "The PDF file is empty or the text could not be extracted."


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_pdf_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


async def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_pdf, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = asyncio.run(get_conversational_chain())
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID from different possible YouTube URL formats
        video_id = None
        if "youtu.be/" in youtube_video_url:
            video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/watch" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]

        if not video_id:
            raise ValueError("Invalid YouTube URL")

        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])

        return transcript

    except Exception as e:
        raise e


def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text


def search_wikipedia(topic):
    try:
        summary = wikipedia.summary(topic, sentences=3)
        page_url = wikipedia.page(topic).url
        return summary, page_url
    except wikipedia.exceptions.DisambiguationError as e:
        # If the topic is ambiguous, return suggestions
        return f"Multiple results found. Did you mean: {', '.join(e.options)}", None
    except wikipedia.exceptions.PageError:
        # If the topic does not exist on Wikipedia, return an error message
        return "Sorry, no results found on Wikipedia.", None


def search_google(topic, num_links=5):
    try:
        links = search(topic, num_results=num_links)
        return links
    except Exception as e:
        return f"Error: {e}"

st.set_page_config(page_title="Shiksha Sangam", layout="wide")
def main():
    # st.set_page_config("Shiksha Sangam")
    st.title("Shiksha Sangam")
    st.write(
        "Welcome to Your Academic Helper! This powerful tool seamlessly integrates advanced chat capabilities "
        "for PDF documents and a YouTube video Summarizer, along with Gesture-Based Math Solver , Plagarism detector and finally Find Resources section,"
        " providing you with comprehensive study support. Engage in enlightening discussions with your PDFs "
        ", while the Summarizer distills extensive videos into essential points for "
        "effortless understanding and finally Find Resources section, where you can effortlessly discover "
        "relevant links and materials related to your provided topic. Tailored for students and researchers alike, "
        "our platform serves as an "
        "indispensable resource, facilitating efficient studying and improved comprehension.")
    option = st.radio("Choose an option:", ("Multiple PDF Chat", "YouTube Summarizer","Gesture Math Solver", "PDF Summarization","AI Plagiarism Detector", "Find Resources"))

    if option == "Multiple PDF Chat":
        st.header("Chat with Multiple PDF's")
        user_question = st.text_input("Enter Any Topic From the PDF")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                        accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_pdf_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    elif option == "YouTube Summarizer":
        st        .header("Get Detailed Notes and Summaries From YouTube Videos")
        youtube_link = st.text_input("Enter YouTube Video Link:")

        if st.button("Get Detailed Notes"):
            try:
                transcript_text = extract_transcript_details(youtube_link)
                if transcript_text:
                    summary = generate_gemini_content(transcript_text, prompt_youtube)
                    st.markdown("## Detailed Notes:")
                    st.write(summary)
            except Exception as e:
                st.error(f"Error retrieving transcript or generating summary: {e}")

    elif option == "Find Resources":
        st.header("Find Resources")
        topic = st.text_input("Enter a Topic:")
        if st.button("Search"):
            wikipedia_summary, wikipedia_url = search_wikipedia(topic)
            if wikipedia_url:
                st.markdown(f"### Wikipedia Summary:")
                st.write(wikipedia_summary)
                st.markdown(f"[Link to Wikipedia Article]({wikipedia_url})")
            else:
                st.write(wikipedia_summary)

            google_links = search_google(topic)
            st.markdown("### Google Search Results:")
            for link in google_links:
                st.markdown(f"[{link}]({link})")

    elif option == "PDF Summarization":
        st.header("PDF Summarization")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Generate Summary"):
                summary = generate_pdf_summary(uploaded_file)
                st.write("Summary:", summary)

    elif option == "Gesture Math Solver":
        # Call the gesture math solver function from gesture_math.py
        run_gesture_math_solver()
    elif option == "AI Plagiarism Detector":
    # Create a text area for user input
        text_area = st.text_area("Enter text to analyze:", "")

        if text_area:
            if st.button("Analyze"):
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.info("Your Input Text")
                    st.success(text_area)

                with col2:
                    st.info("Detection Score")
                    perplexity = calculate_perplexity(text_area)  # Update this if you have a specific model
                    burstiness_score = calculate_burstiness(text_area)

                    st.write("Perplexity:", perplexity)
                    st.write("Burstiness Score:", burstiness_score)

                    if perplexity > 30000 and burstiness_score < 0.2:
                        st.error("Text Analysis Result: AI-generated content")
                    else:
                        st.success("Text Analysis Result: Likely not generated by AI")

                    st.warning(
                        "Disclaimer: AI plagiarism detector apps can assist in identifying potential instances of plagiarism; however, it is important to note that their results may not be entirely flawless or completely reliable. These tools employ advanced algorithms, but they can still produce false positives or false negatives. Therefore, it is recommended to use AI plagiarism detectors as a supplementary tool alongside human judgment and manual verification for accurate and comprehensive plagiarism detection."
                    )

                with col3:
                    st.info("Basic Details")
                    plot_top_repeated_words(text_area)  # Implement the plotting function



if __name__ == "__main__":
    main()

