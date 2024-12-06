import streamlit as st
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

from llama_index.core import StorageContext, load_index_from_storage

# Initialize storage context and load index
storage_context = StorageContext.from_defaults(persist_dir="./clinicaltrial_01_index")
index = load_index_from_storage(storage_context, index_id="clinicaltrial_01")

# Define the document context and QA template
document_context = "clinical trial and detailed information about locations."
template = (
    "This document contains important information about {document_context}."
    "\n---------------------\n"
    "Based on the provided document, your input query will be answered using the most relevant information."
    "consider community titles as hospital names."
    
    "\n---------------------\n"
    "Document Context: {document_context}\n"
    "User Query: {query_str}\n"
    "Response:\n"
)
qa_template_llamaindex = PromptTemplate(template)

# Configure query engine with the QA template
query_engine = index.as_query_engine(
    text_qa_template=qa_template_llamaindex, similarity_top_k=5
)

# Set up OpenAI model and RAG tool
#llm = OpenAI(model="gpt-4o", temperature=0)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="Clinical trial data",
    description="It has clinical trial data, hospital names, websiten names, Therapeutic Areas	Last Updated	Phases of Clinical Studies	Types of Studies	# of Employees Involved in Research	Days Open	Extended Office Hours	Dry Ice	Blinded Pharmacist	Dedicated Monitoring Space	Internet Access for the Monitor	Onsite Laboratory	Infusion Capability	Populations	Type of IRB used by this site	Appropriate storage facilities for product	Able to appropriately store Schedule II drugs ",
)

# Initialize ReActAgent
agent = ReActAgent.from_tools([rag_tool], llm=llm, max_iterations=5, verbose=False)

# Streamlit App
st.title("Healthcare QA Chatbot")
st.markdown(
    """
    Welcome to the **Healthcare QA Chatbot**! You can ask questions about clinical trials, locations, 
    and other healthcare topics. The chatbot uses LlamaIndex and a custom prompt template for accurate responses.
    """
)

# Display the template in a collapsible section
with st.expander("View QA Template"):
    formatted_template = template.format(document_context=document_context, query_str="<Your Query>")
    st.text(formatted_template)


# Input fields for the user query
st.subheader("Ask a Question")
user_query = st.text_area(
    "Enter your question:",
    placeholder="E.g., 'What is the location of clinical trials for obesity?'",
    height=100,
)

# Optional: Display the document context (static or editable)
#st.markdown("### Document Context")
#st.info(f"{document_context}")

if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Processing your query..."):
            query_str = user_query
            response = agent.chat(query_str)
            #response = query_engine.query(query_str)
            st.success("Answer retrieved!")
            # Display the response cleanly
            st.markdown("### Answer")
            st.text(response)  # Ensure clean text display
    else:
        st.warning("Please enter a valid question.")


# Footer
st.markdown("---")
st.markdown(
    "Powered by **Digitalgrub Automation*"
)
