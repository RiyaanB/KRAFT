import streamlit as st

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

import test

# from src.build_context_simple import build_context_simple
# from src.build_context_iterative import build_context_iterative

# Define a function for the chatbot's response logic
# This is a placeholder function, you'll need to replace it with your actual chatbot logic
def get_chatbot_response(question, edge_traversal, k, search_strategy, response_type):
    # Your chatbot's logic goes here
    # For example, it could change its behavior based on the edge_traversal strategy, k, and search_strategy
    
    # llm = ChatOpenAI(openai_api_key="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", temperature=0.0, model_name='gpt-3.5-turbo')
    # embeddings_model = OpenAIEmbeddings(openai_api_key="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")

    # info_context = search_strategy(llm, embeddings_model, question, edge_traversal, k, 2, 2)

    # if response_type == "yesno":

    #         template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and COMPULSORILY give your best guess for answer as \"(YES)\" or \"(NO)\""
    #         prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
    #         llm_chain_yesno_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

    #         response = llm_chain_yesno_answer_pipeline.run(query=question, info_context=info_context)

    #         return response

    # elif response_type == "short":

    #         template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and make sure to give your final answer in () parentheses"
    #         prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
    #         llm_chain_short_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

    #         response = llm_chain_short_answer_pipeline.run(query=question, info_context=info_context)

    #         return response
    
    return "No response due to unknown response type"

# Streamlit app
def main():
    st.title("Chatbot Configuration")

    # User input for the question
    question = st.text_input("Ask a question:")

    # Select edge traversal strategy
    edge_traversal = st.radio("Select Edge Traversal Strategy", ("Nearest Neighbor", "Classic"))
    edge_traversal = {
        "Nearest Neighbor": "nearest_neighbor",
        "Classic": "classic"
    }[edge_traversal]

    # Select k value
    k = st.slider("Select k value", 1, 5, 1)

    # Select search strategy
    search_strategy = st.radio("Select Search Strategy", ("Simple", "Iterative"))

    # Select response type
    response_type = st.radio("Select Response Type", ("Yes/No", "Short"))
    response_type = {
        "Yes/No": "yesno",
        "Short": "short"
    }[response_type]

    # Button to get response
    if st.button("Get Response"):
        response = get_chatbot_response(question, edge_traversal, k, search_strategy, response_type)
        st.text("Chatbot Response: {}".format(response))

if __name__ == "__main__":
    main()
