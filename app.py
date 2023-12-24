import streamlit as st

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models.openai import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings

import re

import test

from src.build_context_simple import build_context_simple
from src.build_context_iterative import build_context_iterative

import networkx as nx
import matplotlib.pyplot as plt

# def draw_graph(graph):
#     pos = nx.spring_layout(graph)  # You can use different layouts like circular_layout, random_layout, etc.

#     labels = nx.get_node_attributes(graph, 'label')

#     # Draw the graph
#     nx.draw(graph, pos, labels=labels, with_labels=True, node_color='skyblue', edge_color='black', node_size=1500, font_size=10)

#     # Optionally, draw edge labels
#     edge_labels = nx.get_edge_attributes(graph, 'label')
#     nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

#     # Save the graph to an image file
#     plt.savefig("graph_image.png", format="PNG")

def draw_graph(graph):
    # Use a layout that provides a more structured appearance
    pos = nx.kamada_kawai_layout(graph)

    labels = nx.get_node_attributes(graph, 'label')

    # Draw the graph with adjusted node size and font size
    nx.draw(graph, pos, labels=labels, with_labels=True, node_color='skyblue', edge_color='black', node_size=1000, font_size=8)

    # Optionally, draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Save the graph to an image file
    plt.savefig("graph_image.png", format="PNG")

# Define a function for the chatbot's response logic
# This is a placeholder function, you'll need to replace it with your actual chatbot logic
def get_chatbot_response(question, edge_traversal, k, search_strategy, response_type):
    # Your chatbot's logic goes here
    # For example, it could change its behavior based on the edge_traversal strategy, k, and search_strategy
    
    llm = ChatOpenAI(openai_api_key="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", temperature=0.0, model_name='gpt-3.5-turbo')
    embeddings_model = OpenAIEmbeddings(openai_api_key="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")

    if search_strategy == build_context_simple:
        info_context = search_strategy(llm, embeddings_model, question, edge_traversal, k)
    elif search_strategy == build_context_iterative:
        info_context, graph = search_strategy(llm, embeddings_model, question, edge_traversal, k, 2, 2)
        print(graph)
        draw_graph(graph)
    else:
        if response_type == "yesno":

            template = "Answer the question: \"{query}\". Explain reasoning and COMPULSORILY give your best guess for answer as \"(YES)\" or \"(NO)\""
            prompt = PromptTemplate(template=template, input_variables=["query"])
            llm_chain_yesno_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_yesno_answer_pipeline.run(query=question)
            print(response)
            if "(NO)" in response or "(No)" in response:
                return "", False, response
            if "(YES)" in response or "(Yes)" in response:
                return "", True, response
            return "", "Unsure", response
        
        elif response_type == "short":
            template = "Answer the question: \"{query}\". Explain reasoning and make sure to give your final answer in () parentheses"
            prompt = PromptTemplate(template=template, input_variables=["query"])
            llm_chain_short_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_short_answer_pipeline.run(query=question)
            print(response)
            potential_answers = re.findall(r'\((.*?)\)', response)
            if len(potential_answers) == 0:
                return "", "Unsure", response
            else:
                # return last thing in parentheses
                return "", potential_answers[-1], response

    if response_type == "yesno":

            template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and COMPULSORILY give your best guess for answer as \"(YES)\" or \"(NO)\""
            prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
            llm_chain_yesno_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_yesno_answer_pipeline.run(query=question, info_context=info_context)

            return response

    elif response_type == "short":

            template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and make sure to give your final answer in () parentheses"
            prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
            llm_chain_short_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_short_answer_pipeline.run(query=question, info_context=info_context)

            return response
    
    return "No response due to unknown response type"

# Streamlit app
def main():
    st.title("Knowledge graph Retrieval-Augmented Framework for Text generation (KRAFT)")

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
    search_strategy = st.radio("Select Search Strategy", ("Simple", "Iterative", "None (baseline)"))
    search_strategy = {
        "None": None,
        "Simple": build_context_simple,
        "Iterative": build_context_iterative
    }[search_strategy]

    # Select response type
    response_type = st.radio("Select Response Type", ("Yes/No", "Short"))
    response_type = {
        "Yes/No": "yesno",
        "Short": "short"
    }[response_type]

    # Button to get response
    if st.button("Get Response"):
        response = get_chatbot_response(question, edge_traversal, k, search_strategy, response_type)
        st.text_area("Chatbot Response:", response, height=100)

        if search_strategy == build_context_iterative:
            st.image("graph_image.png")

if __name__ == "__main__":
    main()
