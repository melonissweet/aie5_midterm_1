from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from operator import itemgetter
from langgraph.graph import StateGraph, END
from typing import Annotated, List, TypedDict
from langchain.schema import BaseMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import chainlit as cl
from langdetect import detect
from qdrant_client import QdrantClient
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import uuid
from langgraph.graph.message import add_messages
from langsmith import traceable
from langgraph.prebuilt import InjectedState

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
qdrant_url = os.getenv("QDRANT_CLOUD_URL")
qdrant_api_key = os.getenv("QDRANT_CLOUD_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
# Load LangSmith API key from environment
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "LangGraph-Agent")

# Define LLMs
large_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=openai_api_key)
base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=openai_api_key)

# Define Embeddings
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="Snowflake/snowflake-arctic-embed-l", model_kwargs={'trust_remote_code': True}
)

# Define RAG components
rag_prompt_template = """
You are an helpful legal assistant that answers clients questions based on the provided legal, by-law, and guidance context.
You must only use the provided context and avoid adding any extra knowledge.
Use simple and clear English in your response that your clients will be able to easily understand your explanation.
Avoid using difficult legal terms. If legal terms need to be used, provide short and concise explanation.

## Question:
{question}
## Context:
{context}
"""

rag_prompt = PromptTemplate.from_template(rag_prompt_template)

# Connect to Qdrant
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
collection_name = "ontario_ltb_toronto_guidance"

qdrant_vs = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=huggingface_embeddings
)
retriever = qdrant_vs.as_retriever(search_kwargs={"k": 10})


rag_chain = (
    {
        "context": RunnableLambda(lambda x: x.get("translated_input", "")) | retriever,  
        "question": RunnableLambda(lambda x: x.get("translated_input", ""))
    }
    | RunnablePassthrough.assign(context=lambda x: x["context"])
    | {
        "response": rag_prompt | base_llm | StrOutputParser(),
        "context": RunnableLambda(lambda x: x["context"])  # Ensure context is passed correctly
    }
)


@tool
@traceable
def retrieve(state: Annotated[dict, InjectedState]):
    """Use Retrieval Augmented Generation to retrieve information from the documentations 
    related to landlord and tenancy legal, by-law, and guidance in Ontario, Canada.
    """
    if not isinstance(state, dict):
        raise TypeError(f"Expected dict for state, got {type(state)}")
    
    if "translated_input" not in state:
        raise ValueError("Missing 'translated_input' in state.")

    retrieved_docs = rag_chain.invoke({"translated_input": state["translated_input"]})
    return {"context": retrieved_docs}

# Define Agent State with better tracking
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] 
    detected_lang: str
    translated_input: str  # Stores translated input
    verification_passed: bool  # Tracks if translation verification passed

# Define Agent Pipeline
class AgentPipeline:
    def __init__(self):
        self.tavily_tool = TavilySearchResults(max_results=5)
        self.tool_belt = [self.tavily_tool, retrieve]
        self.llm_with_tools = base_llm.bind_tools(self.tool_belt)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("detect_language", self.detect_language)
        graph.add_node("translate_input", self.translate_input)
        graph.add_node("verify_translation", self.verify_translation)
        graph.add_node("translate_text_with_feedback", self.translate_text_with_feedback)
        graph.add_node("agent", self.call_model)
        graph.add_node("action", ToolNode(self.tool_belt))
        graph.add_node("translate_output", self.translate_output)
        
        graph.set_entry_point("detect_language")
        graph.add_edge("detect_language", "translate_input")
        graph.add_edge("translate_input", "verify_translation")
        graph.add_conditional_edges("verify_translation", self.handle_translation_verification, {
            "pass": "agent",
            "fail": "translate_text_with_feedback"
        })
        graph.add_edge("translate_text_with_feedback", "agent")
        graph.add_conditional_edges("agent", self.tool_call_or_helpful, {
            "continue": "agent",
            "action": "action",
            "end": "translate_output"
        })
        graph.add_edge("action", "agent")
        graph.add_edge("translate_output", END)
        
        return graph.compile()
    
    @traceable
    async def detect_language(self, state: AgentState):
        user_input = state["messages"][-1].content
        detected_lang = detect(user_input)
        state["detected_lang"] = detected_lang
        return state
    @traceable
    async def translate_input(self, state: AgentState):
        last_message_content = state["messages"][-1].content
        if state["detected_lang"] != "en":
            translated_input = await self.translate_text(state["detected_lang"], "en", state["messages"][-1].content)
            state["translated_input"] = translated_input
            state["messages"].append(HumanMessage(content=translated_input))
        else:
            state["translated_input"] = last_message_content
        return state
    @traceable
    async def verify_translation(self, state: AgentState):
        original_text = state["messages"][0].content
        translated_text = state["translated_input"]
        verification_prompt = f"""
        Verify if the following translation preserves the original meaning accurately.

        Original: {original_text}
        Translated: {translated_text}

        Respond with 'Y' if correct, 'N' if incorrect.
        """
        prompt_template = PromptTemplate.from_template(verification_prompt)
        verification_chain = prompt_template | base_llm | StrOutputParser()
        verification_response = await verification_chain.ainvoke({})
        state["verification_passed"] = "Y" in verification_response
        return state
    @traceable
    async def translate_text_with_feedback(self, state: AgentState):
        """Retries translation with additional verification feedback."""
        original_text = state["messages"][0].content
        first_translation = state["translated_input"]

        robust_translation_prompt = f"""
        You are an professional translator. You can support translation between various languages 
        but you have most expertise in translating between Korean, Japanese, and English.
        The initial translation was found to be inaccurate!
        The ISO 639 language codes of the language is provided to let you know the source language and target language.
        Please re-translate the following text from {state["detected_lang"]} to en while ensuring accuracy with the presevation of the full meaning.
        Only return the translated text, no explanations or extra details.
        Translate the formality in the {state["detected_lang"]} to common/plain en.

        Original: {original_text}
        First Attempt Translation: {first_translation}

        Translation:
        """
        prompt_template = PromptTemplate.from_template(robust_translation_prompt)
        robust_translation_chain = prompt_template | base_llm | StrOutputParser()
        improved_translation = await robust_translation_chain.ainvoke({})

        state["translated_input"] = improved_translation
        state["messages"][-1] = HumanMessage(content=improved_translation)  # Replace with new translation
        return state

    @traceable
    async def translate_output(self, state: AgentState):
        if state["detected_lang"] != "en":
            translated_response = await self.translate_text("en", state["detected_lang"], state["messages"][-1].content)
            return {"messages": [HumanMessage(content=translated_response)]}
        return {"messages": [HumanMessage(content=state["messages"][-1].content)]}
    
    @traceable
    async def translate_text(self, source_lang, target_lang, text):
        translation_prompt = f"""
        You are an professional translator. You can support translation between various languages 
        but you have most expertise in translating between Korean, Japanese, and English.
        The ISO 639 language codes of the language is provided to let you know the source language and target language.
        Translate the following text from {source_lang} to {target_lang}, ensuring accuracy and preservance of original meaning.  
        Only return the translated text, no explanations or extra details.
        Translate the formality in the {source_lang} to common/plain {target_lang}.

        {text}

        Translation:"""
        
        prompt_template = PromptTemplate.from_template(translation_prompt)
        translation_chain = prompt_template | base_llm | StrOutputParser()
        return await translation_chain.ainvoke({"text": text})
    @traceable
    async def call_model(self, state: AgentState):
        """Calls the LLM to generate a response based on user input."""
        response = await self.llm_with_tools.ainvoke(state["messages"])
        
        # Ensure messages list is updated correctly
        updated_messages = state["messages"] + [response]

        return {"messages": updated_messages}

    @traceable
    async def handle_translation_verification(self, state: AgentState):
        """
        Determines whether the translation verification passes or fails.
        If verification fails, it triggers a more robust translation process before proceeding.
        """
        return "pass" if state["verification_passed"] else "fail"

    @traceable
    async def tool_call_or_helpful(self, state: AgentState):
        if len(state["messages"]) > 10: # checks whether to end the query cycle as the first step of this function
            return "end"
        
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "action"
        
        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        helpfulness_prompt = """
        Given an initial query and a final response, determine if the response is helpful.
        Indicate helpfulness with 'Y' and unhelpfulness with 'N'.
        
        Initial Query:
        {initial_query}
        
        Final Response:
        {final_response}
        """

        prompt_template = PromptTemplate.from_template(helpfulness_prompt)
        helpfulness_chain = prompt_template | base_llm | StrOutputParser()
        helpfulness_response = await helpfulness_chain.ainvoke({
            "initial_query": initial_query.content,
            "final_response": final_response.content
        })

        return "end" if "Y" in helpfulness_response else "continue"

@cl.on_chat_start
async def on_chat_start():
    agent_graph = AgentPipeline()
    cl.user_session.set("agent_graph", agent_graph)
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(content="Agent is ready! Ask your questions.").send()

@cl.on_message
async def main(message):
    agent_graph = cl.user_session.get("agent_graph")
    thread_id = cl.user_session.get("thread_id")
    inputs = {"messages": [HumanMessage(content=message.content)]}
    msg = cl.Message(content="")
    graph_config = {"configurable": {"thread_id": thread_id, "cl_msg": msg}}
    
    async for chunk in agent_graph.graph.astream(input=inputs, config=graph_config, stream_mode="updates"):
        final_response = None  # Store the last response

        for node, values in chunk.items():
            if node == "translate_output":  # Ensure we only capture the final node's response
                final_response = values["messages"][-1].content  # Get last message

        if final_response:  # Send only the final response
            await msg.stream_token(final_response)

        # for node, values in chunk.items():
        #     await msg.stream_token(f"**{node} response:**\n{values['messages'][0].content}\n\n")
        # for node, values in chunk.items():
        #     if node == "action":
        #         node_header = f"**Receiving update from node: '{node}'**\nTool Used: {values['messages'][0].name}\n"
        #     else:
        #         node_header = f"**Receiving update from node: '{node}'**\n"
        #     await msg.stream_token(node_header)
            
        #     for response in values["messages"]:
        #         response_header = f"{response.content}\n\n"
        #         await msg.stream_token(response_header) 
    
    await msg.send()
