# import os
# import openai
# from dotenv import load_dotenv
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from langsmith import Client
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# import gradio as gr

# # Load environment variables from .env file
# load_dotenv()

# # Set OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize LangSmith Client
# client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# # Wrap the OpenAI API to enable tracing
# wrapped_openai = wrap_openai(openai)

# # 1. Configuration and Tools
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# @traceable  # Auto-trace this function
# class SEOTool:
#     @tool("SEO Optimizer")
#     def optimize_content(content: str) -> str:
#         """Optimize content for SEO."""
#         try:
#             # Use OpenAI to generate SEO optimized content
#             response = openai.ChatCompletion.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are an SEO optimization assistant."},
#                     {"role": "user", "content": f"Optimize the following content for SEO:\n\n{content}"}
#                 ],
#                 max_tokens=500
#             )
#             optimized_content = response.choices[0].message['content'].strip()
#             return optimized_content

#         except Exception as error:
#             print("Error while optimizing content:", error)
#             return str(error)

# # 2. Creating an Agent for SEO tasks
# seo_agent = Agent(
#     role='SEO Specialist',
#     goal='Optimize content for SEO using the SEO Optimizer Tool',
#     backstory='Expert in optimizing content for better search engine rankings.',
#     tools=[SEOTool.optimize_content],
#     verbose=True,
#     llm=llm
# )

# # 3. Defining a Task for SEO operations
# seo_task = Task(
#     description='This will be replaced by user prompt',
#     expected_output='Optimized content',
#     agent=seo_agent,
#     tools=[SEOTool.optimize_content]
# )

# # 4. Creating a Crew with SEO focus
# seo_crew = Crew(
#     agents=[seo_agent],
#     tasks=[seo_task],
#     process=Process.sequential,
#     manager_llm=llm
# )

# # 5. Define SEO interface function
# def seo_interface(content: str) -> str:
#     seo_task.description = content
#     result = seo_crew.kickoff()
#     return result

# # 6. Define and launch Gradio interface
# iface = gr.Interface(
#     fn=seo_interface,
#     inputs=gr.Textbox(lines=10, placeholder="Enter content to be optimized for SEO"),
#     outputs="text",
#     title="SEO Content Optimizer",
#     description="Optimize your content for SEO via a natural language interface."
# )

# iface.launch()


import os
import logging
from openai import OpenAI
import openai
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import gradio as gr
import agentops  # Make sure to import the agentops library

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangSmith Client
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# Initialize agentops
agentops.init()  # Initialize agentops at the start

# Wrap the OpenAI API to enable tracing
wrapped_openai = wrap_openai(openai)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable
class SEOTool:
    @tool("SEO Optimizer")
    def optimize_content(content: str) -> str:
        """Optimize content for SEO."""
        try:
            # Start a session for tracking
            agentops.start_session()  # Start session
            client.start_session()  # Start LangSmith session

            # Use OpenAI to generate SEO optimized content
            response = wrapped_openai.Completions.create(
                model="gpt-3.5-turbo",
                prompt=f"Optimize the following content for SEO:\n\n{content}",
                max_tokens=500
            )
            optimized_content = response.choices[0].text.strip()
            return optimized_content

        except Exception as error:
            logger.error("Error while optimizing content: %s", error)
            return str(error)

        finally:
            # End sessions
            try:
                client.end_session()  # End LangSmith session
                agentops.end_session()  # End agentops session
            except Exception as session_error:
                logger.error("Error ending sessions: %s", session_error)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# 2. Creating an Agent for SEO tasks
seo_agent = Agent(
    role='SEO Specialist',
    goal='Optimize content for SEO using the SEO Optimizer Tool',
    backstory='Expert in optimizing content for better search engine rankings.',
    tools=[SEOTool.optimize_content],
    verbose=True,
    llm=llm
)

# 3. Defining a Task for SEO operations
seo_task = Task(
    description='This will be replaced by user prompt',
    expected_output='Optimized content',
    agent=seo_agent,
    tools=[SEOTool.optimize_content]
)

# 4. Creating a Crew with SEO focus
seo_crew = Crew(
    agents=[seo_agent],
    tasks=[seo_task],
    process=Process.sequential,
    manager_llm=llm
)

# 5. Define SEO interface function
def seo_interface(content: str) -> str:
    seo_task.description = content
    result = seo_crew.kickoff()
    return result

# 6. Define and launch Gradio interface
iface = gr.Interface(
    fn=seo_interface,
    inputs=gr.Textbox(lines=10, placeholder="Enter content to be optimized for SEO"),
    outputs="text",
    title="SEO Content Optimizer",
    description="Optimize your content for SEO via a natural language interface."
)

iface.launch()
