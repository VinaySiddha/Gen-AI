import os
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

# Load environment variables from .env file
load_dotenv()
cli = OpenAI()
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangSmith Client (assuming it's used for something else in your project)
client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# Wrap the OpenAI API to enable tracing
wrapped_openai = wrap_openai(openai)

# 1. Configuration and Tools
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

@traceable  # Auto-trace this function
class SEOTool:
    @tool("SEO Optimizer")
    def optimize_content(content: str) -> str:
        """Optimize content for SEO."""
        try:
            # Use OpenAI to generate SEO optimized content
            response = cli.completions.create(
                model="gpt-3.5-turbo",
                # messages=[
                #     {"role": "system", "content": "You are an SEO optimization assistant."},
                #     {"role": "user", "content": f"Optimize the following content for SEO:\n\n{content}"}
                # ],
                # max_tokens=500
            )
            optimized_content = response.choices[0].message['content'].strip()
            return optimized_content

        except Exception as error:
            print("Error while optimizing content:", error)
            return str(error)

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



# import openai
# from dotenv import load_dotenv
# import os
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from langsmith import Client
# from langsmith import traceable

# # Load environment variables from .env file
# load_dotenv()

# # Set environment variables
# openai.api_key = os.getenv("OPENAI_API_KEY")
# langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# # Initialize LangSmith Client
# client = Client(api_key=langsmith_api_key)

# # 1. Configuration and Tools
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# @traceable
# class SEOTool:
#     @tool("SEO Optimizer")
#     def optimize_content(content: str):
#         """Optimize content for SEO."""
#         try:
#             # Use OpenAI to generate SEO optimized content
#             response = openai.Completion.create(
#                 model="gpt-3.5-turbo",
#                 prompt=f"Optimize the following content for SEO:\n\n{content}",
#                 max_tokens=100
#             )
#             optimized_content = response.choices[0].text.strip()
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
#     expected_output='Optimize content for SEO using the SEO Optimizer Tool',
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
# def seo_interface(content):
#     seo_task.description = content
#     result = seo_crew.kickoff()
#     return result


# # 6. Define and launch Gradio interface
# import gradio as gr

# iface = gr.Interface(
#     fn=seo_interface,
#     inputs=gr.Textbox(lines=10, placeholder="Enter content to be optimized for SEO"),
#     outputs="text",
#     title="SEO Content Optimizer",
#     description="Optimize your content for SEO via a natural language interface."
# )

# iface.launch()


# import openai
# from dotenv import load_dotenv
# import os
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI
# from langsmith import Client
# import logging
# import streamlit as st
# from langsmith import traceable

# # Load environment variables from .env file
# load_dotenv()

# # Set environment variables
# openai.api_key = os.getenv("OPENAI_API_KEY")
# langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# # Initialize LangSmith Client
# client = Client(api_key=langsmith_api_key)

# # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # 1. Configuration and Tools
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# @traceable
# class SEOTool:
#     @tool("SEO Optimizer")
#     def optimize_content(content: str):
#         """Optimize content for SEO."""
#         # Use OpenAI to generate SEO optimized content
#         response = openai.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an SEO optimization assistant."},
#                 {"role": "user", "content": f"Optimize the following content for SEO:\n\n{content}"}
#             ],
#             max_tokens=500
#         )
#         optimized_content = response.choices[0].message['content'].strip()
#         return optimized_content

#         # except Exception as error:
#         #     logger.error("Error while optimizing content: %s", error)
#         #     return str(error)

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
#     expected_output='Optimize content for SEO using the SEO Optimizer Tool',
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

# # 5. Define SEO interface function with logging
# def seo_interface(content):
#     seo_task.description = content
    
#     # logger.info("Starting SEO optimization task for content: %s", content)
    
#     # try:
#     result = seo_crew.kickoff()
#         # logger.info("SEO optimization result: %s", result)
#     # except Exception as e:
#         # logger.error("Error during SEO optimization: %s", e)
    
#     return result

# # 6. Define and launch Streamlit interface
# st.title("SEO Content Optimizer")
# st.write("Optimize your content for SEO via a natural language interface.")

# content = st.text_area("Enter content to be optimized for SEO", height=200)

# if st.button("Optimize"):
#     with st.spinner("Optimizing..."):
#         result = seo_interface(content)
#     st.write(result)
