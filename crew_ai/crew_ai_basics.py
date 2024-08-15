# # # ### Inbuilt tool

# import openai
# from dotenv import load_dotenv
# import os
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI

# # Load environment variables from .env file
# load_dotenv()

# # Set environment variables
# # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "SEO Optimization Agents"

# # Set OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # 1. Configuration and Tools
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# class SEOTool:
#     @tool("SEO Optimizer")
#     def optimize_content(content: str):
#         """Optimize content for SEO."""
#         try:
#             # Use OpenAI to generate SEO optimized content
#             response = openai.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 prompt=f"Optimize the following content for SEO:\n\n{content}",
#                 max_tokens=500
#             )
#             optimized_content = response.choices[0].text.strip()
#             return optimized_content

#         except Exception as error:
#             print("Error while optimizing content:", error)
#             return error

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



import openai
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# Set environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# Initialize LangSmith Client
client = Client(api_key=langsmith_api_key)

# 1. Configuration and Tools
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

class SEOTool:
    @tool("SEO Optimizer")
    def optimize_content(content: str):
        """Optimize content for SEO."""
        try:
            # Use OpenAI to generate SEO optimized content
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=f"Optimize the following content for SEO:\n\n{content}",
                max_tokens=500
            )
            optimized_content = response.choices[0].text.strip()
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
    expected_output='Optimize content for SEO using the SEO Optimizer Tool',
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
def seo_interface(content):
    seo_task.description = content
    
    # Start trace
    # trace_id = client.start_trace("SEO Optimization Task")
    
    # try:
    result = seo_crew.kickoff()
        
    #     # Log the task and result
    #     client.log_event(trace_id, {"task_description": seo_task.description, "result": result})
    # except Exception as e:
    #     client.log_event(trace_id, {"error": str(e)}, error=True)
    # finally:
    #     client.end_trace(trace_id)
    
    return result

# 6. Define and launch Gradio interface
import gradio as gr

iface = gr.Interface(
    fn=seo_interface,
    inputs=gr.Textbox(lines=10, placeholder="Enter content to be optimized for SEO"),
    outputs="text",
    title="SEO Content Optimizer",
    description="Optimize your content for SEO via a natural language interface."
)

iface.launch()







################################ ----precode------######################################


# import os
# from crewai import Agent, Task, Crew, Process
# from crewai_tools import SerperDevTool
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# from crewai_tools import SerperDevTool
# os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
# search_tool = SerperDevTool()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# # You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.

# # os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
# # os.environ["OPENAI_MODEL_NAME"] ='openhermes'  # Adjust based on available model
# # os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

# search_tool = SerperDevTool()

# # Define your agents with roles and goals
# researcher = Agent(
#   role='Senior Research Analyst',
#   goal='Uncover cutting-edge developments in AI and data science',
#   backstory="""You work at a leading tech think tank.
#   Your expertise lies in identifying emerging trends.
#   You have a knack for dissecting complex data and presenting actionable insights.""",
#   verbose=True,
#   allow_delegation=False,
#   tools=[search_tool]
#   # You can pass an optional llm attribute specifying what mode you wanna use.
#   # It can be a local model through Ollama / LM Studio or a remote
#   # model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#   #
#   # import os
#   # os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#   #
#   # OR
#   #
#   # from langchain_openai import ChatOpenAI
#   # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7)
# )
# writer = Agent(
#   role='Tech Content Strategist',
#   goal='Craft compelling content on tech advancements',
#   backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
#   You transform complex concepts into compelling narratives.""",
#   verbose=True,
#   allow_delegation=True
# )

# # Create tasks for your agents
# task1 = Task(
#   description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
#   Identify key trends, breakthrough technologies, and potential industry impacts.""",
#   expected_output="Full analysis report in bullet points",
#   agent=researcher
# )

# task2 = Task(
#   description="""Using the insights provided, develop an engaging blog
#   post that highlights the most significant AI advancements.
#   Your post should be informative yet accessible, catering to a tech-savvy audience.
#   Make it sound cool, avoid complex words so it doesn't sound like AI.""",
#   expected_output="Full blog post of at least 4 paragraphs",
#   agent=writer
# )

# # Instantiate your crew with a sequential process
# crew = Crew(
#   agents=[researcher, writer],
#   tasks=[task1, task2],
#   verbose=2, # You can set it to 1 or 2 to different logging levels
# )

# # Get your crew to work!
# result = crew.kickoff()

# print("######################")
# print(result)


# ##################################################### Custom Tool



# import psycopg2
# import openai
# from dotenv import load_dotenv
# import os
# from crewai import Agent, Task, Crew, Process
# from langchain.tools import tool
# from langchain_openai import ChatOpenAI

# # Load environment variables from .env file
# load_dotenv()

# # Set environment variables
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Crew AI Agents"


# # Set OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Get database connection details from environment variables
# db_host = os.getenv("DB_HOST")
# db_name = os.getenv("DB_NAME")
# db_user = os.getenv("DB_USER")
# db_password = os.getenv("DB_PASSWORD")

# # 1. Configuration and Tools
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai.api_key)

# class PostgreSQLTool:
#     @tool("PostgreSQL Executor")
#     def execute_sql_command(command: str):
#         """Execute SQL commands on a PostgreSQL database."""
#         try:
#             # Establish a connection
#             print("***************************************",command)
#             conn = psycopg2.connect(
#                 host=db_host,
#                 dbname=db_name,
#                 user=db_user,
#                 password=db_password
#             )

#             # Create a cursor
#             cursor = conn.cursor()

#             # Execute the SQL command
#             cursor.execute(command)

#             # Commit the changes (if it's a DML statement)
#             conn.commit()

#             # Fetch and return the results (if it's a SELECT statement)
#             if command.strip().upper().startswith("SELECT"):
#                 results = cursor.fetchall()
#                 return results

#             # Close the cursor and connection
#             cursor.close()
#             conn.close()

#         except (Exception, psycopg2.Error) as error:
#             print("Error while executing SQL command:", error)
#             return error

# # 2. Creating an Agent for SQL tasks
# sql_agent = Agent(
#     role='Database Engineer',
#     goal='Execute SQL commands on a PostgreSQL database using the PostgreSQL Executor Tool',
#     backstory='Expert in writing and executing SQL commands on PostgreSQL databases.',
#     tools=[PostgreSQLTool.execute_sql_command],
#     verbose=True,
#     llm=llm
# )

# # 3. Defining a Task for SQL operations
# sql_task = Task(
#     description='This will be replaced by user prompt',
#     expected_output='Execute SQL commands on the PostgreSQL database using the PostgreSQL Executor Tool',
#     agent=sql_agent,
#     tools=[PostgreSQLTool.execute_sql_command]
# )

# # 4. Creating a Crew with SQL focus
# sql_crew = Crew(
#     agents=[sql_agent],
#     tasks=[sql_task],
#     process=Process.sequential,
#     manager_llm=llm
# )

# # 5. Define SQL interface function
# def sql_interface(command):
#     sql_task.description = command
#     result = sql_crew.kickoff()
#     return result

# # 6. Define and launch Gradio interface
# import gradio as gr

# iface = gr.Interface(
#     fn=sql_interface,
#     inputs=gr.Textbox(lines=2, placeholder="Enter SQL command or task"),
#     outputs="text",
#     title="SQL Command Executor",
#     description="Execute SQL commands or tasks on a PostgreSQL database via a natural language interface."
# )

# iface.launch()

# #####################################################




#################################################################






# import openai
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

# # Auto-trace LLM calls in-context
# client = wrap_openai(openai.Client())

# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content

# pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?