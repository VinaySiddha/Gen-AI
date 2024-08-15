import os
import openai
import requests
import uuid
from pymongo import MongoClient
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import gradio as gr
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Load environment variables from a .env file
load_dotenv()

# Get the OpenWeatherMap API key from the environment variable
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your OpenAI API key
MONGODB_URI = os.getenv('MONGODB_CONNECTION_STRING')  # MongoDB URI

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['chatbot_db']  # Corrected the database name to 'chatbot_db'
history_collection = db['conversations']

# Initialize OpenAI with LangSmith tracing
openai_client = openai.Client(api_key=OPENAI_API_KEY)
wrapped_openai = wrap_openai(openai_client)

# Generate a new session ID
def generate_session_id():
    return str(uuid.uuid4())

# Function to store conversation history in MongoDB
def store_history(session_id, location, response):
    history_collection.insert_one({
        'session_id': session_id,
        'location': location,
        'response': response
    })

# Function to retrieve conversation history from MongoDB
def get_history(session_id):
    return list(history_collection.find({'session_id': session_id}))

# Define the weather tool
class Tool:
    @staticmethod
    def get_weather(location):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"The current weather in {location} is {weather} with a temperature of {temp}°C."
        elif response.status_code == 404:
            return f"Sorry, I couldn't find the weather for '{location}'. Please check the spelling or try another location."
        else:
            return "Sorry, there was an error fetching the weather. Please try again later."

# Define an agent to get the weather
weather_agent = Agent(
    role='Weather Reporter',
    goal='Fetch the current weather information',
    backstory='You are an AI tasked with providing weather updates for various locations.',
    verbose=True,
)

# Task for the agent to perform
weather_task = Task(
    description='This will be replaced by user prompt',
    expected_output='Weather',
    agent=weather_agent,
)

# Set up the crew
we_crew = Crew(
    agents=[weather_agent],
    tasks=[weather_task],
)

# Define the traced function
@traceable
def handle_user_input(user_input, session_id):
    # Check if the user is asking about previous locations
    if user_input.lower().strip() == "previous":
        # List previous locations and allow user to choose one
        history_entries = get_history(session_id)
        if not history_entries:
            return "No previous locations found."
        
        history_text = "\nPrevious Locations:\n" + "\n".join(
            [f"{i + 1}: {entry['location']}" for i, entry in enumerate(history_entries)]
        )
        return history_text
    
    # Check if the user input is a request for weather based on a previous location
    try:
        index = int(user_input.strip()) - 1
        history_entries = get_history(session_id)
        if 0 <= index < len(history_entries):
            location = history_entries[index]['location']
            weather_info = Tool.get_weather(location)
            return f"Weather for {location}:\n{weather_info}"
        else:
            return "Invalid index. Please enter a valid number from the previous locations."
    except ValueError:
        # Process new weather requests
        if user_input.strip() == "":
            return "Please enter a valid location."

        weather_task.description = user_input
        result = we_crew.kickoff()

        # Fetch the weather information
        weather_info = Tool.get_weather(user_input)

        # Store the query and response in MongoDB
        store_history(session_id, user_input, weather_info)

        # Format the history for display
        history_entries = get_history(session_id)
        history_text = "\nConversation History:\n" + "\n".join(
            [f"Location: {entry['location']}, Response: {entry['response']}" for entry in history_entries[-5:]]
        )  # Show last 5 interactions

        return f"{weather_info}\n\n{history_text}"

# Generate a new session ID for each new session
session_id = generate_session_id()

# Gradio interface
iface = gr.Interface(
    fn=lambda user_input: handle_user_input(user_input, session_id),
    inputs=gr.Textbox(lines=2, placeholder="Enter a location or 'previous' to list past queries"),
    outputs="text",
    title="Weather Finder with Memory and MongoDB",
    description="Enter a location to get the current weather, or type 'previous' to view past queries and their weather updates."
)

iface.launch()


# import os
# import openai
# import requests
# import uuid
# from pymongo import MongoClient
# from crewai import Agent, Task, Crew
# from dotenv import load_dotenv
# import gradio as gr
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

# # Load environment variables from a .env file
# load_dotenv()

# # Get the OpenWeatherMap API key from the environment variable
# API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Replace with your OpenAI API key
# MONGODB_URI = os.getenv('MONGODB_CONNECTION_STRING')  # MongoDB URI

# # Initialize MongoDB client
# mongo_client = MongoClient(MONGODB_URI)
# db = mongo_client['chatbot_db']  # Corrected the database name to 'chatbot_db'
# history_collection = db['conversations']
# flagged_collection = db['flagged_inputs']  # Collection for storing flagged inputs

# # Initialize OpenAI with LangSmith tracing
# openai_client = openai.Client(api_key=OPENAI_API_KEY)
# wrapped_openai = wrap_openai(openai_client)



# # Generate a new session ID
# def generate_session_id():
#     return str(uuid.uuid4())

# # Function to store conversation history in MongoDB
# def store_history(session_id, location, response):
#     history_collection.insert_one({
#         'session_id': session_id,
#         'location': location,
#         'response': response
#     })

# # Function to retrieve conversation history from MongoDB
# def get_history(session_id):
#     return list(history_collection.find({'session_id': session_id}))

# # Function to flag an input
# def flag_input(session_id, location):
#     flagged_collection.insert_one({
#         'session_id': session_id,
#         'location': location,
#         'flagged': True
#     })

# # Define the weather tool
# class Tool:
#     @staticmethod
#     def get_weather(location):
#         url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
#         response = requests.get(url)
#         data = response.json()

#         if response.status_code == 200:
#             weather = data['weather'][0]['description']
#             temp = data['main']['temp']
#             return f"The current weather in {location} is {weather} with a temperature of {temp}°C."
#         elif response.status_code == 404:
#             return f"Sorry, I couldn't find the weather for '{location}'. Please check the spelling or try another location."
#         else:
#             return "Sorry, there was an error fetching the weather. Please try again later."

# # Define an agent to get the weather
# weather_agent = Agent(
#     role='Weather Reporter',
#     goal='Fetch the current weather information',
#     backstory='You are an AI tasked with providing weather updates for various locations.',
#     verbose=True,
# )

# # Task for the agent to perform
# weather_task = Task(
#     description='This will be replaced by user prompt',
#     expected_output='Weather',
#     agent=weather_agent,
# )

# # Set up the crew
# we_crew = Crew(
#     agents=[weather_agent],
#     tasks=[weather_task],
# )

# # Define the traced function
# @traceable
# def handle_user_input(user_input, session_id, flag=True):
#     # Check if the user is asking about previous locations
#     if user_input.lower().strip() == "previous":
#         # List previous locations and allow user to choose one
#         history_entries = get_history(session_id)
#         if not history_entries:
#             return "No previous locations found."
        
#         history_text = "\nPrevious Locations:\n" + "\n".join(
#             [f"{i + 1}: {entry['location']}" for i, entry in enumerate(history_entries)]
#         )
#         return history_text
    
#     # Check if the user input is a request for weather based on a previous location
#     try:
#         index = int(user_input.strip()) - 1
#         history_entries = get_history(session_id)
#         if 0 <= index < len(history_entries):
#             location = history_entries[index]['location']
#             weather_info = Tool.get_weather(location)
#             return f"Weather for {location}:\n{weather_info}"
#         else:
#             return "Invalid index. Please enter a valid number from the previous locations."
#     except ValueError:
#         # Process new weather requests
#         if user_input.strip() == "":
#             return "Please enter a valid location."

#         weather_task.description = user_input
#         result = we_crew.kickoff()

#         # Fetch the weather information
#         weather_info = Tool.get_weather(user_input)

#         # Store the query and response in MongoDB
#         store_history(session_id, user_input, weather_info)
        
#         # Flag the input if necessary
#         if flag:
#             flag_input(session_id, user_input)
#             weather_info += "\n(Note: This input has been flagged.)"

#         # Format the history for display
#         history_entries = get_history(session_id)
#         history_text = "\nConversation History:\n" + "\n".join(
#             [f"Location: {entry['location']}, Response: {entry['response']}" for entry in history_entries[-5:]]
#         )  # Show last 5 interactions

#         return f"{weather_info}\n\n{history_text}"

# # Generate a new session ID for each new session
# session_id = generate_session_id()


# # Function to handle user input and return the output
# def handle_user_input(user_input, session_id):
#     # The same logic as before for processing user input and fetching weather
#     if user_input.lower().strip() == "previous":
#         history_entries = get_history(session_id)
#         if not history_entries:
#             return "No previous locations found."
        
#         history_text = "\nPrevious Locations:\n" + "\n".join([f"{i + 1}: {entry['location']}" for i, entry in enumerate(history_entries)])

#         return history_text

#     # Process new weather requests
#     if user_input.strip() == "":
#         return "Please enter a valid location."

#     weather_info = Tool.get_weather(user_input)

#     # Store the query and response in MongoDB
#     store_history(session_id, user_input, weather_info)

#     # Format the history for display
#     history_entries = get_history(session_id)
#     history_text = "\nConversation History:\n" + "\n".join([f"Location: {entry['location']}, Response: {entry['response']}" for entry in history_entries[-5:]])  # Show last 5 interactions

#     return f"{weather_info}\n\n{history_text}"

# # Create a new session ID
# session_id = generate_session_id()

# # Gradio Interface with Flagging
# iface = gr.Interface(
#     fn=lambda user_input: handle_user_input(user_input, session_id),
#     inputs=gr.Textbox(lines=2, placeholder="Enter a location or 'previous' to list past queries"),
#     outputs="text",
#     title="Weather Finder with Memory and MongoDB",
#     description="Enter a location to get the current weather, or type 'previous' to view past queries and their weather updates.",
#     allow_flagging="manual",  # Enables the manual flagging option
#     flagging_options=["Wrong location", "Incorrect weather", "Other"]  # Custom flagging options
# )

# iface.launch()

