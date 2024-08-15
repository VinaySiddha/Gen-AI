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

# Get the OpenWeatherMap API key and other necessary keys from the environment variables
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MONGODB_URI = os.getenv('MONGODB_CONNECTION_STRING')

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client['chatbot_db']
history_collection = db['conversations']
flagged_collection = db['flagged_inputs']

# Initialize OpenAI with LangSmith tracing
openai_client = openai.Client(api_key=OPENAI_API_KEY)
wrapped_openai = wrap_openai(openai_client)

# Function to generate a new session ID
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

# Function to flag an input
def flag_input(session_id, location):
    flagged_collection.insert_one({
        'session_id': session_id,
        'location': location,
        'flagged': True
    })

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

# Define the traced function for handling user input
@traceable
def handle_user_input(user_input, session_id, flag=False):
    if user_input.lower().strip() == "previous":
        history_entries = get_history(session_id)
        if not history_entries:
            return "No previous locations found."
        
        history_text = "\nPrevious Locations:\n" + "\n".join(
            [f"{i + 1}: {entry['location']}" for i, entry in enumerate(history_entries)]
        )
        return history_text
    
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
        if user_input.strip() == "":
            return "Please enter a valid location."

        weather_task.description = user_input
        result = we_crew.kickoff()

        weather_info = Tool.get_weather(user_input)
        store_history(session_id, user_input, weather_info)
        
        if flag:
            flag_input(session_id, user_input)
            weather_info += "\n(Note: This input has been flagged.)"

        history_entries = get_history(session_id)
        history_text = "\nConversation History:\n" + "\n".join(
            [f"Location: {entry['location']}, Response: {entry['response']}" for entry in history_entries[-5:]]
        )

        return f"{weather_info}\n\n{history_text}"

# Generate a new session ID
session_id = generate_session_id()

# Define a custom FlaggingCallback class
class CustomFlaggingCallback:
    def __init__(self, session_id):
        self.session_id = session_id
    
    def setup(self):
        # Setup is called during interface initialization
        pass
    
    def handle(self, user_input, output, flag_option):
        flag_input(self.session_id, user_input)
        return f"Flagged '{user_input}' with reason: {flag_option}"

# Create an instance of the custom flagging callback
custom_flagging_callback = CustomFlaggingCallback(session_id)

iface = gr.Interface(
    fn=lambda user_input: handle_user_input(user_input, session_id),
    inputs=gr.Textbox(lines=2, placeholder="Enter a location or 'previous' to list past queries"),
    outputs="text",
    title="Weather Finder with Memory and MongoDB",
    description="Enter a location to get the current weather, or type 'previous' to view past queries and their weather updates.",
    allow_flagging="manual",
    flagging_options=["Wrong location", "Incorrect weather", "Other"],
    flagging_callback=custom_flagging_callback
)

# Launch the Gradio Interface
iface.launch()
