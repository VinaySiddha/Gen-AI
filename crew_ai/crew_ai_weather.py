# import os
# import requests
# from crewai import Agent, Task, Crew
# from dotenv import load_dotenv
# import gradio as gr

# # Load environment variables from a .env file
# load_dotenv()

# # Get the OpenWeatherMap API key from the environment variable
# API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# # In-memory storage for user queries
# previous_queries = []

# class tool:
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

# # Function to handle user queries and remember previous ones
# def fun(location):
#     if location.strip() == "":
#         return "Please enter a valid location."

#     weather_task.description = location
#     result = we_crew.kickoff()

#     # Store the user's query
#     previous_queries.append(location)

#     # Fetch the weather information
#     weather_info = tool.get_weather(location)

#     # Combine results with memory of past queries
#     if previous_queries:
#         previous_queries_info = "\nPrevious queries:\n" + "\n".join(previous_queries[-5:])  # Show last 5 queries
#     else:
#         previous_queries_info = ""

#     return f"{weather_info}\n{previous_queries_info}"

# # Gradio interface
# iface = gr.Interface(
#     fn=fun,
#     inputs=gr.Textbox(lines=2, placeholder="Enter a location (e.g., 'New York')"),
#     outputs="text",
#     title="Weather Finder with Memory",
#     description="Enter a location to get the current weather. The tool will also remember your last few queries."
# )

# iface.launch()



import os
import requests
from crewai import Agent, Task, Crew,crew
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the OpenWeatherMap API key from the environment variable
API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

class tool:
    def get_weather(location):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"The current weather in {location} is {weather} with a temperature of {temp}°C."
        else:
            return "Sorry, I couldn't find the weather for that location."

# Define an agent to get the weather
weather_agent = Agent(
    role='Weather Reporter',
    goal='Fetch the current weather information',
    backstory='You are an AI tasked with providing weather updates for various locations.',
    verbose=True,
    # tools=[tool.get_weather]
)

# Task for the agent to perform
weather_task = Task(
    description='This will be replaced by user prompt',
    expected_output='Weather',
    agent=weather_agent,
    # tools=[tool.get_weather]
)

# Assign the task to the agent
# weather_task = WeatherTask(
#     description="Get the current weather for a specified location.",
#     agent=weather_agent,
#     expected_output="A string describing the current weather and temperature."
# )

    # Set up the crew
we_crew = Crew(
        agents=[weather_agent],
        tasks=[weather_task],
)

 
   # Kickoff the crew to perform the task
def fun(location):
    weather_task.description = location
    result = we_crew.kickoff()
    # Print the result
    return result

# 6. Define and launch Gradio interface
import gradio as gr

iface = gr.Interface(
    fn=fun,
    inputs=gr.Textbox(lines=10, placeholder="Enter location"),
    outputs="text",
    title="Weather Finder",
    description="Enter yourLocation to find weather"
)

iface.launch()