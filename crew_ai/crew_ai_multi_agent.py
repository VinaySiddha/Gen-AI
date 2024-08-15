from crewai import Agent, Task, Crew, Process
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENWEATHERMAP_API_KEY')
# Define the function to fetch weather
def fetch_weather(location):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    params = {
         'q': location,
         'appid': api_key,
         'units': 'metric'
     }
    response = requests.get(base_url,params=params)
    data = response.json()

    if response.status_code == 200:
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"The current weather in {location} is {weather} with a temperature of {temp}°C."
    else:
        return None

# Researcher agent to fetch weather data
researcher_agent = Agent(
    role='Weather Data Fetcher',
    goal='Fetch the current weather data for a given location',
    backstory='An expert in accessing and retrieving weather data from APIs.'
)

# Analyst agent to analyze weather data
analyst_agent = Agent(
    role='Weather Data Analyst',
    goal='Analyze the fetched weather data',
    backstory='A specialist in analyzing weather data and extracting key information.'
)

# Writer agent to present weather data
writer_agent = Agent(
    role='Weather Report Writer',
    goal='Present the weather data in a user-friendly format',
    backstory='An expert in creating concise and informative weather reports.'
)

# Task for the researcher agent to fetch weather data
fetch_weather_task = Task(
    description='Fetch the current weather data for a given location',
    expected_output='Raw weather data from the API',
    agent=researcher_agent,
    tools=[fetch_weather]
)

# Task for the analyst agent to analyze weather data
analyze_weather_task = Task(
    description='Analyze the fetched weather data',
    expected_output='Analyzed weather data with key information',
    agent=analyst_agent
)

# Task for the writer agent to present the weather data
write_weather_report_task = Task(
    description='Present the weather data in a user-friendly format',
    expected_output='A user-friendly weather report',
    agent=writer_agent,
    context=[fetch_weather_task, analyze_weather_task]
)

# Define the crew with the agents and tasks
weather_crew = Crew(
    agents=[researcher_agent, analyst_agent, writer_agent],
    tasks=[fetch_weather_task, analyze_weather_task, write_weather_report_task],
    process=Process.sequential,
    # manager_llm=llm
)

# Define a function to start the process
def get_weather_report(location):
    # Fetch weather data
    weather_data = fetch_weather(location)
    if weather_data:
         # Analyze weather data
        # analyzed_data = analyst_agent.(weather_data)
        #  # Write weather report
        # weather_report = writer_agent.handle(analyzed_data)
        return weather_data
    else:
        return "Error: Unable to fetch weather data. Please check the location and API key."

# Example usage
location = input("Enter the location: ")
weather_report = get_weather_report(location)
print(weather_report)