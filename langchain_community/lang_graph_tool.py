import requests

class WeatherQueryNode:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process(self, state):
        location = state.get('location', 'London')
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.api_key}"
        )
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        state['weather'] = f"The weather in {location} is {weather_description} with a temperature of {temperature}K."
        return state

class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.start_node = None
        self.end_node = None

    def add_node(self, name, node):
        self.nodes[name] = node

    def set_start(self, name):
        self.start_node = self.nodes.get(name)

    def set_end(self, name):
        self.end_node = self.nodes.get(name)

    def run(self, state):
        if not self.start_node:
            raise ValueError("Start node not set")
        current_state = state
        current_state = self.start_node.process(current_state)
        return current_state

def create_weather_graph(api_key: str):
    weather_node = WeatherQueryNode(api_key)
    graph = StateGraph()
    graph.add_node("weather_query", weather_node)
    graph.set_start("weather_query")
    graph.set_end("weather_query")
    return graph

if __name__ == "__main__":
    api_key = "8c2ae2725a5727448f1b874e3c7292fa"  # Replace with your actual OpenWeather API key

    # Create the state graph
    weather_graph = create_weather_graph(api_key)

    # Example: Running the graph with a user's location input
    initial_state = {'location': 'India'}
    
    final_state = weather_graph.run(initial_state)

    # Print the weather information
    print(final_state['weather'])
