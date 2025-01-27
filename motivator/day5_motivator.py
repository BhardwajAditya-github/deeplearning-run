import pyttsx3
from datetime import datetime
import json
import random

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load tasks and motivations from the JSON file
def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        speak("Error: tasks.json file not found.")
        return None
    except json.JSONDecodeError:
        speak("Error: Unable to parse tasks.json.")
        return None

# Main function
def main():
    # Load data from JSON
    data = load_data("E:/LLM Journey/deeplearning-run/motivator/tasks.json")
    if not data:
        return
    
    tasks = data.get("tasks", {})
    motivations = data.get("motivations", [])

    # Get today's date
    today = datetime.now().strftime("%B %d")

    # Greet the user
    speak(f"Good morning! Today is {today}.")

    # Check if there are tasks for today
    if today in tasks:
        speak("Here are your tasks for the day:")
        for task in tasks[today]:
            speak(f"- {task}")
    else:
        speak("You have no tasks scheduled for today. Take some time to relax or plan ahead!")

    # Add a random motivational message
    if motivations:
        motivation = random.choice(motivations)
        speak("And here's your motivational boost:")
        speak(motivation)

# Run the script
if __name__ == "__main__":
    main()
