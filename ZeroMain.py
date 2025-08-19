#ZeroMain.py
"""
This is the main chatloop of the app. 
It can be run for CL I/O, or the GUI can use it as backend.
Is fully config defined, so if its not working, first thing to check is the config.py file.
"""

import memory
from config import LLM_SOURSE, OPENROUTER_API_KEY, LOCAL_MODEL_PATH, GEMINI_API_KEY


def call_LLM(prompt):
    if LLM_SOURSE == "OPENROUTER":
        #Placeholder for the actuall call
    elif LLM_SOURSE == "LOCAL":
        #Placeholder for the actuall call
    elif LLM_SOURSE == "GEMINI":
        #Placeholder for the actual call
    else:
        print "invalid config"

def call_LLM_with_memory(prompt):
    #Placeholder for the memory augmented LLM call
    #The API to get the memory input is: memory.full(prompt)
    #memory.full(prompt) funktion does not return the prompt back, so we need to put it here. 
    #so we need to combine both the prompt and the return of the funktion memory.full(prompt)





if __name__ == __main__:
    #the chatloop goes here