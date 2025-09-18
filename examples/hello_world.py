## This script exemplify the most basic use of Agentics as a pydantic transducer from 
## list of strings. 

import asyncio
from pydantic import BaseModel
from agentics import Agentics as AG
from typing import Optional
from dotenv import load_dotenv
import os
<<<<<<< HEAD
from agentics.core.llm_connections import  gemini_llm
=======
from agentics.core.llm_connections import  available_llms, get_llm_provider
>>>>>>> b588b61cad8e7a52f6e89ea42eb16dc2f2923e23
load_dotenv()

## Define output type

class Answer(BaseModel):
    answer: Optional[str] = None
    confidence: Optional[float] = None

async def main():

    ## Collect input text

    input_questions = [
        "What is the capital of Italy?",
        "When is the end of the world expected",
    ]

    ## Transduce input strings into objects of type Answer. 
    ## You can customize this providing different llms and instructions. 
    
<<<<<<< HEAD
    answers = await (AG(atype=Answer, 
                        
                        llm=gemini_llm, ##Select your LLM from list of available options
                        instructions="""Provide an Answer for the following input text 
                        only if it contains an appropriate question that do not contain
                        violent or adult language """
                        ) << input_questions)
=======
    answers = await (AG(atype=Answer) << input_questions)
>>>>>>> b588b61cad8e7a52f6e89ea42eb16dc2f2923e23

    print(answers.pretty_print())
    
if __name__ == "__main__":
    # if len(available_llms)>0:
        asyncio.run(main())
    #else: print("Please set API key in your .env file.")
