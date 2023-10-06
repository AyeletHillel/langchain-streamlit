import os 
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


os.environ['OPENAI_API_KEY'] = apikey

st.title("🍳 Cooking Buddy")
prompt = st.text_input("What ingredients do you have??")


## Prompt Templates
meal_template = PromptTemplate(
    input_variables=['ingredients'],
    template="What is a single meal name that can be made with these ingredients: {ingredients}",
)


recipe_template = PromptTemplate(
    input_variables=['meal'],
    template="Provide a simple recipe for making: {meal}. Begin your response with a cheerful message like 'Fantastic, {meal} can be a great choice!' and then provide a simple recipe for it",
)

## llms 
llm = OpenAI(temperature=0.9)
meal_chain = LLMChain(llm=llm, prompt=meal_template, verbose=True)
recipe_chain = LLMChain(llm=llm, prompt=recipe_template, verbose=True)


simple_sequential_chain= SimpleSequentialChain(chains=[meal_chain, recipe_chain], verbose=True)

if prompt:
    
        response = simple_sequential_chain.run(prompt)
        st.write(response)
        
    
    
        
