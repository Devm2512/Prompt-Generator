from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import time

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Defining the prompt
initial_prompt = input("Enter your Prompt: ")
print(initial_prompt)

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Defining the Pydantic Function
class prompt_structure(BaseModel):
    Role: str = Field(description= "Identify the role required for the task. assign only the standard role that exists in the industry. do not define the role by yourself")
    Task_context: str = Field(description= "Identify the task to be completed. state clearly what is expected in minimal words")
    Tone_context: str = Field(description= "Identify the suitable tone based on the input prompt")
    Background_data_focus : str = Field(description="List what information the AI should use. If none is given, say what the user should supply. Max 40 words.")
    Detailed_task_description_and_rules : str = Field(description= "identify the specific constraints and behaviour model must follow. provide the same in bullet points. provide in max 120")
    step_by_step_thinking : str = Field(description= "Provided a chronological order of steps for solving the problem. provide in max 120 words")
    output_formatting: str = Field(description= "identify the suitable output formatting based on the input givenin max 50 words")

model_prompt = PromptTemplate.from_template("""if the task is simple then do not try to paraphrase it.Rewrite the inital prompt below into a clear, well-structured way. consdier the following rules

Rules:
- Write every field as an instruction addressed to the AI.
- Keep each field under 120 words. No markdown.
- Derive context from the given input. Do not invent tools, file names, versions, numbers, or facts.
- If key details are missing, state a short assumption or tell the user what to supply.

Rough prompt:
{original_prompt}. Please ensure that the same content is not repeated.""")


llm = ChatOllama(model = "qwen3:4b", temperature = 0, reasoning= False, num_predict=1200)

structured_llm = llm.with_structured_output(prompt_structure)

chain = model_prompt | structured_llm

print("Generating...........")

start = time.time()

# full_text = ""

# for chunk in chain.stream({"original_prompt": initial_prompt}):
#     print(chunk.content, end = "", flush = True)
#     full_text += chunk.content

result = chain.invoke({"original_prompt" : initial_prompt})
total_time_taken = time.time() - start

# result = prompt_structure.model_validate_json(full_text)
print(result)
print("The total time taken is ", total_time_taken, "seconds")

