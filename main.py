from langchain_ollama import OllamaLLM

llms = OllamaLLM(model="llama3.1")
default_prompt = "Jawab dalam Bahasa Indonesia: "
user_question = input("Masukkan pertanyaan Anda: ")
prompt = default_prompt + user_question
result = llms.invoke(prompt)
print(result)