from transformers import pipeline
from langchain.prompts import PromptTemplate
import csv

class CustomHuggingFaceLLM:
    def init(self, pipe):
        self.pipe = pipe

    def call(self, inputs):
        response = self.pipe(inputs["question"], max_length=50)
        return {"text": response[0]["generated_text"]}

class CustomSequentialChain:
    def init(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm

    def call(self, inputs):
        prompt_text = self.prompt.format(question=inputs["question"])
        response = self.llm({"question": prompt_text})
        return response

def format_output(question, rag_result, llm_result):
    output = f"Question: {question}\n\n"
    output += "RAG API Result:\n"
    output += f"{rag_result}\n\n"
    output += "LLM Result:\n"
    output += f"{llm_result}\n"
    output += "-" * 50 + "\n"
    return output

def create_chain(pipe):
    template = """
        Question: {question}
        Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    hf_llm = CustomHuggingFaceLLM(pipe)

    return CustomSequentialChain(QA_CHAIN_PROMPT, hf_llm)

if name == 'main':
    from RagApi import RagApi  # Asumimos que RagApi está configurado correctamente
    ra = RagApi(load_vectorstore=True)  # change this line to build from scratch

    # Usar el pipeline de transformers
    pipe = pipeline("text-generation", model="mlabonne/OrpoLlama-3-8B")

    # Crear la cadena LLM utilizando el pipeline de Hugging Face
    nonragchain = create_chain(pipe)

    questions = [
        "How do I get a job at Google?",
        "What are the key elements of a strong resume?",
        "How can I prepare for a technical interview?",
        "What are some tips for negotiating salary?",
        "How can I improve my work-life balance?",
        "How can I showcase my problem-solving skills on my resume?",
        "What are some effective ways to highlight my technical skills on my resume?",
        "How can I quantify my achievements and impact on my resume?",
        "How can I showcase my leadership and teamwork skills on my resume?",
        "How can I make my resume stand out among a large pool of applicants?",
    ]

    results = []

    for question in questions:
        rag_result = ra.chain({"query": question})["result"]
        llm_result = nonragchain({"question": question})["text"]

        results.append([question, rag_result, llm_result])

        print(format_output(question, rag_result, llm_result))

    # Export results to a CSV file
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "RAG API Result", "LLM Result"])
        writer.writerows(results)
