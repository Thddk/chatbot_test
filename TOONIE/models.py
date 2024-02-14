import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from auto_gptq import AutoGPTQForCausalLM

def get_models(model_name):
    if model_name == "Ko-PlatYi-6B":
        return get_koplatyi_models()
    elif model_name == "kullm":
        return get_kullm_models()
    elif model_name == "koalpaca":
        return get_koalpaca_models()
    elif model_name == "test":
        return get_test_models()


def get_koplatyi_models():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id = "kyujinpy/Ko-PlatYi-6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    prompt_template = """
            [INST] <<SYS>>
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
            <</SYS>>
            
            Generate the next agent response by answering the question. Answer it as succinctly as possible. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.
            CONTEXT:
            
            {context}
            Question : {question}[\INST]
            Make sure to answer in Korean.
     """
    return tokenizer, model, prompt_template


def get_kullm_models():
    tokenizer = "j5ng/kullm-5.8b-GPTQ-8bit"
    model = AutoGPTQForCausalLM.from_quantized(tokenizer, device="cuda:0", use_triton=False)
    prompt_template = """
    아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요.

    ### 명령어:
    {context}

    ### 입력:
    {question}

    ### 응답:

     """
    return tokenizer, model, prompt_template


def get_koalpaca_models():
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    prompt_template = """
    ### 질문: {question}
    
    ### 맥락: {context}
    
    ### 답변:

     """
    return tokenizer, model, prompt_template

def get_test_models():
    tokenizer = AutoTokenizer.from_pretrained("LDCC/LDCC-SOLAR-10.7B")
    model = AutoModelForCausalLM.from_pretrained("LDCC/LDCC-SOLAR-10.7B")

    # tokenizer = AutoTokenizer.from_pretrained("PracticeLLM/SOLAR-tail-10.7B-Merge-v1.0")
    # model = AutoModelForCausalLM.from_pretrained("PracticeLLM/SOLAR-tail-10.7B-Merge-v1.0")
    # tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-0-70b-16bit")
    # model = AutoModelForCausalLM.from_pretrained("upstage/SOLAR-0-70b-16bit")

    prompt_template = """
             [INST]
            Answer the following QUESTION based on the CONTEXT given. If you do not know the answer and the CONTEXT doesn't contain the answer truthfully say "I don't know".CONTEXT: {context}
            Question : {question}
            [\INST]
                """
    return tokenizer, model, prompt_template



# # prompt 예시
# prompt_template = """
#             [INST] <<SYS>>
#             You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#
#             If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
#             <</SYS>>
#
#             Make sure to answer in Korean. Generate the next agent response by answering the question. Answer it as succinctly as possible. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.
#             CONTEXT:
#
#             {context}
#             Question : {question}[\INST]
#
#             """

# """Answer based on context:\n\n{context}\n\n{question}"""

# prompt_template = """Answer the following QUESTION based on the CONTEXT
# given. If you do not know the answer and the CONTEXT doesn't
# contain the answer truthfully say "I don't know".
#
# CONTEXT:
# {context}
#
#
# ANSWER:
# """

# if word_kor and word_kor != 'None':
#     condense_template = """
#         <history>
#         {chat_history}
#         </history>
#
#         Human: <history>를 참조하여, 다음의 <question>의 뜻을 명확히 하는 새로운 질문을 한국어로 생성하세요. 새로운 질문은 원래 질문의 중요한 단어를 반드시 포함합니다.
#
#         <question>
#         {question}
#         </question>
#
#         Assistant: 새로운 질문:"""
# else:
#     condense_template = """
#         <history>
#         {chat_history}
#         </history>
#         Answer only with the new question.
#
#         Human: using <history>, rephrase the follow up <question> to be a standalone question. The standalone question must have main words of the original question.
#
#         <quesion>
#         {question}
#         </question>
#
#         Assistant: Standalone question:"""


# def llama_v2_prompt(
#     messages: list[dict]
# ):
#     B_INST, E_INST = "[INST]", "[/INST]"
#     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#     BOS, EOS = "<s>", "</s>"
#     DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#
#     if messages[0]["role"] != "system":
#         messages = [
#             {
#                 "role": "system",
#                 "content": DEFAULT_SYSTEM_PROMPT,
#             }
#         ] + messages
#     messages = [
#         {
#             "role": messages[1]["role"],
#             "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
#         }
#     ] + messages[2:]
#
#     messages_list = [
#         f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
#         for prompt, answer in zip(messages[::2], messages[1::2])
#     ]
#     messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
#
#     return "".join(messages_list)
#
#
