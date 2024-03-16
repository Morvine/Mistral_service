import io
import json
import os
import minio

from fastapi import FastAPI, Body, status, HTTPException
from fastapi.responses import JSONResponse

import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dotenv import load_dotenv

from app.logic.summary import Conversation, generate



load_dotenv()


MODELS_PATH = f"{os.getenv('MODELS_PATH', '/volumes/ml_models')}/{os.getenv('MODEL_NAME', 'Vikhr-7B-instruct_0.2')}"
model = AutoModelForCausalLM.from_pretrained(
    MODELS_PATH,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODELS_PATH, use_fast=False)

generation_config = GenerationConfig.from_pretrained(MODELS_PATH)
generation_config.max_new_tokens = eval(os.getenv('MAX_NEW_TOKENS', 300))
generation_config.repetition_penalty = 1.1

app = FastAPI()
client = minio.Minio(
    endpoint=os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    # endpoint='127.0.0.1:9000',
    access_key=os.getenv('MINIO_ACCESS_KEY', 'cjKMqPAaGfpnsIdRzNZG'),
    secret_key=os.getenv('MINIO_SECRET_KEY', 'WjNFiKfpZAVBDScjhp6w4KzFSy7jRkuB50EhoVl3'),
    secure=False
)


MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'my-bucket')

@app.post("/summary")
def summary(data=Body()):
    minio_path = data['file_path']
    question = data['question']
    base_path = os.path.dirname(minio_path)

    try:
        response = client.get_object(MINIO_BUCKET, minio_path)
        data = response.data.decode('utf8').replace("'", '"')
        data = json.loads(data)
    except ValueError:
        return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "Файл не найден"}
        )
    finally:
        response.close()
        response.release_conn()
    if 'history' not in data.keys():
        last_user_message = f'Текст стенограммы:\n{data["data"]}\n\nВопрос пользователя:\n"{question}"\n\nЗадание:\nИспользуя текст стенограммы, ответь на вопрос пользователя.\nДля ответа на вопросы используй только информацию, содержащуюся в тексте стенограммы.\nЕсли ответ на вопрос не содержится в стенограмме, выводи "Ответ на вопрос не найден.".'
        inp = last_user_message
    else:
        inp = question

    conversation = Conversation()

    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(tokenizer)
    output = generate(model, tokenizer, prompt, generation_config)
    conversation.add_bot_message(output)

    tmp_history = conversation.messages
    if len(tmp_history) > 11:
        tmp_history = tmp_history[:3] + tmp_history[-8:]

    result = {'data': output, 'history': tmp_history}

    if result is not None and result['data'] is not None:
        value_as_bytes = str(result).encode('utf-8')
        res_filename = f'{base_path}/' + str(hashlib.md5(value_as_bytes).hexdigest()) + '.json'

        res_data = io.BytesIO(value_as_bytes)

        client.put_object(bucket_name=MINIO_BUCKET, object_name=f'{res_filename}', data=res_data,
                          length=len(value_as_bytes))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'result': res_filename})

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Отсутствует результат декодирования"})



