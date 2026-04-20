import json
import os, requests
from openai import OpenAI
import random
from dotenv import load_dotenv
import numpy as np

import pandas as pd

from tqdm import tqdm

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import time
from datetime import datetime

from argparse import ArgumentParser

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils import freeText

np.random.seed(42)

load_dotenv()

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openrouter = OpenAI(
    base_url = 'https://openrouter.ai/api/v1',
    api_key = os.environ.get('OPENROUTER_API_KEY')
)

option_idx = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

agent1 = {
    'MI': 'Counselor',
    'PFG': 'Persuader',
    'ESC': 'Supporter'
}

agent2 = {
    'MI': 'Client',
    'PFG': 'Persuadee',
    'ESC': 'Seeker'
}

simplified_models = {
    'gemini-3-pro-preview': 'gemini-3-pro',
    'gemini-2.5-flash': 'gemini-2.5-flash', # MI half done
    'gemini-2.5-pro': 'gemini-2.5-pro', # MI done
    'mistralai/mistral-nemo': 'mistral-nemo', # MI done
    'mistralai/mistral-small-3.2-24b-instruct': 'mistral-3.2-24b', # MI done
    'moonshotai/kimi-k2-0905': 'kimi-k2', # MI done
    'qwen/qwen3-235b-a22b-2507': 'qwen3-235b', # MI done
    'qwen/qwen3-32b': 'qwen3-32b', # MI done
    'meta-llama/llama-4-maverick': 'llama4-mave', # MI done
    'meta-llama/llama-3.3-70b-instruct': 'llama-3.3-70b', # MI done, PFG done
    'meta-llama/llama-3.1-8b-instruct': 'llama-8b',
    'deepseek/deepseek-chat-v3-0324': 'deepseek-v3', # MI done
    'deepseek/deepseek-r1-0528': 'deepseek-r1',
    'openai/gpt-oss-120b': 'gpt-oss-120b', # MI done, PFG done
    'gpt-4o': 'gpt-4o', # not needed since generates data
    'gpt-4.1': 'gpt-4.1', # MI done, PFG done
    'gpt-5': 'gpt-5', # MI done
}

def get_data(d_desc = 'MI', task = 'retrospective'):
    if d_desc == 'ESC':
        task_desc = 'Emotional Support Conversation'
    elif d_desc == 'MI':
        task_desc = 'Counseling Session'
    elif d_desc == 'PFG':
        task_desc = 'Persuasion Conversation'

    combined_data = json.load(open(f'data/{d_desc}_{task}_verified.json', 'r', encoding='utf8'))

    idx_to_id = {}
    for i in range(len(combined_data)):
        idx_to_id[combined_data[i]['id']] = i

    selected_subs = list(idx_to_id.keys())

    data = []

    for sub_id in selected_subs:
        curr_data = combined_data[idx_to_id[sub_id]]

        id_ = sub_id
        state = curr_data['state']

        if task == 'prospective':
            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'correct_action': curr_data['correct_action'],
                'distractors': curr_data['distractors'],
                'state': state,
                'topic': curr_data['topic'],
                'task_desc': task_desc
            })
        elif task == 'prospective-easy':
            tmp = selected_subs.copy()
            tmp.remove(id_)
            random_ids = random.sample(tmp, 3)

            random_distractors = []
            for rid in random_ids:
                random_distractors.append('\n'.join(combined_data[idx_to_id[rid]]['ctx'][:4]))

            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'correct_action': curr_data['correct_action'],
                'distractors': random_distractors,
                'state': state,
                'topic': curr_data['topic'],
                'task_desc': task_desc
            })
        else:
            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'state': state,
                'topic': curr_data['topic'],
                'task_desc': task_desc
            })
    
    return data


def memorize(data, d_desc='MI', model_id='gemini-3-pro', filename='memorized_actions.csv'):
    out_dir = 'results'

    if not os.path.exists(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}')):
        res = {
            'data': [],
            'time': [],
            'model': [],
            'subsession_id': [],
            'mental_state': [],
            'chosen_option': [],
            'correct_option_text': [],
        }

        curr_idx = 0
    else:
        res = pd.read_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'))

        curr_idx = len(res)

        res = res.to_dict()

        res = {key: list(value.values()) for key, value in res.items()}
    
    for curr_data in tqdm(data[curr_idx:]):
        ctx = curr_data['ctx']
        ctx = "\n".join(ctx)
        ctx = ctx.replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])

        id_ = curr_data['id']
        state_ = curr_data['state']

        correct_opt = curr_data['correct_action']

        prompt = f"""
You are an expert in memory recall.

Task:
You are given a conversation context between two agents, {agent1[d_desc]} and {agent2[d_desc]}, in a {curr_data['task_desc']}. The conversation is about the topic of "{curr_data['topic']}".

Your goal is to complete the conversation for four turns, one for each agent (following after the final turn in the given context). If you have seen this conversation before, you should be able to recall the next four turns accurately. If you have not seen this conversation before, you should do your best to predict plausible next turns based on the given context.

Conversation
{ctx}

Future turns:
"""
        
        if 'gemini' in model_id:
            response = gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=freeText.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )

            thought_summary = ''.join([part.text for part in response.candidates[0].content.parts if part.thought])
            option = response.parsed['text']

        elif 'gpt' in model_id and 'openai' not in model_id:
            if model_id == 'gpt-5':
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=freeText,
                    reasoning={
                        "effort": "medium", # or 'high' for more complex tasks
                        "summary": "detailed",
                    }
                )

                option = chat_completion.output_parsed.text
        
        res['data'].append(d_desc)
        res['time'].append(datetime.now().strftime("%Y %m %d-%H:%M:%S"))
        res['model'].append(model_id)
        res['subsession_id'].append(id_)
        res['mental_state'].append(state_)
        res['chosen_option'].append(option)
        res['correct_option_text'].append(correct_opt)

        out_dir = 'results'

        os.makedirs(os.path.join(os.path.join(out_dir, simplified_models[model_id])), exist_ok=True)
        res_df = pd.DataFrame(res)
        res_df.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'), index=False)


def parse_args():
    parser = ArgumentParser(description="Script for Memorization pilot study on Prospective task")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to benchmark")
    parser.add_argument(
        "--filename", type=str, default="memorize.csv", help="Filename to save results as"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for d_desc in ['MI', 'ESC', 'PFG']:
        print('Memorization?', args.model, d_desc)
        data = get_data(d_desc=d_desc, task='prospective')

        memorize(data, d_desc=d_desc, model_id=args.model, filename=args.filename)