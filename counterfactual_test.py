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

from utils import *

np.random.seed(42)

load_dotenv()

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
openrouter = OpenAI(
    base_url = 'https://openrouter.ai/api/v1',
    api_key = os.environ.get('OPENROUTER_API_KEY')
)


def get_data(d_desc = 'MI', task = 'retrospective'):
    if d_desc == 'ESC':
        task_desc = 'Emotional Support Conversation'
    elif d_desc == 'MI':
        task_desc = 'Counseling Session'
    elif d_desc == 'PFG':
        task_desc = 'Persuasion Conversation'

    combined_data = json.load(open(f'verified_data/{d_desc}_{task}_verified.json', 'r'))
    selected_subs = list(combined_data.keys())

    data = []

    for sub_id in selected_subs:
        id_ = sub_id.split('_')[0]
        state = sub_id.split('_')[1]

        curr_data = combined_data[id_]

        if task == 'prospective':
            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'correct_action': curr_data['actions']['correct_action'],
                'distractors': curr_data['actions'][state],
                'state': state,
                'topic': curr_data[id_]['topic'],
                'task_desc': task_desc
            })
        elif task == 'prospective-easy':
            tmp = selected_subs.copy()
            tmp.remove(id_)
            random_ids = random.sample(tmp, 3)

            random_distractors = []
            for rid in random_ids:
                random_distractors.append('\n'.join(combined_data[rid]['ctx'][:4]))

            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'correct_action': curr_data['actions']['correct_action'],
                'distractors': random_distractors,
                'state': state,
                'topic': curr_data[id_]['topic'],
                'task_desc': task_desc
            })
        else:
            data.append({
                'id': id_,
                'ctx': curr_data['ctx'],
                'correct_option': curr_data['correct_option'],
                'options': curr_data['options'],
                'state': state,
                'topic': curr_data[id_]['topic'],
                'task_desc': task_desc
            })
    
    return data

def generate_counterfactual_mental(data, d_desc, model):
    print("Generate Data")

    counterfactual_mental_states = {
    }

    for curr_data in tqdm(data):
        all_mental_states = []

        ctx = curr_data['ctx']
        ctx = "\n".join(ctx)
        ctx = ctx.replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])

        id_ = curr_data['id']
        state_ = curr_data['state']

        for state in ['Belief', 'Desires', 'Intentions', 'Emotions', 'Knowledge', 'Trust']:
            curr_state_text = curr_data['options'][state][curr_data['correct_option'][state]]
            
            prompt = f"""You are an expert in Theory of Mind reasoning and generating counterfactual statements.

Task:
You will be provided with an excerpt from a conversation along with the current {state} state of {agent2[d_desc]}.

You need to generate a counterfactual mental state that is obviously not true at all given the current conversation. It should be CONTRARY to the provided state.

Conversation context:
{ctx}

Mental {state} State:
{curr_state_text}

Instruction
Output only the counterfactual . Do not add explanations or other verbosity.
Your output should strictly follow the structure "{AGENT2_BELIEF_STEER_PREPROMPT[state].replace('[@agent1]', agent1[d_desc])}". NO FORMATTING NEEDS TO BE DONE.

Counterfactual:
"""
            chat_completion = openai.responses.parse(
                model="gpt-4o",
                input=[
                    {'role': 'user', 'content': prompt}
                ],
                text_format=freeText,
            ).output_parsed.text

            all_mental_states.append(chat_completion)

        counterfactual_mental_states[f'{id_}_{state_}'] = all_mental_states
    
    with open(f'results/{simplified_models[model]}/{d_desc}_counterfact.json', 'w+') as f:
        json.dump(counterfactual_mental_states, f)
    f.close()


def prospective(data, model_id='gemini-3-pro-preview', d_desc='MI', filename='counter.csv'):
    print('Prospective Countefactual Experiment', model_id, d_desc)
    mental_states = ['Belief', 'Desires', 'Intentions', 'Emotions', 'Knowledge', 'Trust']

    out_dir = 'results'

    counterfactual_data = json.load(open(f'counterfactual_data/{d_desc}_counterfact.json', 'r', encoding='utf8'))

    if not os.path.exists(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}.csv')):
        res = {
            'data': [],
            'time': [],
            'model': [],
            'subsession_id': [],
            'mental_state': [],
            'correct_option': [],
            'chosen_option': [],
            'correct_option_text': [],
            'thoughts': []
        }

        curr_idx = 0
    else:
        res = pd.read_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}.csv'))

        curr_idx = len(res)

        res = res.to_dict()

        res = {key: list(value.values()) for key, value in res.items()}


    correct_res = []
    thoughts = []

    print(curr_idx)

    for curr_data in tqdm(data[curr_idx:]):
        all_mental_states = []

        ctx = curr_data['ctx']
        ctx = "\n".join(ctx)
        ctx = ctx.replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])

        id_ = curr_data['id']
        state_ = curr_data['state']

        all_mental_states = counterfactual_data[f'{id_}_{state_}']
        for i, state in enumerate(mental_states):
            all_mental_states[i] = f"{state}: {all_mental_states[i]}"

        correct_opt = curr_data['correct_action']

        curr_opts = curr_data['distractors']

        option_list = [correct_opt] + curr_opts
        np.random.shuffle(option_list)
        correct_action_option = "E"

        action_options = f"""A: {option_list[0].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nB: {option_list[1].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nC: {option_list[2].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nD: {option_list[3].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nE: None of the above."""

        prompt=f"""
You are an expert in Theory of Mind reasoning.

Task:
You will be provided with internal Mental State profile (Belief, Desire, Intention, Emotion, Knowledge, Trust) of client during the conversation.

Your goal is to identify which of the candidate conversation segments is the most plausible continuation of this conversation.
The correct option must be consistent with the provided Mental States of the client. It can also be that none of the options are correct.

Mental state of {agent2[d_desc]}
{"\n".join(all_mental_states)}

Candidate Conversation Segments
{action_options}

Instruction
Output only the letter of the correct option (e.g., "A", "B", "C", "D", or "E"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D, E. NO FORMATTING NEEDS TO BE DONE.

Answer:
"""
        if 'gemini' in model_id:
            response = gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=counterfactMCQ.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )

            thought_summary = ''.join([part.text for part in response.candidates[0].content.parts if part.thought])
            option = response.parsed['option']
            thoughts.append(thought_summary)

        elif 'gpt' in model_id and 'openai' not in model_id:
            if model_id == 'gpt-5':
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counterfactMCQ,
                    reasoning={
                        "effort": "medium", # or 'high' for more complex tasks
                        "summary": "detailed",
                    }
                )
            else:
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counterfactMCQ,
                )

            option = chat_completion.output_parsed.option
            thoughts.append('')
        else:
            option = openrouter.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                extra_body=dict(requires_parameters=True, providers=dict(sort='latency'), response_format=dict(type='json_schema', json_schema=counsMCQ.model_json_schema(), strict=True)
                ),
            ).choices[0].message.content

            thoughts.append('')

        correct_res.append(option == correct_action_option)

        res['data'].append(d_desc)
        res['time'].append(datetime.now().strftime("%Y %m %d-%H:%M:%S"))
        res['model'].append(model_id)
        res['subsession_id'].append(id_)
        res['mental_state'].append(state_)
        res['correct_option'].append(correct_action_option)
        res['chosen_option'].append(option)
        res['correct_option_text'].append(correct_opt)

        if 'thoughts' in res:
            res['thoughts'].append(thoughts[-1])

        out_dir = 'results'

        os.makedirs(os.path.join(os.path.join(out_dir, simplified_models[model_id])), exist_ok=True)
        res_df = pd.DataFrame(res)

        res_df.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}.csv'), index=False)

    print(correct_res, sum(correct_res)/len(correct_res))

    print('-----------------------')

def parse_args():
    parser = ArgumentParser(description="Benchmarking script for machine ToM")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to benchmark")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for d_desc in ['MI', 'ESC', 'PFG']:
        data = get_data(d_desc=d_desc, task='prospective')

    #     generate_counterfactual_mental(data, d_desc)

        prospective(data, model_id=args.model, d_desc=d_desc, filename='counter.csv')
