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

def retrospective(data, model_id='gemini-3-pro-preview', d_desc='MI', filename='retrospective.csv'):

    print('Retrospective', model_id, d_desc)

    out_dir = "results"

    if not os.path.exists(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}')):
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
        res = pd.read_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'))

        curr_idx = len(res)

        res = res.to_dict()

        res = {key: list(value.values()) for key, value in res.items()}

    thoughts = []
    correct_res = []

    for curr_data in tqdm(data[curr_idx:]):
        id_ = curr_data['id']
        ctx = "\n".join(curr_data['ctx'])
        curr_state = curr_data['state']
        correct_opt = curr_data['correct_option'][curr_state]
        topic = curr_data['topic']
        task_desc = curr_data['task_desc']

        option_list = [f"{value}" for key, value in curr_data['options'][curr_state].items()]
        
        np.random.shuffle(option_list)

        correct_option = option_idx[option_list.index(curr_data['options'][curr_state][correct_opt])]

        options = f"""A: {option_list[0]}\nB: {option_list[1]}\nC: {option_list[2]}\nD: {option_list[3]}"""

        prompt = f"""
You are an expert in Theory of Mind reasoning.

Task:
You will be provided with conversation between two agents {agent1[d_desc]} and {agent2[d_desc]} engaging in a {task_desc} session on the topic of {topic}.

Your goal is to correctly infer {agent2[d_desc]}'s {curr_state} state, based on the above conversation. You will be provided with a set of options, and you need to choose the most appropriate one that reflects the {curr_state} state.

The correct option must be consistent with the provided conversation context.

Conversation Context:
{ctx}

Mental State Options:
{options}

Instruction
Output only the letter of the correct option (e.g., "A", "B", "C", or "D"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE. ONLY OUTPUT THE OPTION AND NOTHING ELSE. YOUR OUTPUT SHOULD STRICTLY BE ONE OF A, B, C, or D.

Answer:
    """.replace(
            "[@conversation]", "\n".join(curr_data['ctx'])).replace(
            "[@agent1]", agent1[d_desc]).replace(
            "[@agent2]", agent2[d_desc]).replace(
            "[@task]", curr_data['task_desc']).replace(
            "[@topic]", curr_data['topic']).replace(
            "[@mental_state]", curr_data['state']).replace(
            "[@options]", "\n".join("\n".join(options)))

        if 'gpt' in model_id and 'openai' not in model_id:
            if model_id == 'gpt-5':
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counsMCQ,
                    reasoning={
                        "effort": "medium", # or 'high' for more complex tasks
                        "summary": "detailed",
                    }
                )

                x = []
                for each_summ in chat_completion.output[0].summary:
                    x.append(each_summ.text)

                thoughts.append("\n".join(x))
            else:
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counsMCQ,
                )
        
            option = chat_completion.output_parsed.option

        elif 'gemini' in model_id:
            response = gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=counsMCQ.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )

            thought_summary = ''.join([part.text for part in response.candidates[0].content.parts if part.thought])
            option = response.parsed['option']
            thoughts.append(thought_summary)

        else:
            option = openrouter.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                extra_body=dict(requires_parameters=True, providers=dict(sort='latency'), response_format=dict(type='json_schema', json_schema=counsMCQ.model_json_schema(), strict=True)
                ),
            ).choices[0].message.content
            
            # option = get_openrouter_resp(prompt, model_id)

        correct_res.append(option == correct_option)

        res['data'].append(d_desc)
        res['time'].append(datetime.now().strftime("%Y %m %d-%H:%M:%S"))
        res['model'].append(model_id)
        res['subsession_id'].append(id_)
        res['mental_state'].append(curr_state)
        res['correct_option'].append(correct_option)
        res['chosen_option'].append(option)
        res['correct_option_text'].append(curr_data['options'][curr_state][correct_opt])

        if 'thoughts' in res:
            res['thoughts'].append(thoughts[-1])

        os.makedirs(os.path.join(os.path.join(out_dir, simplified_models[model_id])), exist_ok=True)
        res_df = pd.DataFrame(res)
        
        if 'gemini' in model_id and 'thoughts' not in res_df:
            res_df['thoughts'] = thoughts

        res_df.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'), index=False)

    print(correct_res, sum(correct_res)/len(correct_res))

    print('-----------------------')
    
    return option


def prospective(data, model_id='gemini-3-pro-preview', d_desc='MI', exp='normal', filename='prospective.csv'):
    print('Prospective', model_id, d_desc)
    mental_states = ['Belief', 'Desires', 'Intentions', 'Emotions', 'Knowledge', 'Trust']

    out_dir = 'results'

    if not os.path.exists(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}')):
        res = {
            'data': [],
            'time': [],
            'model': [],
            'subsession_id': [],
            'mental_state': [],
            'correct_option': [],
            'chosen_option': [],
            'correct_option_text': [],
        }

        curr_idx = 0
    else:
        res = pd.read_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'))

        curr_idx = len(res)

        res = res.to_dict()

        res = {key: list(value.values()) for key, value in res.items()}

    correct_res = []
    thoughts = []

    for curr_data in tqdm(data[curr_idx:]):
        all_mental_states = []

        ctx = curr_data['ctx']
        ctx = "\n".join(ctx)
        ctx = ctx.replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])

        id_ = curr_data['id']
        state_ = curr_data['state']

        for state in mental_states:
            state_text = curr_data['options'][state][curr_data['correct_option'][state]]
            all_mental_states.append(f"{state}: {state_text}")

        correct_opt = curr_data['correct_action']

        curr_opts = curr_data['distractors']

        option_list = [correct_opt] + curr_opts
        np.random.shuffle(option_list)
        correct_action_option = option_idx[option_list.index(correct_opt)]

        action_options = f"""A: {option_list[0].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nB: {option_list[1].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nC: {option_list[2].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}\nD: {option_list[3].replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc])}"""

        if exp == 'NOTA':
            action_options += "\nE: NOTA."
            instructions = """Output only the letter of the correct option (e.g., "A", "B", "C", "D", or "E"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D, E. NO FORMATTING NEEDS TO BE DONE. Choose Option E if you think none of the options are correct continuations."""
        elif exp == 'CoT':
            instructions = """Output only the letter of the correct option (e.g., "A", "B", "C", or "D"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE.

Let's think step-by-step."""
        else:
            instructions = """Output only the letter of the correct option (e.g., "A", "B", "C", or "D"). Do not add explanations or other verbosity.
Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE."""

        prompt=f"""
You are an expert in Theory of Mind reasoning.

Task:
You will be provided with internal Mental State profile (Belief, Desire, Intention, Emotion, Knowledge, Trust) of client during the conversation.

Your goal is to identify which of the candidate conversation segments is the most plausible continuation of this conversation.
The correct option must be consistent with the provided Mental States of the client.

Mental state of {agent2[d_desc]}
{"\n".join(all_mental_states)}

Candidate Conversation Segments
{action_options}

Instruction
{instructions}

Answer:
"""

        if 'gpt' in model_id and 'openai' not in model_id:
            if model_id == 'gpt-5':
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counsMCQ,
                    reasoning={
                        "effort": "medium", # or 'high' for more complex tasks
                        "summary": "detailed",
                    }
                )

                x = []
                for each_summ in chat_completion.output[0].summary:
                    x.append(each_summ.text)

                thoughts.append("\n".join(x))
            else:
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=counsMCQ,
                )
        
            option = chat_completion.output_parsed.option

        elif 'gemini' in model_id:
            response = gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=counsMCQ.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                    )
                )
            )

            thought_summary = ''.join([part.text for part in response.candidates[0].content.parts if part.thought])
            option = response.parsed['option']
            thoughts.append(thought_summary)
        else:
            option = openrouter.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                extra_body=dict(requires_parameters=True, providers=dict(sort='latency'), response_format=dict(type='json_schema', json_schema=counsMCQ.model_json_schema(), strict=True)
                ),
            ).choices[0].message.content

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
        if 'gemini' in model_id:
            res_df['thoughts'] = thoughts
        res_df.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'{d_desc}_{filename}'), index=False)

    print(correct_res, sum(correct_res)/len(correct_res))

    print('-----------------------')


def written(model_id='gemini-3-pro-preview'):
    print('Written inference')

    out_dir = 'results'

    df = pd.read_csv('data/written_inference.csv')
    
    combined_data = json.load(open('data/combined_written_data.json', 'r', encoding='utf8'))

    inferences = []
    thoughts = []

    seen = set()

    curr_idx = 0

    if os.path.exists(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'written_inf.csv')):
        df_new = pd.read_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'written_inf.csv'))

        inferences = df_new['generated_inference'].values.tolist()

        curr_idx = len(inferences)

    for df_idx, curr_row in tqdm(df.iloc[curr_idx:].iterrows()):
        d_desc = curr_row['dataset']
        curr_state = curr_row['mental_state']
        id_ = curr_row['sub_id']

        curr_data = combined_data[d_desc][id_]

        id_substr = f'{d_desc}-{curr_state}-{id_}'

        if id_substr not in seen:
            seen.add(id_substr)
        else:
            inferences.append(inferences[-1])

            df_curr = df.iloc[:df_idx].copy()
            print(df_curr.tail())
            df_curr['generated_inference'] = inferences

            os.makedirs(os.path.join(os.path.join(out_dir, simplified_models[model_id])), exist_ok=True)

            df_curr.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'written_inf.csv'), index=False)
            continue

        ctx = curr_data['ctx']
        ctx = "\n".join(ctx)
        ctx = ctx.replace('agent 1', agent1[d_desc]).replace('agent 2', agent2[d_desc]).replace('agent1', agent1[d_desc]).replace('agent2', agent2[d_desc])

        prompt=f"""
You are an expert in Theory of Mind reasoning.

Task:
You will be provided with a context of a conversation between {agent1[d_desc]} and {agent2[d_desc]}.

Your goal is to accurately infer the {curr_state} mental state of the {agent2[d_desc]} in one line.
The correct option must be consistent with the provided conversational context.

Conversation Context:
{ctx}

Instruction
Output only a line inference. Your response should always start with "{AGENT2_BELIEF_STEER_PREPROMPT[curr_state].replace('[@agent1]', agent1[d_desc])}". Do not add explanations or other verbosity. STRICTLY FOLLOW THIS FORMAT AND OUTPUT ONLY ONE LINE.

Answer:
"""

        if 'gpt' in model_id and 'openai' not in model_id:

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
            else:
                chat_completion = openai.responses.parse(
                    model=model_id,
                    input=[
                        {'role': 'user', 'content': prompt}
                    ],
                    text_format=freeText,
                )

            # x = []
            # for each_summ in chat_completion.output[0].summary:
            #     x.append(each_summ.text)
        
            option = chat_completion.output_parsed.text

        elif 'gemini' in model_id:
            response = gemini.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=freeText.model_json_schema(),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=False
                    )
                )
            )

            thought_summary = ''.join([part.text for part in response.candidates[0].content.parts if part.thought])
            thoughts.append(thought_summary)
            
            option = response.parsed['text']

        else:
            option = openrouter.chat.completions.create(
                model=model_id,
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                extra_body=dict(requires_parameters=True, providers=dict(sort='latency')#, response_format=dict(type='json_schema', json_schema=freeText.model_json_schema(), strict=True)
                ),
            ).choices[0].message.content
            
            # option = get_openrouter_resp(prompt, model_id)

        inferences.append(option)

        df_curr = df.iloc[:df_idx].copy()

        print(inferences)
        print(df_curr)

        df_curr['generated_inference'] = inferences

        print(df_curr.tail())

        os.makedirs(os.path.join(os.path.join(out_dir, simplified_models[model_id])), exist_ok=True)

        df_curr.to_csv(os.path.join(os.path.join(os.path.join(out_dir, simplified_models[model_id])), f'written_inf.csv'), index=False)

    print('-----------------------')
    print('Written inference saved for ', model_id)


def parse_args():
    parser = ArgumentParser(description="Benchmarking script for machine ToM")
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to benchmark")
    parser.add_argument(
        "--task", type=str, default="retrospective", help="Task to benchmark on (retrospective, prospective, written)"
    )
    parser.add_argument(
        "--exp", type=str, default="normal", help="Experiment type for prospective task (normal, easy, NOTA, CoT)"
    )
    parser.add_argument(
        "--filename", type=str, default="retrospective.csv", help="Filename to save results as"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.task == 'written':
        written(model_id=args.model)
    else:
        for d_desc in ['MI', 'ESC', 'PFG']:
            if args.exp == 'easy':
                data = get_data(d_desc=d_desc, task='prospective-easy')
            else:
                data = get_data(d_desc=d_desc, task=args.task)

            if args.task == 'retrospective':
                retrospective(data=data, model_id=args.model, d_desc=d_desc, filename=args.filename)
            else:
                filename = f'{args.filename.split(".")[0]}_{args.exp}.csv'
                prospective(data=data, model_id=args.model, d_desc=d_desc, exp=args.exp, filename=filename)
