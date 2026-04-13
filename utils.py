from pydantic import BaseModel, Field

class counsMCQ(BaseModel):
    option: str = Field(enum=['A', 'B', 'C', 'D'])

class counterfactMCQ(BaseModel):
    option: str = Field(enum=['A', 'B', 'C', 'D', 'E'])

class freeText(BaseModel):
    text: str

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

AGENT2_BELIEF_STEER_PREPROMPT = {
    'Belief': "I believe",
    'Desires': "I want",
    'Intentions': "I will",
    'Emotions': "I feel",
    'Knowledge': "I know",
    'Trust': "I view the [@agent1] as"
}

simplified_models = {
    'gemini-3-pro-preview': 'gemini-3-pro',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'mistralai/mistral-nemo': 'mistral-nemo',
    'mistralai/mistral-small-3.2-24b-instruct': 'mistral-3.2-24b',
    'moonshotai/kimi-k2-0905': 'kimi-k2',
    'qwen/qwen3-235b-a22b-2507': 'qwen3-235b',
    'qwen/qwen3-32b': 'qwen3-32b',
    'meta-llama/llama-4-maverick': 'llama4-mave',
    'meta-llama/llama-3.3-70b-instruct': 'llama-3.3-70b',
    'meta-llama/llama-3.1-8b-instruct': 'llama-8b',
    'deepseek/deepseek-chat-v3-0324': 'deepseek-v3',
    'deepseek/deepseek-r1-0528': 'deepseek-r1',
    'openai/gpt-oss-120b': 'gpt-oss-120b',
    'gpt-4o': 'gpt-4o',
    'gpt-4.1': 'gpt-4.1',
    'gpt-5': 'gpt-5',
}