import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# seed GPT-3 with natural language type prompts:


seed = """Help Max achieve his goals.\n
To fix the car, I need a person, who is a mechanic.\n
To bury the treasure, I need a tool, which is a shovel.\n
To make the farmer happy, I need three animals, which are a cow, pig, and chicken.\n"""

# globals for parsing
vowels = {"a", "e", "i", "o", "u"}

goals = ["Cut down the tree.", "Put the King of the Jungle to sleep.", "Bake a cake.", "Electrocute the water and kill the eel."]
actions = ["tool", "person", "animal"]

def parse_goal(goal):
    # convert goal as command to a prompting fragment
    # assumes goal has a punctuation in final character
    converted_goal = "To " + goal[0].lower() + goal[1:-1] + ","
    return converted_goal


def expand_goal(current_goal, sampling="greedy"):
    # assumes input ends with a comma
    # return array of expansions (or tuples w/ score/class??)
    # sampling method determines number of sub-goals returned

    # pre-fixed options
    action_expansions = {"tool": "which", "person": "who", "animal": "which"}

    # extended subgoals with score tuples
    subgoals = []
    for action in action_expansions.keys():
        # handle grammar
        if action[0] in vowels:
            parsed_action = f' I need an {action}'
        else:
            parsed_action = f' I need a {action}'

        print("parsed action: ", action)

        prompt = current_goal + parsed_action
        response = openai.Completion.create(
            engine="davinci",
            prompt=seed + prompt,
            temperature=0.7,
            max_tokens=0,
            top_p=1,
            logprobs=5,
            frequency_penalty=0,
            presence_penalty=0,
            echo=True
        )

        score = np.sum(response["choices"][0].logprobs.token_logprobs[1:])
        subgoals.append((prompt + f', {action_expansions[action]}', score))

    sorted_subgoals = sorted(subgoals, key=lambda x: x[1], reverse=True)  # lowest log prob = best = first idx

    if sampling == "greedy":
        return [sorted_subgoals[0]]
    else:
        return sorted_subgoals


def complete_plan(current_goal, sampling="greedy", max_token_len=10, num_generations=10):
    # generates a plan to complete goal
    # ends on a "\n" (or max num tokens)
    # returned plan(s) depend on sampling scheme
    #     returns an array of tuples: [(completion, joint_prob)]

    response = openai.Completion.create(
        engine="davinci",
        prompt=seed + current_goal,
        temperature=0.7,
        max_tokens=max_token_len,
        top_p=1,
        logprobs=5,
        frequency_penalty=0,
        presence_penalty=0,
        n=num_generations,
        stop=["\n"]
    )

    generations = []
    for idx, sample in enumerate(response["choices"]):
        # handle stop token
        first_stop = np.where(np.array(response["choices"][idx].logprobs.tokens) == '\n')
        end_token_idx = first_stop[0][0] if first_stop is not None and len(first_stop[0]) != 0 else 1

        # extract joint prob of generation
        log_prob = np.sum(response["choices"][idx].logprobs.token_logprobs[
                          :end_token_idx])  # UPDATE: only sum prior to stop token

        # append to existing goal stream
        generations.append((current_goal + sample["text"], log_prob))

    sorted_completions = sorted(generations, key=lambda x: x[1], reverse=True)
    if sampling == "greedy":
        return [sorted_completions[0]]
    else:
        return sorted_completions

def get_problem_solutions(sort=False):
    sampling = "brainstorm"
    problem_solutions = {}
    for goal in goals:
        parsed_goal = parse_goal(goal)
        expansions = expand_goal(parsed_goal, sampling=sampling)
        all_completions = []
        for expansion, score in expansions:
            completions = complete_plan(expansion, sampling=sampling)
            all_completions.extend(completions)
        if sort:
            sorted_completions = sorted(all_completions, key=lambda x: x[1])
            problem_solutions[goal] = sorted_completions
        else:
            problem_solutions[goal] = all_completions

    print(problem_solutions)
    return problem_solutions

#get_problem_solutions()

# plot expand goal - always returns goal, I need a tool/animal/person, logprob - fixed
def plot_expand_goal():
    sampling = "brainstorm"
    problem_solutions = {}
    for goal in goals:
        print(goal)
        parsed_goal = parse_goal(goal)
        expansions = expand_goal(parsed_goal, sampling=sampling)
        print(expansions)

#plot_expand_goal()

# plot complete plan - parse, final word before the end
def plot_complete_plan(output_file=None):
    if output_file is None:
        problem_solutions = get_problem_solutions()
    else:
        f = open(output_file)
        problem_solutions = json.load(f)
    prob_goal = dict()
    for goal in goals:
        prob_goal[goal] = {a: dict() for a in actions}
        goal_generations = problem_solutions[goal]
        for generation in goal_generations:
            act = find_action(generation[0])
            gen = get_generated_vocab(generation[0])
            if gen not in prob_goal[goal][act]:
                prob_goal[goal][act][gen] = 0
            prob_goal[goal][act][gen] += np.exp(generation[1])

    norm_prob_goal = normalize_probs(prob_goal)

    for goal in goals:
        for action in actions:
            d = norm_prob_goal[goal][action]
            generations = list(d.keys())
            prob = [d[g] for g in generations]
            plot_data(generations, prob, goal, action)
        
def find_action(generation):
    return [a for a in actions if a in generation][0]

def get_generated_vocab(generation):
    return ' '.join(generation.split(',')[-1].split(' ')[4:])[:-1]

def normalize_probs(prob_dict):
    output = dict()
    for goal in prob_dict:
        output[goal] = dict()
        for action in prob_dict[goal]:
            output[goal][action] = dict()
            prob_sum = 0
            for gen in prob_dict[goal][action]:
                prob_sum += prob_dict[goal][action][gen]
            for gen in prob_dict[goal][action]:
                output[goal][action][gen] = prob_dict[goal][action][gen]/prob_sum
    return output

def plot_data(generations, prob, goal, action):
    df = pd.DataFrame({'generations': generations, 'probabilities': prob})
    fig = px.bar(df, x='generations', y='probabilities', title=f'goal: {goal} | action: {action}')
    goal = goal.replace(" ", "_")
    fig.write_image(f'plots/plot_{goal}_{action}.png')


#plot_complete_plan('data/nonsortedcompleteoutputs.json')
