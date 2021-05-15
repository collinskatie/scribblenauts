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

# seed = """Help Max achieve his goals.\n
# To fix the lamp, I need a person, who is an electrician.\n
# To bury the treasure, I need a tool, which is a shovel.\n
# To make the farmer happy, I need three animals, which are a cow, pig, and chicken.\n"""

# seed = """Help Max achieve his goals.\n
# To fix the lamp, I need an electrician.\n
# To bury the treasure, I need a shovel.\n
# To make the farmer happy, I need three animals, which are a cow, pig, and chicken.\n"""

seed = """Help Max achieve his goals.\n
To fix the lamp, I need person, who is an electrician.
Kick off the beach party for Max! I will need some items, in particular, I will need a beach ball, party hat, and invitation list to invite my friends, and the beach ball needs to be inflated.\n
I need to prepare for the new school year, so I need supplies, and the supplies that I need are a book, a pencil, and a backpack.\n"""

# inspired by people's responses
# seed = """Help Max achieve his goals.\n
# Max wants to host a birthday party for a friend! To bake a cake, I need an EZ bake oven and the sun.\n
# Tourists on top of a building can't see the view - clouds are in the way! To help the tourists see the view, I need to huff and puff until i blow the clouds away.\n
# To electrocute the water to destroy the sea creature, I need to get a boat that is made out of something that will protect me from the current and then plug in a toaster on board and throw it into the water, to stand on land and throw in a toaster plugged in from the house.\n
# """


# globals for parsing
vowels = {"a", "e", "i", "o", "u"}

#goals = ["Cut down the tree.", "Put the King of the Jungle to sleep.", "Bake a cake.", "Electrocute the water and kill the eel."]
actions = ["tool", "person", "animal"]

# from Scribbelnauts
goals = ["Cut down the tree.", "Put the King of the Jungle to sleep.", "Bake a cake.", "Electrocute the water and destroy the sea creature.",
        "Turn the runt of the litter into an award-winning pig!", "Conceal something in my cake to help my friend burrow through the prison walls!",
        "Make me look unique so I can draw in a crowd for my act!"]


def parse_goal(goal):
    # convert goal as command to a prompting fragment
    # assumes goal has a punctuation in final character
    converted_goal = "To " + goal[0].lower() + goal[1:-1] + ","
    return converted_goal


def expand_goal(current_goal, sampling="greedy"):
    # assumes input ends with a comma
    # return array of expansions (or tuples w/ score/class??)
    # sampling method determines number of sub-goals returned

    current_goal += " I need"

    # pre-fixed options
    action_expansions = {"tool": "which", "person": "who", "animal": "which"}
    # how many items to request -- currently, either single or several
    action_count = ["single", "several"]

    # extended subgoals with score tuples
    subgoals = []
    for action in action_expansions.keys():
        for count in action_count:
            # handle grammar
            if count == "several":
                # handle special punctuation
                #                 if action == "person":
                #                     parsed_action = f' {count} people'
                #                 else: parsed_action = f' {count} {action}s'
                if action == "person":
                    parsed_action = f' people'
                else:
                    parsed_action = f' {action}s'
            else:
                if action[0] in vowels:
                    parsed_action = f' an {action}'
                else:
                    parsed_action = f' a {action}'

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

            score = np.mean(response["choices"][0].logprobs.token_logprobs[1:])
            subgoals.append((prompt + f', {action_expansions[action]}', score))

    sorted_subgoals = sorted(subgoals, key=lambda x: x[1], reverse=True)  # lowest log prob = best = first idx
    print("sorted subgoals: ", sorted_subgoals)

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
        log_prob = np.mean(response["choices"][idx].logprobs.token_logprobs[
                          :end_token_idx])  # UPDATE: only sum prior to stop token

        # append to existing goal stream
        generations.append((current_goal + sample["text"], log_prob))

    sorted_completions = sorted(generations, key=lambda x: x[1], reverse=True)
    if sampling == "greedy":
        return [sorted_completions[0]]
    else:
        return sorted_completions

sampling = "brainstorm"
use_flat_model = True # toggle to flip b/w using flat vs. hierarchal/constrained model
max_token_len= 30
problem_solutions = {}
for goal in goals:
    parsed_goal = parse_goal(goal)
    if use_flat_model:
        parsed_goal += " I need"
        all_completions = []
        completions = complete_plan(parsed_goal, sampling=sampling, max_token_len=max_token_len)
        all_completions.extend(completions)
    else:
        # use hierarchal/constrained sampling
        expansions = expand_goal(parsed_goal, sampling=sampling)
        all_completions = []
        for expansion, score in expansions:
            completions = complete_plan(expansion, sampling=sampling, max_token_len=max_token_len)
            all_completions.extend(completions)
    sorted_completions = sorted(all_completions, key=lambda x: x[1], reverse=True)
    problem_solutions[goal] = sorted_completions

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
    if output_file:
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
