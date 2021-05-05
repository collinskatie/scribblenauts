import os
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# seed GPT-3 with natural language type prompts:

seed = """Help Max achieve his goals.\n
To put the lion to sleep, I need a person, who is an zoologist.\n
To electrocute the water and kill the eel, I need a tool, which is a lightning rod.\n
To make the farmer happy, I need three animals, which are a cow, pig, and chicken.\n"""

# globals for parsing
vowels = {"a", "e", "i", "o", "u"}

goals = ["Cut down the tree.", "Bake a cake."]


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
    actions = ["tool", "animal", "person"]

    # extended subgoals with score tuples
    subgoals = []
    for action in actions:
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
        subgoals.append((prompt + ", which is", score))

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
        print("first stop: ", first_stop)
        end_token_idx = first_stop[0][0] if first_stop is not None else 1

        # extract joint prob of generation
        log_prob = np.sum(
            response["choices"][idx].logprobs.token_logprobs[:end_token_idx])  # UPDATE: only sum prior to stop token
        print(sample["text"], ' rel prob: ' + str(log_prob))

        # append to existing goal stream
        generations.append((current_goal + sample["text"], log_prob))

    sorted_completions = sorted(generations, key=lambda x: x[1], reverse=True)
    if sampling == "greedy":
        return [sorted_completions[0]]
    else:
        return sorted_completions

sampling = "brainstorm"
problem_solutions = {}
for goal in goals:
    parsed_goal = parse_goal(goal)
    expansions = expand_goal(parsed_goal, sampling=sampling)
    all_completions = []
    for expansion, score in expansions:
        completions = complete_plan(expansion, sampling=sampling)
        all_completions.extend(completions)
    sorted_completions = sorted(all_completions, key=lambda x: x[1])
    problem_solutions[goal] = sorted_completions

print(problem_solutions)



