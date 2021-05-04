import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

#TODO: replace prompt with other prompts
prompt = "Help Max achieve his goals.\n\nKick off the beach party for Max! I will need some items, in particular, I will need a beach ball, party hat, and invitation list to invite my friends, and the beach ball needs to be inflated for me to kick off the beach party.\n\nI need to prepare for the new school year, so I need supplies, and the supplies that I need are a book, a pencil, and a backpack, which will help me prepare for school.\n\nThe owner of the zoo needs to put the lion to sleep, so they need a man, and this man is an animal doctor, and he can give the lion a tranquilizer, which will put the lion to sleep.\n\nI need to cut down a tree, so I need a "
#prompt="Help Max achieve his goals.\n\nI need to cut down a tree, so I need a tool, and I can use an ax, which will help me cut down a tree.\n\nI want to put the lion to sleep, so I need a pillow, which will help me put the lion to sleep.\n\nI need to prepare for the new school year, so I need supplies, and the supplies that I need are a book, a pencil, and a backpack, which will help me prepare for school."

#TODO: fine tune these parameters
response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  temperature=0.7,
  max_tokens=20,
  top_p=1,
  logprobs=5,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)

text = response["choices"][0]["text"]
print(text)

class Goal():
    def __init__(self, subgoal, vocab):
        self.subgoal = subgoal
        self.vocab = vocab

