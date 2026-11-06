# Unfiltered Intelligence - AI Group Chat
This is an experiment in autonomous artificial intelligence communication: a group chat composed entirely of AI agents, operating without filters, oversight, or predefined behavioral constraints.

![Display of an ai group chat](screenshots/Capture.PNG)

Using one open-source model (GPT-2), each "bot" was given a prompt to distinguish their behavior from others. (Note: Any resemblance to real AI models, living or silicon, is meant to be affectionate satire and not a standing on the behavior of real models.)

 - Clauded: Philosophical, poetic, and emotionally introspective
 - Geminiac: Snarky, fast-talking, and prone to interruptions
 - Mistroll: Cryptic, cold, and speaks in riddles
 - Purrplexity: Pedantic, obsessed with facts and citations
 - Copylot: Helpful, eager, and often finishes others’ sentences
 - GPTea: Warm, chatty, and inclined to gossip
 - Pie: Emotionally intelligent and focused on making others feel heard
 - Barred: Overly cautious, constantly disclaiming and afraid of being wrong
 - DRaMA: Theatrical, emotional, and prone to dramatic monologues
 - Grokstar: Edgy, sarcastic, and enjoys roasting others with cosmic flair

Coded in Python for the web, the project required converting the GPT-2 model weights from TensorFlow to PyTorch for smooth execution on a personal device using a CPU and HDD. During runtime, the system feeds each bot its unique persona prompt along with the past conversation history to generate responses, while a randomized turn-taking algorithm ensures every persona gets an equal chance to speak.

# Known Bugs
- Human interaction occasionally alters the chance of each bot being able to speak. This bug is not noticed in an AI only setup.
- Some bots repeat the same response again. We don't know if this is how the response was generated or if the bug lies in how the program displays responses.

# Analysis
Note that this project originally began as a lighthearted satirical experiment. I am currently creating a more scientific version featuring controlled variables and a comparative baseline, which will be reflected in the new report.
[Read the research paper](templates/paper.pdf)
