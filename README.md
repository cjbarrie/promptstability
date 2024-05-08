# promptstability
Repo for paper analyzing stability of outcomes resulting from variations in language model prompt specification

- psa.py: initial PromptStabilityAnalysis class function (cosine-based new prompts), returning full analysis (KA vs. number of prompt repetitions KA vs. cosine similarity of prompt, poor performing prompts)
- psa_temp.py: PromptStabilityAnalysis class function with temperature (temperature-based new prompts), returning KA vs temperature ONLY
    - test_tweets.py: testing psa_temp.py on tweets data
    - test_manifestos.py: testing psa_temp.py on UK manifestos data
