import warnings
import subprocess

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint")

# Define the paths to the example scripts
scripts = [
    "00_tweets_example.py",
    "01_news_example.py",
    "02a_manifestos_example.py",
    "02b_manifestos_multi_example.py"
]

# Run each script
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error running {script}: {result.stderr}")
    print(f"Finished running {script}")

print("All scripts have been executed.")
