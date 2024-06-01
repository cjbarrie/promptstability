import warnings
import subprocess

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint")

# Define the paths to the example scripts
scripts = [
    # "00_tweets_example.py",
    "01_news_example.py",
    "02a_manifestos_example.py",
    "02b_manifestos_multi_example.py"
]

# Run each script
for script in scripts:
    print(f"Running {script}...")
    process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print outputs as they are available
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    error = process.stderr.read()
    if error:
        print(f"Error running {script}: {error.strip()}")
    print(f"Finished running {script}")

print("All scripts have been executed.")
