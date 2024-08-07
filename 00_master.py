import warnings
import subprocess
import concurrent.futures

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint")

# Define the paths to the example scripts
scripts = [
    "00a_tweets_rd_example.py",
    "00b_tweets_pop_example.py",
    "01a_news_example.py",
    "01b_news_short_example.py",
    "02a_manifestos_example.py",
    "02b_manifestos_multi_example.py",
    "03a_stance_example.py",
    "03b_stance_long_example.py",
    "04a_mii_example.py",
    "04b_mii_long_example.py",
    "05a_synth_example.py",
    "05b_synth_short_example.py"
]

def run_script(script):
    print(f"Running {script}...")
    try:
        process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"{script} stdout: {output.strip()}")

        error = process.stderr.read()
        if error:
            print(f"{script} stderr: {error.strip()}")

        process.wait()  # Ensure the process has finished
        print(f"Finished running {script} with return code {process.returncode}")

    except subprocess.TimeoutExpired:
        print(f"Script {script} timed out.")
    except Exception as e:
        print(f"Error running {script}: {str(e)}")

# Use ThreadPoolExecutor to run scripts in parallel
max_workers = 2  # Number of scripts to run in parallel

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(run_script, script): script for script in scripts}
    for future in concurrent.futures.as_completed(futures):
        script = futures[future]
        try:
            future.result()
        except Exception as exc:
            print(f"{script} generated an exception: {exc}")

print("All scripts have been executed.")
