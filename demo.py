import os
import openai
from data_generation import generate_examples

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    for subject in ['dog', 'car', 'basketball']:
        print(f"Generating an example of a {subject}.")
        generate_examples(subject=subject, number_of_examples=1)

if __name__ == "__main__":
    main()