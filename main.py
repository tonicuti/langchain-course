import os

from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"Your OpenAI API Key is: {api_key}")
    
if __name__ == "__main__":
    main()