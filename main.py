from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="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",
)


file = open("sampleCode.txt", "r")
content = file.read()

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} Summarize all the functions as docstrings in {content}. Dont rewrite the existing code and function name{AI_PROMPT}",
)
print(completion.completion)


