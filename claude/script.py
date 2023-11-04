from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import sys
import json
import os
import fileinput


filename = "dummy-files/test.py"

def _extract_relevant_info(text):
    start_index = text.find('"""')
    if start_index != -1:
        extracted_text = text[start_index:]
        return extracted_text
    else:
        return None

def generate_docstring(file_str, fn_name, key):
    with open(file_str, 'r') as f:
        content = f.read()
    anthropic = Anthropic(
        api_key=key,
    )

    prompt_file = open("resources/prompt.txt", "r")
    prompt = prompt_file.read()
    prompt = prompt.replace("[fn_name]", fn_name)
    prompt = prompt.replace("[file]", content)

    # TODO: replace this default docstring template with one generated by Claude?
    # (so it can support any language, not just Python)
    template_file = open("resources/python_docstring_template.txt", "r")
    docstring_template = template_file.read()
    prompt = prompt.replace("[docstring_example]", docstring_template)

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=prompt,
    )
    return _extract_relevant_info(completion.completion)

def add_docstring(key):
    # diff text file all strings, parse the file to only fetch statements with additions
    # changed file names
    with open("diff.txt", '+rb') as f:
        print("Hello world", f)
        # intelligent regex 
        content = f.readlines()
        fns_with_docstring = dict()
        contains_docstring = False
        in_func = False
        for line in content:
            line = line.decode('utf-8')  # Decode the bytes to a string
            if line.startswith("+def "):
                in_func = True
                func_name = line.split('+def ')[1].split('(')[0]
                # regex to check if there exists a docstring
            if line.replace(' ', '') == '+\n' or line == '\n':
                if in_func and not contains_docstring:
                    fns_with_docstring[func_name] = generate_docstring(filename, func_name, key)
                in_func = False
                contains_docstring = False
                func_name = ""
            if '"""' in line:
                contains_docstring = True
    return fns_with_docstring


def merge_docstring(fns_without_docstring):
    with open(filename, '+rb') as f:
        content = f.readlines()
        fn_wo_doc = False
        current_docstring = ""
        docstring_placement = {}
        for i, line in enumerate(content):
            line = line.decode('utf-8')
            if any([fn_name in line for fn_name in fns_without_docstring.keys()]) and "def" in line:  # if a fn without docstring is defined on this line
                fn_wo_doc = True
                for fn_name in fns_without_docstring.keys():
                    if fn_name in line:
                        current_docstring = fns_without_docstring[fn_name]
                        break
            if ":" in line and fn_wo_doc:
                docstring_placement[i] = current_docstring
                current_docstring = ""
                fn_wo_doc = False
    
    print('docstring_placement', docstring_placement)

    # with fileinput.input(files=(filename,), inplace=True) as file:
    #     for line_num, line in enumerate(file, start=1):
    #         # Check if this line should have content added
    #         if line_num in fns_to_add:
    #             content_to_add = fns_to_add[line_num]
    #             print(content_to_add, end='')
    #         print(line, end='')

if __name__ == "__main__":
    key = sys.argv[1]
    docstring_dict = add_docstring(key)
    merge_docstring(docstring_dict)
    print(docstring_dict)
    