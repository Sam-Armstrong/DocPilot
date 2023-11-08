def add_docstring(key):
    # diff text file all strings, parse the file to only fetch statements with additions
    # changed file names
    with open("diff.txt", "+rb") as f:
        # intelligent regex
        content = f.readlines()
        file_fns_without_docstring = {}
        filename = " "
        contains_docstring = False
        in_func = False
        for i, line in enumerate(content):
            line = line.decode("utf-8")  # Decode the bytes to a string
            if line.startswith("+++"):
                start_index = line.find("/")
                if start_index != -1:
                    filename = line[start_index + 1 :].rstrip("\n")
                    fns_without_docstring = {}
            if line.replace(" ", "").startswith("+def"):
                in_func = True
                func_name = line.replace(" ", "").split("+def")[1].split("(")[0]
                # regex to check if there exists a docstring
            if (
                line.replace(" ", "") == "+\n"
                or line.replace(" ", "") == "\n"
                or i == len(content) - 1
            ):
                if in_func and not contains_docstring:
                    fns_without_docstring[func_name] = generate_docstring(
                        filename, func_name, key
                    )
                    if filename not in file_fns_without_docstring:
                        file_fns_without_docstring[filename] = {}
                    file_fns_without_docstring[filename] = fns_without_docstring
                in_func = False
                contains_docstring = False
                func_name = ""
            if '"""' in line:
                contains_docstring = True
    return file_fns_without_docstring
