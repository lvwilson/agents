import re

def find_replace(source: str, command: str) -> str:
    # Define the regular expression pattern to extract SEARCH and REPLACE sections
    pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    
    # Use re.search to find the sections
    match = re.search(pattern, command, re.DOTALL)
    
    if not match:
        raise ValueError("Command format is incorrect or missing SEARCH and REPLACE sections.")
    
    # Extract SEARCH and REPLACE parts
    search_text = match.group(1).strip()
    replace_text = match.group(2).strip()
    
    # Perform the replacement
    result = source.replace(search_text, replace_text)
    
    return result

# # Example usage
# source_code = """
# def greeting():
#     print("Hello")
# """

# command = """
# <<<<<<< SEARCH
# def greeting():
#     print("Hello")
# =======
# def greeting():
#     print("Goodbye")
# >>>>>>> REPLACE
# """

# # Applying the find and replace operation
# result = find_replace(source_code, command)
# print("Modified Source:\n", result)
