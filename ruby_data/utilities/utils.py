def first_digit_index(s: str) -> int:
    for i, e in enumerate(s):
        if e.isdigit():
            return i
    raise Exception("The filename is not compatible with the expected format.")



