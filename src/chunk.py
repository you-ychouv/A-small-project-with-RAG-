# text input here is the output from extraction
def text_chunking(text, size, overlap):
    chunk = []

    begin = 0
    while begin < len(text):
        end = begin + size

        ch = text[begin:end]

        chunk.append(ch)

        begin = end - overlap

    return chunk

