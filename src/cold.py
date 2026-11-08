import pymupdf

def extract(file_path): 
    doc = pymupdf.open(file_path)

    full_text = ""

    for page_number, page in enumerate(doc):

        text = page.get_text()

        full_text += text + "\n"

    return full_text


if __name__ == "__main__":

    pdf_path = "/Users/user/Documents/3A/Cold chain RAG exp/data/Thermal_Energy_Storage_Technology_in_solar_Energy_.pdf"

    text = extract(pdf_path)
    #t = text_chunking(text)
    print(text[:5000])