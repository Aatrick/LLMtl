from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = ""

    for item in book.get_items():
        if isinstance(item, ebooklib.epub.EpubHtml):
            text += item.get_content().decode('utf-8')

    soup = BeautifulSoup(text, 'html.parser')
    text = ''.join([p.text for p in soup.find_all('p')])
    with open("output.txt", "w", encoding='utf-8') as file:
        file.write(text)
    print("Success")
    return text

extract_text_from_epub('assets/ref/Unnamed_Memory_I_.epub')

def generate_text(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    requests = [
        "南硫黄島原生自然環境保全地域は、自然",
        "The capybara is a giant cavy rodent",
    ]

    for req in requests:
        input_ids = tokenizer.encode(req, return_tensors="pt").to(device=model.device)
        tokens = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
        out = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print("INPUT:\n" + req)
        print("OUTPUT:\n" + out)
        print()
        print()

#generate_text("Rakuten/RakutenAI-7B")