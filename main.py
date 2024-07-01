import ebooklib
import nltk
import torch
import transformers
from bs4 import BeautifulSoup
from ebooklib import epub
from nltk import word_tokenize
from nltk.translate import meteor
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPTNeoXJapaneseForCausalLM,
    GPTNeoXJapaneseTokenizer,
)
import ollama

nltk.download("punkt")
nltk.download("wordnet")


def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    text = ""

    for item in book.get_items():
        if isinstance(item, ebooklib.epub.EpubHtml):
            text += item.get_content().decode("utf-8")

    soup = BeautifulSoup(text, "html.parser")
    text = "".join([p.text for p in soup.find_all("p")])
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(text)
    print("Success")
    return text


def score(ref, tl):
    with open(ref, "r", encoding="utf-8") as file:
        generated = file.read()
        generated = generated.replace("\n", " ")
    with open(tl, "r", encoding="utf-8") as file:
        reference = file.read()
        reference = reference.replace("\n", " ")
    print(round(meteor([word_tokenize(reference)], word_tokenize(generated)), 4))


def rakuten_pipeline(text):
    model_path = "Rakuten/RakutenAI-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device=model.device)
    tokens = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def aya101_pipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
    aya_model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def yamshadow_pipeline(text):
    messages = [{"role": "user", "content": f"Translate to English: {text}"}]

    tokenizer = AutoTokenizer.from_pretrained("automerger/YamshadowExperiment28-7B")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model="automerger/YamshadowExperiment28-7B",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"]


def dareties_pipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("yunconglong/DARE_TIES_13B")
    aya_model = AutoModelForCausalLM.from_pretrained("yunconglong/DARE_TIES_13B")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def llm_pipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v2.0")
    model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-13b-v2.0")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


def gpt_neo_pipeline(text):
    tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])


@torch.inference_mode
def mistral_pipeline(text):
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    peft_id = "hpprc/Mixtral-8x7B-Instruct-ja-en"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
    ).eval()

    model = PeftModel.from_pretrained(model=model, model_id=peft_id)

    messages = [
        # {"role": "user", "content": "Translate this English sentence into Japanese.\n" + input("En > ")},
        {
            "role": "user",
            "content": "Translate this Japanese sentence into English.\n"
            + input("Ja > "),
        },
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        num_beams=5,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0][len(inputs.input_ids[0]) :]
    out = tokenizer.decode(gen_ids, skip_special_tokens=True)
    out = out.split("\n")[
        0
    ]  # 生成しすぎることがあるので最初の一文だけ取り出すのがいいかも
    return out


def table_of_scores(texts: list, ref_path: str):
    with open(ref_path, "r", encoding="utf-8") as file:
        ref = file.read()
    scores = []
    for text in texts:
        for _ in texts:
            scores.append(round(meteor([word_tokenize(text)], word_tokenize(_)), 4))
        scores.append(round(meteor([word_tokenize(ref)], word_tokenize(text)), 4))

    print("Table : RakutenAI | Aya101 | Yamshadow | DARE_TIES | Reference")
    print(
        f"RakutenAI | {scores[0]} | {scores[1]} | {scores[2]} | {scores[3]} | {scores[4]}"
    )
    print(
        f"Aya101    | {scores[5]} | {scores[6]} | {scores[7]} | {scores[8]} | {scores[9]}"
    )
    print(
        f"Yamshadow | {scores[10]} | {scores[11]} | {scores[12]} | {scores[13]} | {scores[14]}"
    )
    print(
        f"DARE_TIES | {scores[15]} | {scores[16]} | {scores[17]} | {scores[18]} | {scores[19]}"
    )
    print(
        f"llm jp    | {scores[20]} | {scores[21]} | {scores[22]} | {scores[23]} | {scores[24]}"
    )
    print(
        f"GPT-Neo   | {scores[25]} | {scores[26]} | {scores[27]} | {scores[28]} | {scores[29]}"
    )
    print(
        f"Mistral   | {scores[30]} | {scores[31]} | {scores[32]} | {scores[33]} | {scores[34]}"
    )


def merger(
    texts: list,
):  # TODO : Make a function taking all the generated text and merging them into one clean text by studying the probability of each word.
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": f"Here are multiple translations of the same text, merge the translations into one coherent and well written text, here's the first translation : ```{texts[0]}``` , here's the second translation : ```{texts[1]}``` , here's the third translation : ```{texts[2]}``` , here's the fourth translation : ```{texts[3]}```",
            },
        ],
    )
    with open("merged.txt", "w", encoding="utf-8") as file:
        file.write(response["message"]["content"])


def main(text):
    texts = [
        rakuten_pipeline(text),
        aya101_pipeline(text),
        yamshadow_pipeline(text),
        dareties_pipeline(text),
        llm_pipeline(text),
        gpt_neo_pipeline(text),
        mistral_pipeline(text),
    ]
    table_of_scores(texts, "assets/ref/ref.txt")
    merger(texts)


rakuten_pipeline("こんにちは")
