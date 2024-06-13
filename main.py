import ebooklib
import nltk
import torch
import transformers
from bs4 import BeautifulSoup
from ebooklib import epub
from nltk import word_tokenize
from nltk.translate import meteor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')
nltk.download('wordnet')


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

def score(ref,tl):
    with open(ref, "r", encoding='utf-8') as file:
        generated = file.read()
        generated=generated.replace("\n"," ")
    with open(tl, "r", encoding='utf-8') as file:
        reference = file.read()
        reference=reference.replace("\n"," ")
    print(round(meteor([word_tokenize(reference)], word_tokenize(generated)), 4))

def RakutenPipeline(text):
    model_path = "Rakuten/RakutenAI-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    requests = [
        "南硫黄島原生自然環境保全地域は、自然",
        "The capybara is a giant cavy rodent",
        text,
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
        return tokenizer.decode(tokens[0], skip_special_tokens=True)

def Aya101Pipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
    aya_model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])

def YamShadowPipeline(text):
    messages = [{"role": "user", "content": f"Translate to English: {text}"}]

    tokenizer = AutoTokenizer.from_pretrained("automerger/YamshadowExperiment28-7B")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model="automerger/YamshadowExperiment28-7B",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return outputs[0]["generated_text"]

def DareTiesPipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("yunconglong/DARE_TIES_13B")
    aya_model = AutoModelForCausalLM.from_pretrained("yunconglong/DARE_TIES_13B")
    inputs = tokenizer.encode(f"Translate to English: {text}", return_tensors="pt")
    outputs = aya_model.generate(inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])

with open ("tl.txt", "w", encoding='utf-8') as file:
    tl=RakutenPipeline("草もまばらな大地が広がる中、青年は天にそびえる塔を馬上から見上げる。「これが魔女の棲む塔か」　目の前に建つ塔を仰ぐ彼には、気負いの欠片もない。　黒に近い茶色の髪。瞳は日の落ちた後の空と同じ、深い青だ。　身なりのよさと秀麗な容姿からは、生まれながらの気品がうかがわれる。だがそれだけでなく、鍛えられた体に漂う隙のなさは、まだ若い彼に戦線に立つ覇者の印象を与えていた。　そのまま馬を降りて塔へと踏みこみそうな彼を、後ろから弱々しい声が止める。「殿下、やっぱりやめましょうよ……」「うるさいぞ、ラザル。ここで怯んでどうする」　青年は呆れ顔で振り返る。殿下と呼ばれる彼は、この塔の東に広がる大国ファルサスの王太子、オスカーだ。従者である幼馴染一人だけを連れてここに来た彼は、平然と嘯く。「せっかく城を抜け出してきたのに。帰ったら意味がないだろうが。単なる観光か」「観光で魔女のところに来る人間なんかおりません！」　──魔女。　それは大陸に五人しかいない、絶大な力によって異質とみなされる女たちだ。『閉ざされた森の魔女』『水の魔女』『呼ばれぬ魔女』『沈黙の魔女』『青き月の魔女』　この五つが彼女たちの通り名だ。魔女は気まぐれに現れ、その絶大な魔力を以て災いを呼び、消えうせる。数百年もの長きにわたる畏れと災厄の象徴だ。　中でも最も強大な力を持つとされる魔女が、『青き月の魔女』で、彼女はどこの国にも属さぬ荒野に青い塔を建て、その最上階に棲んでいる。この塔を登りきることのできた達成者は、彼女がその望みを叶えると言われているが、挑戦者が皆、塔から帰らないことが広まると、塔に近づく者も次第にいなくなっていった。　そんな塔を二人が訪ねてきたのは、目的があってのことだ。　ラザルと呼ばれた青年は、年若い主君に訴える。「やっぱり危ないですって。魔女に呪いを増やされたらどうするんですか！」「それはその時だ。もう他に手がかりもないだろう」「まだ何か他に手段がありますから……探せばきっと……」　すがりつくような言葉を聞きながらオスカーは馬を降りる。彼は鞍につけていた長剣を取ると腰の剣帯につけなおした。「他に手段と言われても。十五年も何も見つからなかっただろうが。──まず『青き月の魔女』に会って呪いを解く方法を聞く。駄目だったらこのまま呪いをかけた張本人の『沈黙の魔女』のところに行って呪いを解かせる。完璧じゃないか」「全然完璧じゃないです」　ラザルは泣きながらようやく馬を降りた。ひょろっとした細い体は、どう見ても戦闘向きではない。武器も持っていないのは、とりあえず慌てて出立したせいだ。彼は城を抜け出した時もそうだったように、小走りに主君を追いかける。「殿下のお気持ちは分かります……。ですが十五年もの間、誰も魔女たちに接触してこなかったのは危険が大きすぎるからです！　第一、『沈黙の魔女』は見つからないし、『青き月の魔女』に至ってはこの塔を登りきれた人間が誰もいなかったじゃないですか！」「確かに歩いて登るには高いな」　塔の壁は、青みがかった水晶のような材質だ。それが継ぎ目もなく空高くまで伸びている。　オスカーはそのずっと先、よく見えない先端を仰いだ。「まぁ何とかなるだろ」「何ともなりませんよ！　罠がいっぱいらしいですよ！　貴方に何かあったら、私はどんな顔して城に帰ればいいんですか」「沈痛な顔して帰れ」　軽く肩を竦めると、オスカーは無造作に歩き出す。「待ってください。私も行きますって」　それを見たラザルが慌てて、二人分の馬を木に繫いで後を追った。　事の始まりは十五年前のことだ。ある晩、城の一室に、魔女の宣告が響いた。『お前はもう子を生すことができない。そこにいるお前の息子もだ。お前たちの血は女の腹を食い破るだろう。ファルサス王家はお前たちを以て絶えるのだ！』　そんな呪いの言葉を、オスカー自身覚えているわけではない。彼の記憶に残っているのは、月を背にした魔女の影と、自分を抱きしめる父親の震える腕だけだ。「子を生すことができない」と言われても、当時五歳の彼にはその重大さが分からなかった。蒼白な父の顔に、ただぼんやりと何かよくないことが起きたのだ、と思っただけだ。　彼は王の唯一の子だった。王家の存亡に直結する問題は、ごく一部の者を除いて伏せられ、解呪の方法を探すために何人もの優秀な魔法士や学者が時を費やした。　一方、オスカー自身は利発で豪胆な少年となって武と学を修めた。その優秀さと整った容貌は、呪いのことを知らぬ周囲に期待を持たせるには充分なもので、国内ではもっぱら「将来は歴史に名を残す王になるだろう」と囁かれている。　だが、呪いの問題が解決しなければ、残るものは悪名だけだ。　十歳を過ぎ、呪いの意味を理解できるようになった頃から、オスカーは自分でも解呪の方法を探したが、いくら文献を調べても、また剣の腕を磨いて手がかりがあると思しき遺跡を訪れても、呪いを解くための糸口さえも得ることはできなかった。　──そして、あの夜から十五年が過ぎた。　近い将来王となるべき彼は、国境を越えた西、魔女の棲むという青い塔の前に立っている。「じゃあ行くか」「そんな無造作に扉を！　もっと慎重に開けてください！」　ラザルの悲鳴を聞きながら、オスカーは両開きの扉を押し開けて中に踏み入った。　見回すとそこは、広い円形の広間だ。中央部分は吹き抜けになっていて、右手に上へと登るための通路が見える。階段ではなく緩やかな坂になっている通路は、そのまま壁に沿って螺旋状に上階へと伸びているようだ。他に人の気配もない塔内部を、オスカーは見上げた。「大体記録通りか。入り口部分は」")
    file.write(tl)