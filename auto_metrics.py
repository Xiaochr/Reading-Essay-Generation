import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt2_finetune import GPT2Dataset
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')


def get_nll(content):
    # To calculage NLL, we adopt a gpt2 without fine-tuned, instead of our fine-tuned model
    device = torch.device("cuda")
    model_id = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id, pad_token='<|pad|>')
    model.resize_token_embeddings(len(tokenizer))

    dataset = GPT2Dataset(content, tokenizer, max_length=1024)
    dataloader = DataLoader(dataset, batch_size=2)

    nlls = []
    for batch_index, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
            neg_log_likelihood = outputs[0]
            print(neg_log_likelihood)

        nlls.append(neg_log_likelihood)

    avg_nlls = torch.stack(nlls).sum() / len(nlls)
    print("Avg NLL: {:.4f}".format(avg_nlls.item()))

    return nlls


def get_others(content):
    tokens = []
    for essay in content:
        sents = nltk.sent_tokenize(essay)
    
        temp_words = []
        for sent in sents:
            words = nltk.word_tokenize(sent)
            words = [word.lower() for word in words if word.isalpha()]
            temp_words.extend(words)

        tokens.append(temp_words)

    token_df = pd.DataFrame(np.array([" "] * len(tokens)).T, columns=["tokens"])
    for i in range(len(tokens)):
        token_df.iloc[i, 0] = tokens[i]

    # ttr
    token_df["len"] = token_df["tokens"].apply(lambda x: len(x))
    token_df["unique"] = token_df["tokens"].apply(lambda x: len(set(x)))
    token_df["ttr"] = token_df.apply(lambda x: x["unique"] / x["len"], axis=1)
    avg_ttr = token_df["ttr"].mean()
    print("Avg TTR: {:.4f}".format(avg_ttr))

    # repetition
    token_df["rep"] = token_df["tokens"].apply(lambda x: ngram_repetition(x))
    avg_rep = token_df["rep"].mean()
    print("Avg Repetition: {:.4f}".format(avg_rep))

    # keyword number
    with open("./topics/keywords_school_30.txt", "r") as f:
        word_list_1 = [x[:-1] for x in f.readlines()]

    with open("./topics/keywords_life_30.txt", "r") as f:
        word_list_2 = [x[:-1] for x in f.readlines()]

    with open("./topics/keywords_nature_30.txt", "r") as f:
        word_list_3 = [x[:-1] for x in f.readlines()]

    token_df["keyword_num_1"] = token_df["tokens"].apply(lambda x: cnt_keywords(x, word_list_1))
    token_df["keyword_num_2"] = token_df["tokens"].apply(lambda x: cnt_keywords(x, word_list_2))
    token_df["keyword_num_3"] = token_df["tokens"].apply(lambda x: cnt_keywords(x, word_list_3))

    avg_topic_1 = token_df["keyword_num_1"][:10].mean()
    avg_topic_2 = token_df["keyword_num_2"][10:20].mean()
    avg_topic_3 = token_df["keyword_num_3"][20:].mean()
    avg_key_num = (avg_topic_1 + avg_topic_2 + avg_topic_3) / 3
    print("Avg Keyword Number: {:.4f}".format(avg_key_num))

    # Word Mover Distance
    model = Word2Vec.load("./topics/w2v.model")

    dist_1, dist_2, dist_3 = 0, 0, 0
    for i in range(0, 10):
        temp = [x for x in tokens[i] if x in model.wv]
        dist_1 += model.wv.wmdistance(word_list_1, temp)
    
    for i in range(10, 20):
        temp = [x for x in tokens[i] if x in model.wv]
        dist_2 += model.wv.wmdistance(word_list_2, temp)
    
    for i in range(20, 30):
        temp = [x for x in tokens[i] if x in model.wv]
        dist_3 += model.wv.wmdistance(word_list_3, temp)

    avg_wmd = (dist_1 + dist_2 + dist_3) / 30
    print("Avg WMDistance: {:.4f}".format(avg_wmd))

    return avg_ttr, avg_rep, avg_key_num, avg_wmd



def ngram_repetition(text, n=4):
    cnt = 0
    for i in range(len(text) - n):
        temp = 1 if text[i + n] in text[i:i + n] else 0
        cnt += temp

    return cnt / (len(text) - n)


def cnt_keywords(text, word_list):
    cnt = 0
    for i in text:
        if i in word_list:
            cnt += 1

    return cnt



def calculate_metrics(file_path="./examples/true_essay_30.csv"):
    print(file_path)
    essay_df = pd.read_csv(file_path, index_col=0)
    essay_content = essay_df["essay_content"].values.tolist()

    get_nll(essay_content)
    get_others(essay_content)



if __name__ == "__main__":
    calculate_metrics("./examples/true_essay_30.csv")
    calculate_metrics("./examples/uncontrolled_examples_30.csv")
    calculate_metrics("./examples/origin_controlled_examples_30.csv")
    calculate_metrics("./examples/controlled_examples_30.csv")
    


