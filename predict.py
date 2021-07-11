import torch
import json
from transformers import AutoTokenizer
from metrics import get_entity
import argparse

class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.checkpoint_path)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        with open(args.char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        with open(args.label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()

    def character2id(self, sentence, max_char_len):
        char_ids = []
        for word in sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            if len(word_char_ids) < max_char_len:
                word_char_ids += (max_char_len - len(word_char_ids))*[self.char_vocab['PAD']]
            char_ids.append(word_char_ids)
        return torch.tensor([char_ids])

    def preprocess(self, tokenizer, sentence, mask_padding_with_zero=True):
        input_ids = [tokenizer.cls_token_id]
        firstSWindices = [len(input_ids)]
        for w in sentence:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))

        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        return torch.tensor([input_ids]), torch.tensor([attention_mask]),torch.tensor([firstSWindices])

    def predict(self, sentence):
        sentence = sentence.split(' ')
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, sentence)
        inputs = {'input_ids': input_ids.to(self.device),
                  'attention_mask': attention_mask.to(self.device),
                  'first_subword': firstSWindices.to(self.device),
                  'char_ids':None
                  }
        if self.args.use_char:
            char_ids = self.character2id(sentence,self.args.max_char_len)
            input_ids['char_ids'] = char_ids

        with torch.no_grad():
            outputs = self.model(**inputs)
        input_tensor, cate_pred = outputs[0].max(dim=-1)
        label = get_entity(cate_pred, self.label_set)
        return self.get_result(sentence,label)
    def get_result(self,sentence, label):
        
        results = []
        for lb in label:
            entity = ' '.join(sentence[lb[1]:lb[2]+1])
            results.append([entity,lb[0]])
        return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--char_vocab_path', default='data/charindex.json', type=str)
    parser.add_argument('--label_set_path', default='data/ner_covid19/label_set.txt', type=str)
    parser.add_argument('--model_name_or_path', default='vinai/phobert-base', type=str)

    parser.add_argument('--max_char_len',default = 15, type=int)
    parser.add_argument('--use_char',default=False, type = bool)


    parser.add_argument('--checkpoint_path', default='results/checkpoint.pth')
    args, unk = parser.parse_known_args()

    sentence = '4. Số văn bản đề nghị của cơ sở: 02:2019/ĐKSX-H&X Ngày: 24/12/2019'
    
    p = Predictor(args)
    print(p.predict(sentence))
