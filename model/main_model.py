from torch import nn
from model.layer import WordRep, FeedforwardLayer, BiaffineLayer
from transformers import AutoConfig


class BiaffineNER(nn.Module):
    def __init__(self, args):
        super(BiaffineNER, self).__init__()
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.lstm_input_size = args.num_layer_bert * config.hidden_size
        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim

        self.word_rep = WordRep(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True)
        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)



    def forward(self, input_ids=None, char_ids=None,  first_subword=None, attention_mask=None):

        x = self.word_rep(input_ids=input_ids, attention_mask=attention_mask,
                                      first_subword=first_subword,
                                      char_ids=char_ids)
        x, _ = self.bilstm(x)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        score = self.biaffine(start, end)
        return score
