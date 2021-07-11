from transformers import AutoTokenizer
from dataloader import MyDataSet
from trainer import Trainer
import argparse


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_dataset = MyDataSet(path=args.train_path, char_vocab_path=args.char_vocab_path,
                              tokenizer=tokenizer, label_set_path=args.label_set_path,
                              max_char_len=args.max_char_len,
                              max_seq_length=args.max_seq_length)
    dev_dataset = MyDataSet(path=args.dev_path, char_vocab_path=args.char_vocab_path,
                            tokenizer=tokenizer, label_set_path=args.label_set_path,
                            max_char_len=args.max_char_len,
                            max_seq_length=args.max_seq_length)

    test_dataset = MyDataSet(path=args.test_path, char_vocab_path=args.char_vocab_path,
                             tokenizer=tokenizer, label_set_path=args.label_set_path,
                             max_char_len=args.max_char_len,
                             max_seq_length=args.max_seq_length)

    trainer = Trainer(args=args, train_dataset=train_dataset,
                      dev_dataset=dev_dataset, test_dataset=test_dataset)
    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        print('Test Result:')
        trainer.eval("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--train_path',  type=str)
    parser.add_argument('--dev_path',  type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--char_vocab_path', type=str)
    parser.add_argument('--label_set_path', type=str)
    parser.add_argument('--max_char_len', default=10, type=int)
    parser.add_argument('--max_seq_length', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # model
    parser.add_argument('--use_char', action="store_true")
    parser.add_argument('--char_embedding_dim', default=100, type=int)
    parser.add_argument('--char_hidden_dim', default=200, type=int)
    parser.add_argument('--num_layer_bert', default=1, type=int)
    parser.add_argument('--char_vocab_size', default=108, type=int)
    parser.add_argument('--hidden_dim', default=728, type=int)
    parser.add_argument('--hidden_dim_ffw', default=400, type=int)
    parser.add_argument('--num_labels', default=12, type=int)
    parser.add_argument('--model_name_or_path', type=str)

    # train
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--max_grad_norm', default=1, type=int)
    parser.add_argument('--do_train',action="store_true")
    parser.add_argument('--do_eval',action="store_true")

    parser.add_argument('--save_folder', default='results', type=str)
    args, unk = parser.parse_known_args()

    train(args)