import os
import json
import argparse
import tempfile
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


BPE_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ETBERT_SPECIAL_TOKENS = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
SPECIAL_TOKENS = BPE_SPECIAL_TOKENS
TOKENIZER_MODEL_BPE = "bpe"
TOKENIZER_MODEL_ETBERT_WORDPIECE = "etbert_wordpiece"
TOKENIZER_MODEL_ETBERT_WORDPIECE_TRAINER = "etbert_wordpiece_trainer"
ETBERT_WORDPIECE_VOCAB_SIZE = len(ETBERT_SPECIAL_TOKENS) + 65536


def normalize_bpe_tokenizer_model(tokenizer_model):
    tokenizer_model = (tokenizer_model or TOKENIZER_MODEL_ETBERT_WORDPIECE).lower()

    if tokenizer_model in {"etbert", "wordpiece", "etbert_wordpiece"}:
        return TOKENIZER_MODEL_ETBERT_WORDPIECE

    if tokenizer_model in {
        "etbert_train",
        "wordpiece_train",
        "etbert_wordpiece_train",
        "etbert_wordpiece_trainer",
    }:
        return TOKENIZER_MODEL_ETBERT_WORDPIECE_TRAINER

    if tokenizer_model in {"bpe", "byte_bpe"}:
        return TOKENIZER_MODEL_BPE

    raise ValueError(f"未知 BPE tokenizer model: {tokenizer_model}")


def iter_jsonl_files(input_path):
    """
    支持输入单个 jsonl 文件，也支持输入 jsonl_parts 目录。
    """
    if os.path.isfile(input_path):
        yield input_path
        return

    jsonl_files = []

    for root, dirs, files in os.walk(input_path):
        dirs.sort()
        files.sort()

        for filename in files:
            if filename.lower().endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, filename))

    jsonl_files.sort()

    for path in jsonl_files:
        yield path


def tokens_to_hex_string(tokens, lowercase=True):
    """
    将原来的固定 hex token 重新拼成连续 hex 字符串。

    例如：
    ["4500", "0034", "abcd", "4000"]
        ↓
    "45000034abcd4000"

    这样 BPE 才能在连续字节/hex 片段上学习合并规则。
    """
    if lowercase:
        tokens = [str(t).lower() for t in tokens]
    else:
        tokens = [str(t) for t in tokens]

    return "".join(tokens)


def tokens_to_etbert_bigram_tokens(tokens, lowercase=True):
    """
    参照 ET-BERT：把连续 hex 字符串切成 byte，再生成滑动 byte-bigram。

    例如：
    ["4500", "0034", "abcd"]
        -> bytes: 45 00 00 34 ab cd
        -> tokens: 4500 0000 0034 34ab abcd

    ET-BERT 的词表主体就是 0000-ffff 这些 2-byte hex token。
    """
    hex_string = tokens_to_hex_string(tokens, lowercase=lowercase)

    if len(hex_string) % 2 == 1:
        hex_string = hex_string[:-1]

    byte_tokens = [
        hex_string[i:i + 2]
        for i in range(0, len(hex_string), 2)
        if len(hex_string[i:i + 2]) == 2
    ]

    return [
        byte_tokens[i] + byte_tokens[i + 1]
        for i in range(len(byte_tokens) - 1)
    ]


def tokens_to_etbert_bigram_text(tokens, lowercase=True):
    return " ".join(tokens_to_etbert_bigram_tokens(tokens, lowercase=lowercase))


def tokens_to_training_text(tokens, lowercase=True, tokenizer_model=TOKENIZER_MODEL_ETBERT_WORDPIECE):
    tokenizer_model = normalize_bpe_tokenizer_model(tokenizer_model)

    if tokenizer_model in {
        TOKENIZER_MODEL_ETBERT_WORDPIECE,
        TOKENIZER_MODEL_ETBERT_WORDPIECE_TRAINER,
    }:
        pieces = tokens_to_etbert_bigram_tokens(tokens, lowercase=lowercase)
        return " ".join(pieces), len(pieces)

    text = tokens_to_hex_string(tokens, lowercase=lowercase)
    return text, len(tokens)


def build_bpe_training_corpus(
    input_path,
    corpus_path,
    lowercase=True,
    unit="burst",
    tokenizer_model=TOKENIZER_MODEL_ETBERT_WORDPIECE,
):
    """
    从 flow_burst.jsonl 构建 BPE 训练语料。

    unit = burst:
        每个 burst 作为一行训练语料。

    unit = flow:
        每个 flow 的所有 burst 拼接成一行训练语料。
    """
    tokenizer_model = normalize_bpe_tokenizer_model(tokenizer_model)
    jsonl_files = list(iter_jsonl_files(input_path))

    if len(jsonl_files) == 0:
        raise FileNotFoundError(f"没有找到 jsonl 文件: {input_path}")

    total_lines = 0
    valid_samples = 0
    written_corpus_lines = 0
    total_bursts = 0
    total_hex_tokens = 0
    total_corpus_tokens = 0

    print(f"[INFO] jsonl files: {len(jsonl_files)}")
    print(f"[INFO] tokenizer model: {tokenizer_model}")
    print(f"[INFO] building temporary tokenizer corpus: {corpus_path}")

    with open(corpus_path, "w", encoding="utf-8") as writer:
        for jsonl_path in jsonl_files:
            print(f"[READ] {jsonl_path}")

            with open(jsonl_path, "r", encoding="utf-8") as reader:
                for line in reader:
                    total_lines += 1
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        sample = json.loads(line)
                    except Exception:
                        continue

                    bursts = sample.get("bursts", [])

                    if not isinstance(bursts, list) or len(bursts) == 0:
                        continue

                    valid_samples += 1

                    if unit == "burst":
                        for burst in bursts:
                            tokens = burst.get("tokens", [])

                            if not tokens:
                                continue

                            text, corpus_token_count = tokens_to_training_text(
                                tokens,
                                lowercase=lowercase,
                                tokenizer_model=tokenizer_model,
                            )

                            if text:
                                writer.write(text + "\n")
                                written_corpus_lines += 1
                                total_bursts += 1
                                total_hex_tokens += len(tokens)
                                total_corpus_tokens += corpus_token_count

                    elif unit == "flow":
                        flow_text_parts = []
                        flow_corpus_tokens = 0

                        for burst in bursts:
                            tokens = burst.get("tokens", [])

                            if not tokens:
                                continue

                            text, corpus_token_count = tokens_to_training_text(
                                tokens,
                                lowercase=lowercase,
                                tokenizer_model=tokenizer_model,
                            )

                            if text:
                                flow_text_parts.append(text)
                                total_bursts += 1
                                total_hex_tokens += len(tokens)
                                flow_corpus_tokens += corpus_token_count

                        if flow_text_parts:
                            # 用空格隔开不同 burst，避免 tokenizer 跨 burst 合并
                            writer.write(" ".join(flow_text_parts) + "\n")
                            written_corpus_lines += 1
                            total_corpus_tokens += flow_corpus_tokens

                    else:
                        raise ValueError("unit 只能是 burst 或 flow")

    stats = {
        "total_json_lines": total_lines,
        "valid_samples": valid_samples,
        "written_corpus_lines": written_corpus_lines,
        "total_bursts": total_bursts,
        "total_hex_tokens": total_hex_tokens,
        "total_corpus_tokens": total_corpus_tokens,
        "tokenizer_model": tokenizer_model,
    }

    return stats


def write_vocab_txt(tokenizer, vocab_txt_path):
    vocab = tokenizer.get_vocab()
    id_to_token = sorted(vocab.items(), key=lambda x: x[1])

    with open(vocab_txt_path, "w", encoding="utf-8") as writer:
        for token, idx in id_to_token:
            writer.write(token + "\n")


def build_etbert_wordpiece_vocab():
    vocab = {}

    for token in ETBERT_SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    for value in range(65536):
        token = f"{value:04x}"
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def train_etbert_wordpiece_tokenizer(
    corpus_path,
    output_dir,
    vocab_size=ETBERT_WORDPIECE_VOCAB_SIZE,
    min_frequency=2,
):
    """
    生成 ET-BERT 风格 WordPiece tokenizer。

    这里不做 BPE merge，而是固定使用 0000-ffff 的 2-byte bigram 词表。
    corpus_path 仍然会在前一步生成，用于统计和检查训练语料格式。
    """
    os.makedirs(output_dir, exist_ok=True)

    if vocab_size != ETBERT_WORDPIECE_VOCAB_SIZE:
        print(
            "[WARN] ET-BERT WordPiece 使用固定 0000-ffff 词表，"
            f"请求 vocab_size={vocab_size}，实际使用 {ETBERT_WORDPIECE_VOCAB_SIZE}"
        )

    print("[INFO] using ET-BERT WordPiece tokenizer")
    print(f"[INFO] effective vocab size: {ETBERT_WORDPIECE_VOCAB_SIZE}")
    print(f"[INFO] min_frequency is recorded for comparison only: {min_frequency}")

    tokenizer = Tokenizer(
        models.WordPiece(
            vocab=build_etbert_wordpiece_vocab(),
            unk_token="[UNK]",
            max_input_chars_per_word=4,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    tokenizer_json_path = os.path.join(output_dir, "bpe_tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    vocab_txt_path = os.path.join(output_dir, "bpe_vocab.txt")
    write_vocab_txt(tokenizer, vocab_txt_path)

    print("\n[DONE]")
    print(f"tokenizer json: {tokenizer_json_path}")
    print(f"vocab txt:      {vocab_txt_path}")

    return tokenizer_json_path, vocab_txt_path


def train_etbert_wordpiece_trainer_tokenizer(
    corpus_path,
    output_dir,
    vocab_size=ETBERT_WORDPIECE_VOCAB_SIZE,
    min_frequency=2,
):
    """
    调用 WordPieceTrainer 的 ET-BERT 风格分支。

    这个分支会从 byte-bigram 语料中训练词表；相比固定 0000-ffff 全集，
    它更接近 ET-BERT 代码里的 tokenizer.train(...) 调用。
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] using ET-BERT WordPieceTrainer tokenizer")
    print(f"[INFO] requested vocab size: {vocab_size}")
    print(f"[INFO] min_frequency: {min_frequency}")

    tokenizer = Tokenizer(
        models.WordPiece(
            vocab={},
            unk_token="[UNK]",
            max_input_chars_per_word=4,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=ETBERT_SPECIAL_TOKENS,
        show_progress=True,
    )

    # ET-BERT 原代码把同一个语料文件传了两遍；这里保留这个行为。
    tokenizer.train([corpus_path, corpus_path], trainer=trainer)

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    tokenizer_json_path = os.path.join(output_dir, "bpe_tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    try:
        tokenizer.model.save(output_dir, "wordpiece")
    except Exception:
        pass

    vocab_txt_path = os.path.join(output_dir, "bpe_vocab.txt")
    write_vocab_txt(tokenizer, vocab_txt_path)

    print("\n[DONE]")
    print(f"tokenizer json: {tokenizer_json_path}")
    print(f"vocab txt:      {vocab_txt_path}")
    print(f"wordpiece vocab:{os.path.join(output_dir, 'wordpiece-vocab.txt')}")

    return tokenizer_json_path, vocab_txt_path


def train_bpe_tokenizer(
    corpus_path,
    output_dir,
    vocab_size=30000,
    min_frequency=2,
    tokenizer_model=TOKENIZER_MODEL_ETBERT_WORDPIECE,
):
    """
    训练 BPE tokenizer，并保存：
    - bpe_tokenizer.json
    - bpe_vocab.json
    - bpe_merges.txt
    - bpe_vocab.txt
    """

    tokenizer_model = normalize_bpe_tokenizer_model(tokenizer_model)

    if tokenizer_model == TOKENIZER_MODEL_ETBERT_WORDPIECE:
        return train_etbert_wordpiece_tokenizer(
            corpus_path=corpus_path,
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    if tokenizer_model == TOKENIZER_MODEL_ETBERT_WORDPIECE_TRAINER:
        return train_etbert_wordpiece_trainer_tokenizer(
            corpus_path=corpus_path,
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="[UNK]",
            fuse_unk=False,
        )
    )

    # 这里使用 Whitespace：
    # 如果 unit=burst，每一行没有空格，整个 burst hex 串作为一个序列训练；
    # 如果 unit=flow，不同 burst 用空格隔开，BPE 不会跨 burst 合并。
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=BPE_SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train([corpus_path], trainer=trainer)

    # 加上 BERT 风格的 [CLS] 和 [SEP]
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    tokenizer.decoder = decoders.BPEDecoder()

    tokenizer_json_path = os.path.join(output_dir, "bpe_tokenizer.json")
    tokenizer.save(tokenizer_json_path)

    # 保存 HuggingFace tokenizers 标准格式 vocab.json + merges.txt
    tokenizer.model.save(output_dir, "bpe")

    # 再额外保存一份 UER 风格 vocab.txt，一行一个 token，按 id 排序
    vocab_txt_path = os.path.join(output_dir, "bpe_vocab.txt")
    write_vocab_txt(tokenizer, vocab_txt_path)

    print("\n[DONE]")
    print(f"tokenizer json: {tokenizer_json_path}")
    print(f"vocab txt:      {vocab_txt_path}")
    print(f"vocab json:     {os.path.join(output_dir, 'bpe-vocab.json')}")
    print(f"merges txt:     {os.path.join(output_dir, 'bpe-merges.txt')}")

    return tokenizer_json_path, vocab_txt_path


def test_tokenizer(tokenizer_json_path, test_text=None):
    """
    简单测试 tokenizer 是否可用。
    """
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    tokenizer_config = json.loads(tokenizer.to_str())
    model_type = tokenizer_config.get("model", {}).get("type", "").lower()

    if test_text is None:
        test_text = "45000034abcd40001603030020"

    if model_type == "wordpiece" and " " not in test_text:
        test_text = tokens_to_etbert_bigram_text([test_text])

    enc = tokenizer.encode(test_text)

    print("\n[TEST]")
    print(f"input:  {test_text}")
    print(f"tokens: {enc.tokens}")
    print(f"ids:    {enc.ids}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/jsonl_parts",
        help="输入 flow_burst.jsonl 文件，或者 jsonl_parts 目录",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="C:/HierarchicalBurst-FlowBERT/datasets/pre_train_output_jsonl/bpe_tokenizer",
        help="输出目录，例如 models/bpe_tokenizer",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=65541,
        help="BPE 词表大小，包含特殊符号",
    )

    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="BPE merge 最小频率",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="burst",
        choices=["burst", "flow"],
        help="BPE 训练单位：burst 表示每个 burst 一行；flow 表示每个 flow 一行",
    )

    parser.add_argument(
        "--bpe_tokenizer_model",
        type=str,
        default=TOKENIZER_MODEL_ETBERT_WORDPIECE,
        choices=[
            TOKENIZER_MODEL_ETBERT_WORDPIECE,
            TOKENIZER_MODEL_ETBERT_WORDPIECE_TRAINER,
            TOKENIZER_MODEL_BPE,
        ],
        help="默认 etbert_wordpiece：固定 0000-ffff；etbert_wordpiece_trainer 调用 WordPieceTrainer；bpe 表示旧的连续 hex BPE",
    )

    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="可选：保存中间 BPE 训练语料路径。不指定则使用临时文件",
    )

    parser.add_argument(
        "--keep_corpus",
        action="store_true",
        help="是否保留中间 BPE 训练语料",
    )

    parser.add_argument(
        "--no_lowercase",
        action="store_true",
        help="不把 hex token 转小写。默认会转小写",
    )

    parser.add_argument(
        "--test_text",
        type=str,
        default=None,
        help="可选：训练完成后用于测试 tokenizer 的 hex 字符串",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    use_temp_file = args.corpus_path is None

    if use_temp_file:
        temp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".txt",
        )
        corpus_path = temp.name
        temp.close()
    else:
        corpus_path = args.corpus_path

    stats = build_bpe_training_corpus(
        input_path=args.input,
        corpus_path=corpus_path,
        lowercase=not args.no_lowercase,
        unit=args.unit,
        tokenizer_model=args.bpe_tokenizer_model,
    )

    print("\n[CORPUS STATS]")
    for k, v in stats.items():
        print(f"{k}: {v}")

    tokenizer_json_path, vocab_txt_path = train_bpe_tokenizer(
        corpus_path=corpus_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        tokenizer_model=args.bpe_tokenizer_model,
    )

    test_tokenizer(
        tokenizer_json_path=tokenizer_json_path,
        test_text=args.test_text,
    )

    if use_temp_file and not args.keep_corpus:
        try:
            os.remove(corpus_path)
        except Exception:
            pass
    else:
        print(f"\n[INFO] corpus kept at: {corpus_path}")


if __name__ == "__main__":
    main()
