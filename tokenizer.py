"""
V4: Character-level Tokenizer
V5: BPE (Byte-Pair Encoding) Tokenizer
"""

import tiktoken


class CharTokenizer:
    
    def __init__(self, chars):
        """
        初始化 tokenizer
        
        Args:
            chars: 字符列表或字符串,包含所有可能的字符
        """
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        
        # 字符到索引的映射
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        # 索引到字符的映射
        self.itos = {i: char for i, char in enumerate(self.chars)}
    
    def encode(self, text):
        """
        将文本编码为 token 序列
        
        Args:
            text: 输入文本字符串
            
        Returns:
            token 列表
        """
        return [self.stoi[char] for char in text]
    
    def decode(self, tokens):
        """
        将 token 序列解码为文本
        
        Args:
            tokens: token 列表
            
        Returns:
            解码后的文本字符串
        """
        return ''.join([self.itos[token] for token in tokens])


class BPETokenizer:
    """
    V5: BPE (Byte-Pair Encoding) Tokenizer
    
    优点:
    - 词汇表合理 (~50K)
    - 序列长度短 3-5 倍
    - 理解词汇和子词
    - 训练速度快 2-3 倍
    - 与 GPT-2/3 兼容
    """
    
    def __init__(self, model_name="gpt2"):
        """
        初始化 BPE tokenizer
        
        Args:
            model_name: tiktoken 支持的模型名
                - "gpt2": GPT-2 tokenizer (vocab_size=50257)
                - "cl100k_base": GPT-3.5/4 tokenizer (vocab_size=100256)
        """
        self.model_name = model_name
        self.encoder = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoder.n_vocab
        
        print(f"✅ BPE Tokenizer 初始化完成")
        print(f"   模型: {model_name}")
        print(f"   词汇表大小: {self.vocab_size:,}")
    
    def encode(self, text):
        """将文本编码为 token 序列"""
        return self.encoder.encode(text, allowed_special="all")
    
    def decode(self, tokens):
        """将 token 序列解码为文本"""
        return self.encoder.decode(tokens)
    
    def encode_batch(self, texts):
        """批量编码"""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists):
        """批量解码"""
        return [self.decode(tokens) for tokens in token_lists]


def create_tokenizer(tokenizer_type="char", text=None, model_name="gpt2"):
    """
    工厂函数：创建 tokenizer
    
    Args:
        tokenizer_type: "char" 或 "bpe"
        text: char tokenizer 需要的文本
        model_name: bpe tokenizer 的模型名
    """
    if tokenizer_type == "char":
        if text is None:
            raise ValueError("CharTokenizer 需要提供 text 参数")
        return CharTokenizer(text)
    elif tokenizer_type == "bpe":
        return BPETokenizer(model_name)
    else:
        raise ValueError(f"不支持的 tokenizer 类型: {tokenizer_type}")


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("V4: Character-level Tokenizer")
    print("=" * 60)
    
    # 创建一个简单的 tokenizer,包含常见字符
    text = "hello world"
    char_tokenizer = CharTokenizer(text)
    
    print(f"词汇表大小: {char_tokenizer.vocab_size}")
    print(f"字符集: {char_tokenizer.chars}")
    
    encoded = char_tokenizer.encode(text)
    print(f"\n原文: '{text}'")
    print(f"编码: {encoded}")
    print(f"Token 数量: {len(encoded)}")
    print(f"解码: '{char_tokenizer.decode(encoded)}'")
    
    print("\n" + "=" * 60)
    print("V5: BPE Tokenizer")
    print("=" * 60)
    
    # 测试 BPE tokenizer
    try:
        bpe_tokenizer = BPETokenizer("gpt2")
        
        test_texts = [
            "hello world",
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is amazing!"
        ]
        
        print("\n对比测试:")
        print("-" * 60)
        for test_text in test_texts:
            bpe_encoded = bpe_tokenizer.encode(test_text)
            
            print(f"\n文本: '{test_text}'")
            print(f"BPE tokens: {bpe_encoded}")
            print(f"BPE 长度: {len(bpe_encoded)}")
            print(f"Char 长度: {len(test_text)}")
            print(f"压缩比: {len(test_text)/len(bpe_encoded):.1f}x")
            print(f"解码: '{bpe_tokenizer.decode(bpe_encoded)}'")
        
        print("\n" + "=" * 60)
        print("✅ BPE 显著减少了 token 数量！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n⚠️  BPE Tokenizer 测试失败: {e}")
        print("请先安装: pip install tiktoken")
