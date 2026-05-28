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


# 示例使用
if __name__ == "__main__":
    # 创建一个简单的 tokenizer,包含常见字符
    text = "hello world"
    tokenizer = CharTokenizer(text)
    
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"字符集: {tokenizer.chars}")
    print(f"stoi: {tokenizer.stoi}")
    print(f"itos: {tokenizer.itos}")
    
    # 编码
    encoded = tokenizer.encode("hello")
    print(f"\n编码 'hello': {encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded[:3])
    print(f"解码 {encoded[:3]}: '{decoded}'")