import re
from typing import List, Set, Tuple, Dict

def extract_citations(text: str) -> List[Tuple[int, int, str]]:
    """
    从文本中提取所有引用标记
    
    Returns:
        List of (start_pos, end_pos, citation_text) for each citation
    """
    # 匹配模式：方括号内的数字，支持 [1], [12], [1,2,3], [1-5] 等格式
    # 简单模式：匹配单个或多个数字
    pattern = r'\[(\d+(?:[-,]\s*\d+)*)\]'
    
    citations = []
    for match in re.finditer(pattern, text):
        start, end = match.span()
        citations.append((start, end, match.group()))
    
    return citations

def get_cited_indices(citations: List[Tuple[int, int, str]]) -> Set[int]:
    """从引用标记中提取所有被引用的论文索引"""
    indices = set()
    for _, _, citation in citations:
        # 提取方括号内的数字
        content = citation.strip('[]')
        # 处理 [1,2,3] 或 [1-3] 等复合格式
        parts = re.split(r'[,\-]', content)
        for part in parts:
            part = part.strip()
            if part.isdigit():
                indices.add(int(part))
    return indices

class CitationValidator:
    def __init__(self, metadata_store: Dict[int, Dict]):
        """
        初始化引用验证器
        
        Args:
            metadata_store: 论文元数据存储，key为论文索引，value为论文元数据
                           格式如 {1: {"title": "...", "authors": [...]}}
        """
        self.metadata_store = metadata_store
    
    def validate_citations(self, text: str) -> Tuple[str, List[int]]:
        """
        验证文本中的所有引用
        
        Returns:
            - 清理后的文本（移除无效引用对应的句子）
            - 无效引用的索引列表
        """
        citations = extract_citations(text)
        cited_indices = get_cited_indices(citations)
        
        # 检查哪些索引在metadata_store中不存在
        valid_indices = {idx for idx in cited_indices if idx in self.metadata_store}
        invalid_indices = cited_indices - valid_indices
        
        if not invalid_indices:
            return text, []
        
        # 对无效引用进行处理：移除包含无效引用的句子
        cleaned_text = self._remove_sentences_with_invalid_citations(text, invalid_indices)
        
        return cleaned_text, list(invalid_indices)
    
    def _remove_sentences_with_invalid_citations(self, text: str, invalid_indices: Set[int]) -> str:
        """移除包含无效引用的句子"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            # 检查句子中是否包含无效引用
            citations_in_sentence = extract_citations(sentence)
            indices_in_sentence = get_cited_indices(citations_in_sentence)
            
            if not indices_in_sentence.intersection(invalid_indices):
                cleaned_sentences.append(sentence)
        
        return " ".join(cleaned_sentences)
    
    def verify_and_reindex(self, text: str, source_mapping: Dict[int, int]) -> str:
        """
        重新索引引用：当上下文chunk顺序变化时，将旧索引映射到新索引
        
        Args:
            text: 待处理的文本
            source_mapping: 旧索引到新索引的映射，如 {3: 1, 5: 2}
        """
        def replace_match(match):
            old_indices = match.group(1)
            # 解析旧索引并映射
            new_indices = []
            for part in re.split(r'[,\-]', old_indices):
                part = part.strip()
                if part.isdigit():
                    new_idx = source_mapping.get(int(part), int(part))
                    new_indices.append(str(new_idx))
            return f"[{','.join(new_indices)}]"
        
        pattern = r'\[(\d+(?:[-,]\s*\d+)*)\]'
        return re.sub(pattern, replace_match, text)
    
class RAGCitationValidator(CitationValidator):
    """基于向量库的引用验证器——通过语义相似度验证引用是否真实存在"""
    
    def __init__(self, rag_engine):  # rag_engine来自成员B
        super().__init__(metadata_store={})
        self.rag_engine = rag_engine
        self.context_cache = {}  # 缓存已检索的上下文
    
    def validate_with_semantic_search(self, claim: str, expected_source_idx: int) -> bool:
        """
        通过语义搜索验证某个论断是否能在预期来源中找到支撑
        
        Args:
            claim: 需要验证的论断句子
            expected_source_idx: 声称的来源索引
        
        Returns:
            True如果检索结果中包含预期来源的内容
        """
        # 调用成员B的retrieve接口
        retrieved = self.rag_engine.retrieve(claim, top_k=5)
        
        # 检查预期来源是否出现在Top-K结果中
        for chunk in retrieved:
            if chunk.get("metadata", {}).get("source_index") == expected_source_idx:
                return True
        return False
    
    def hybrid_validate(self, text: str) -> Tuple[str, Dict]:
        """
        混合验证：先做索引验证，再对剩余内容做语义验证
        
        Returns:
            - 验证后的文本
            - 验证统计信息
        """
        stats = {"total_citations": 0, "valid_indices": 0, "invalid_indices": 0, 
                 "semantic_passed": 0, "semantic_failed": 0}
        
        # 第一阶段：索引验证
        citations = extract_citations(text)
        stats["total_citations"] = len(citations)
        
        # 提取所有论断句子和对应的引用
        claims_with_citations = self._extract_claims_with_citations(text)
        
        validated_claims = []
        for claim, cited_indices in claims_with_citations:
            # 对每个引用进行验证
            valid_claims = []
            for idx in cited_indices:
                if idx in self.metadata_store:
                    stats["valid_indices"] += 1
                    valid_claims.append(claim)
                else:
                    # 尝试语义验证
                    if self.validate_with_semantic_search(claim, idx):
                        stats["semantic_passed"] += 1
                        valid_claims.append(claim)
                    else:
                        stats["semantic_failed"] += 1
                        stats["invalid_indices"] += 1
        
        return text, stats  # 简化版，实际应基于验证结果构建最终文本