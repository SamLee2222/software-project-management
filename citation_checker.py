import re
from typing import List, Set, Tuple, Dict, Optional, Any
from dataclasses import dataclass

def extract_citations(text: str) -> List[Tuple[int, int, str]]:
    """
    从文本中提取所有引用标记
    
    Returns:
        List of (start_pos, end_pos, citation_text) for each citation
    """
    pattern = r'\[(\d+(?:[-,]\s*\d+)*)\]'
    
    citations = []
    for match in re.finditer(pattern, text):
        start, end = match.span()
        citations.append((start, end, match.group()))
    
    return citations

def get_cited_indices(citations: List[Tuple[int, int, str]]) -> Set[int]:
    """
    从引用标记中提取所有被引用的论文索引
    支持 [1], [12], [1,2,3], [1-3] 等格式
    """
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

def split_sentences(text: str) -> List[str]:
    """将文本分割为句子（保留标点符号）"""
    # 中英文句号、问号、感叹号
    sentences = re.split(r'(?<=[。.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_claims_with_citations(text: str) -> List[Tuple[str, List[int]]]:
    """
    提取每个句子及其包含的引用索引列表
    Returns:
        List of (sentence, [citation_indices])
    """
    sentences = split_sentences(text)
    result = []
    for sent in sentences:
        citations = extract_citations(sent)
        indices = get_cited_indices(citations)
        if indices:
            result.append((sent, sorted(indices)))
    return result

class CitationValidator:
    def __init__(self, metadata_store: Dict[int, Dict]):
        """
        初始化引用验证器
        Args:
            metadata_store: 论文元数据存储，key为论文索引，value为论文元数据
                           格式如 {1: {"title": "...", "authors": [...]}}
        """
        self.metadata_store = metadata_store or {}

    def load_from_metadata_db(self, db_path: str = "./data/metadata.db") -> Set[int]:
        """
        从 SQLite 元数据库加载有效索引（与成员 C 集成）

        Args:
            db_path: SQLite 数据库路径

        Returns:
            有效索引集合
        """
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # 根据实际表结构调整，这里假设表名为 papers，有 status 字段
            cursor.execute("SELECT rowid FROM papers WHERE status = 'downloaded'")
            rows = cursor.fetchall()
            valid_indices = {row[0] for row in rows}
            conn.close()

            # 构建简单的 metadata_store（仅包含索引，详细元数据可后续扩展）
            for idx in valid_indices:
                if idx not in self.metadata_store:
                    self.metadata_store[idx] = {}
            return valid_indices
        except Exception as e:
            print(f"Warning: Failed to load metadata from {db_path}: {e}")
            return set()
        
    def load_from_chromadb(self, collection_name: str = "papers") -> Set[int]:
        """
        从 ChromaDB 加载有效索引（与成员 B 集成）

        Args:
            collection_name: ChromaDB 集合名称

        Returns:
            有效索引集合
        """
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./data/chromadb"
            ))
            collection = client.get_collection(collection_name)
            results = collection.get(include=["metadatas"])
            valid_indices = set()
            if results and results.get('metadatas'):
                for meta in results['metadatas']:
                    if meta and 'source_index' in meta:
                        idx = meta['source_index']
                        valid_indices.add(idx)
                        if idx not in self.metadata_store:
                            self.metadata_store[idx] = meta
            client.clear_system_cache()
            return valid_indices
        except Exception as e:
            print(f"Warning: Failed to load from ChromaDB: {e}")
            return set()
    
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
        
        # 标记包含无效引用的句子
        marked_text = self._mark_invalid_sentences(text, invalid_indices)
        return marked_text, list(invalid_indices)
    
    def _mark_invalid_sentences(self, text: str, invalid_indices: Set[int]) -> str:
        """
        将包含无效引用的句子标记为 [UNVERIFIED]

        Args:
            text: 原始文本
            invalid_indices: 无效引用索引集合

        Returns:
            标记后的文本
        """
        sentences = split_sentences(text)
        marked_sentences = []

        for sentence in sentences:
            citations = extract_citations(sentence)
            indices_in_sentence = get_cited_indices(citations)

            if indices_in_sentence.intersection(invalid_indices):
                # 在句子开头添加 [UNVERIFIED] 标记
                marked_sentences.append(f"[UNVERIFIED] {sentence}")
            else:
                marked_sentences.append(sentence)

        return " ".join(marked_sentences)
    
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
    
    def get_statistics(self, text: str) -> Dict[str, Any]:
        """
        获取文本的引用统计信息（不修改文本）

        Returns:
            统计字典
        """
        citations = extract_citations(text)
        cited_indices = get_cited_indices(citations)

        valid_count = sum(1 for idx in cited_indices if idx in self.metadata_store)
        invalid_count = len(cited_indices) - valid_count

        return {
            "total_citations": len(citations),
            "unique_indices": len(cited_indices),
            "valid_indices": valid_count,
            "invalid_indices": invalid_count,
            "hit_rate": valid_count / len(cited_indices) if cited_indices else 1.0
        }
    
class RAGCitationValidator(CitationValidator):
    """基于向量库的引用验证器——通过语义相似度验证引用是否真实存在"""
    
    def __init__(self, rag_engine=None, metadata_store: Dict[int, Dict] = None):
        """
        初始化 RAG 验证器

        Args:
            rag_engine: 成员 B 提供的 RAG 引擎，需要有 retrieve(query, top_k) 方法
            metadata_store: 论文元数据存储
        """
        super().__init__(metadata_store)
        self.rag_engine = rag_engine
        self.context_cache = {}
    
    def validate_with_semantic_search(self, claim: str, expected_source_idx: int) -> bool:
        """
        通过语义搜索验证某个论断是否能在预期来源中找到支撑

        Args:
            claim: 需要验证的论断句子
            expected_source_idx: 声称的来源索引

        Returns:
            True 如果检索结果中包含预期来源的内容
        """
        if not self.rag_engine:
            return False

        try:
            retrieved = self.rag_engine.retrieve(claim, top_k=5)
            for chunk in retrieved:
                meta = chunk.get("metadata", {})
                # 支持多种可能的字段名
                idx = meta.get("source_index") or meta.get("index")
                if idx == expected_source_idx:
                    return True
            return False
        except Exception as e:
            print(f"Semantic validation error: {e}")
            return False
        
    def _extract_claims_with_citations(self, text: str) -> List[Tuple[str, List[int]]]:
        """提取每个句子及其包含的引用索引列表"""
        return extract_claims_with_citations(text)
    
    def hybrid_validate(self, text: str) -> Tuple[str, Dict]:
        """
        混合验证：先做索引验证，对索引无效的引用尝试语义验证

        Returns:
            - 标记后的文本
            - 验证统计信息
        """
        stats = {
            "total_citations": 0,
            "valid_indices": 0,
            "invalid_indices": 0,
            "semantic_passed": 0,
            "semantic_failed": 0
        }

        # 提取所有带引用的句子
        claims_with_citations = self._extract_claims_with_citations(text)

        # 用于收集需要标记的句子
        sentences_to_mark = set()
        all_invalid_indices = set()

        for sentence, cited_indices in claims_with_citations:
            stats["total_citations"] += len(cited_indices)
            sentence_has_invalid = False

            for idx in cited_indices:
                if idx in self.metadata_store:
                    stats["valid_indices"] += 1
                else:
                    # 索引无效，尝试语义验证
                    if self.validate_with_semantic_search(sentence, idx):
                        stats["semantic_passed"] += 1
                        # 语义验证通过，可以认为有效
                        stats["valid_indices"] += 1
                    else:
                        stats["semantic_failed"] += 1
                        stats["invalid_indices"] += 1
                        sentence_has_invalid = True
                        all_invalid_indices.add(idx)

            if sentence_has_invalid:
                sentences_to_mark.add(sentence)

        # 标记包含无效引用的句子
        if sentences_to_mark:
            marked_text = self._mark_specific_sentences(text, sentences_to_mark)
        else:
            marked_text = text

        return marked_text, stats
    
    def _mark_specific_sentences(self, text: str, sentences_to_mark: Set[str]) -> str:
        """将特定句子标记为 [UNVERIFIED]"""
        sentences = split_sentences(text)
        marked_sentences = []

        for sentence in sentences:
            if sentence in sentences_to_mark:
                marked_sentences.append(f"[UNVERIFIED] {sentence}")
            else:
                marked_sentences.append(sentence)

        return " ".join(marked_sentences)

def quick_validate(text: str, valid_indices: Set[int]) -> Tuple[str, Dict]:
    """
    快速验证（仅索引校验）

    Args:
        text: 待验证的文本
        valid_indices: 有效的索引集合

    Returns:
        (标记后的文本, 统计信息)
    """
    metadata = {idx: {} for idx in valid_indices}
    validator = CitationValidator(metadata)
    marked_text, _ = validator.validate_citations(text)
    stats = validator.get_statistics(text)
    return marked_text, stats

if __name__ == "__main__":
    # 测试文本
    test_text = """The Transformer architecture introduced self-attention mechanisms [1].
    This enabled parallel processing of sequences [2].
    However, some studies suggest alternative approaches [999].
    The impact on efficiency is significant."""

    # 模拟有效索引
    valid_indices = {1, 2, 3, 4, 5}

    print("=" * 60)
    print("原始文本:")
    print(test_text)
    print("\n" + "=" * 60)

    # 基础验证
    validator = CitationValidator({idx: {} for idx in valid_indices})
    marked_text, invalid = validator.validate_citations(test_text)
    stats = validator.get_statistics(test_text)

    print("标记后的文本:")
    print(marked_text)
    print("\n统计信息:")
    print(f"  - 总引用数: {stats['total_citations']}")
    print(f"  - 唯一索引数: {stats['unique_indices']}")
    print(f"  - 有效索引: {stats['valid_indices']}")
    print(f"  - 无效索引: {stats['invalid_indices']}")
    print(f"  - 命中率: {stats['hit_rate']:.1%}")
    print(f"  - 无效引用列表: {invalid}")

    # 测试快速函数
    print("\n" + "=" * 60)
    marked, stats2 = quick_validate(test_text, valid_indices)
    print("quick_validate 输出:")
    print(marked)
    print(f"命中率: {stats2['hit_rate']:.1%}")