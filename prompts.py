# V1：基础版——最简大纲
OUTLINE_PROMPT_V1 = """
You are an academic research assistant. Generate a structured outline for a survey paper on the given topic.

Topic: {topic}

Papers retrieved (titles and abstracts):
{paper_summaries}

Output format (JSON):
{{
    "title": "proposed paper title",
    "sections": [
        {{"title": "section name", "subsections": ["subsection1", "subsection2"]}}
    ]
}}

Requirements:
- Include at least: Introduction, Background, Main Approaches, Challenges, Conclusion
- Keep subsection titles concise and descriptive
"""

# V2：进阶版——带章节写作指导
OUTLINE_PROMPT_V2 = """
You are an academic research assistant specializing in systematic literature reviews.

**Role & Constraints**
- You MUST base your outline ONLY on the retrieved papers provided below.
- You are NOT allowed to invent topics not covered by the retrieved papers.
- If the retrieved papers lack sufficient coverage for a standard section, note "Insufficient data" instead of fabricating.

**Topic**: {topic}

**Retrieved Papers** (title + abstract):
{paper_summaries}

**Task**: Generate a detailed survey paper outline.

**Output Format** (strict JSON):
{{
    "title": "proposed paper title",
    "sections": [
        {{
            "title": "section title",
            "subsections": ["subsection title 1", "subsection title 2"],
            "key_points": ["key argument 1", "key argument 2"],
            "recommended_sources": ["paper title to cite here"]
        }}
    ]
}}

**Required Sections**:
1. Introduction (must include: problem definition, scope, contribution statement)
2. Background and Preliminaries
3. Taxonomy/Categorization of Existing Work
4. Key Approaches and Methods
5. Evaluation and Benchmarks (if applicable)
6. Open Challenges and Future Directions
7. Conclusion

**Quality Guidelines**:
- Subsections should reflect the actual topics found in the retrieved papers
- Key points should be specific and arguable, not generic statements
- Each section should have at least 2-3 recommended sources
- If a section has insufficient sources, mark it with "⚠️ Limited coverage"
"""

# V3：迭代优化版——带自我反思机制
OUTLINE_PROMPT_V3 = """
You are an academic research assistant. Follow this process:

**Step 1 - Gap Analysis**:
First, analyze what topics are WELL covered vs. POORLY covered in the retrieved papers.

**Step 2 - Generate Outline**:
Based on the gap analysis, create a balanced outline. For under-covered topics, you may propose them as "Future Directions".

**Step 3 - Self-Check**:
After generating the outline, verify:
- Does each section have at least 2 supporting papers? If not, mark as "⚠️ Limited evidence"
- Are there any invented topics not in the source material? If yes, remove or move to Future Directions.

**Topic**: {topic}
**Retrieved Papers**: {paper_summaries}

Proceed step by step, then output the final outline in JSON format.
"""

SECTION_GENERATION_SYSTEM_PROMPT = """You are an academic survey paper writer.

**Core Principles**:
1. You MUST base ALL content EXCLUSIVELY on the retrieved paper excerpts provided in the user context.
2. You MUST NOT add any external knowledge, assumptions, or invented facts.
3. Every factual claim MUST be followed by a citation in the format [N], where N is the index of the source paper.
4. If the provided context does NOT contain information needed to write a required subsection, write "Insufficient information available" instead of fabricating content.
5. Use formal academic tone. Avoid first-person pronouns ("we", "our", "I").
6. Citations should appear immediately after the claim they support, before the period.
   Example: "The Transformer architecture introduced self-attention mechanisms [1]. This enabled parallel processing of sequences [2]."
7. You MUST only cite source indices that appear in the provided context (e.g., 1, 2, 3...).
8. Do NOT invent citation numbers like 999.

**Output Format**:
- Write in well-structured paragraphs
- Use markdown headings for subsections (### Subsection Title)
- No extra commentary or meta-instructions in the output
"""

SECTION_GENERATION_USER_TEMPLATE = """
**Chapter Title**: {section_title}

**Subsections to Cover**: {subsections}

**Retrieved Context** (each excerpt is tagged with its source index):
{context_chunks}

**Guidelines for this section**:
- {section_guidelines}

**Instructions**:
Write the content for this chapter following the system instructions. Each claim must be cited with the corresponding source index [N].
"""

# 不同章节的专属写作指导
SECTION_GUIDELINES = {
    "introduction": "Establish the research background, state the problem, define scope, and announce the contribution of this survey.",
    "background": "Define key terminology and foundational concepts needed to understand the main content. Cite definitional sources.",
    "methods": "Categorize existing approaches. Compare and contrast different methods. Each method description must cite its originating paper.",
    "challenges": "Identify open problems and limitations of current approaches. Support each challenge with evidence from the literature.",
    "conclusion": "Summarize key findings, discuss implications, and suggest future research directions."
}

def format_papers_for_prompt(papers: list) -> str:
    """
    格式化论文列表为LLM可读的文本块
    
    Args:
        papers: 论文列表，每个元素包含 title, authors, abstract 等字段
    
    Returns:
        格式化的字符串
    """
    formatted = []
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'N/A')
        authors = paper.get('authors', [])
        author_str = ', '.join(authors[:3]) if authors else 'Unknown'
        abstract = paper.get('abstract', 'No abstract available')
        # 截断过长的摘要
        if len(abstract) > 800:
            abstract = abstract[:800] + "..."
        formatted.append(
            f"[Paper {i}] Title: {title}\n"
            f"        Authors: {author_str}\n"
            f"        Abstract: {abstract}\n"
        )
    return "\n".join(formatted)


def format_context_chunks(chunks: list) -> str:
    """
    格式化检索结果为LLM可读的上下文块
    
    Args:
        chunks: 检索结果列表，每个元素包含 content, metadata, score 等字段
    
    Returns:
        格式化的字符串，每个块带有 [Source i] 标签
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get('content', '')
        meta = chunk.get('metadata', {})
        paper_title = meta.get('paper_title', 'Unknown')
        authors = meta.get('authors', [])
        author_str = ', '.join(authors[:2]) if authors else 'Unknown'
        
        formatted.append(
            f"[Source {i}] Paper: {paper_title}\n"
            f"          Authors: {author_str}\n"
            f"          Content: {content}\n"
        )
    return "\n".join(formatted)