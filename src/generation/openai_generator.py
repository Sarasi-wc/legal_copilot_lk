"""
OpenAI API-based answer generator.
Alternative to local LLM for users with OpenAI API access.
"""

import os
from typing import Dict, List, Optional
from openai import OpenAI

from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)


class OpenAIGenerator:
    """
    Generates answers using OpenAI API (GPT-4, GPT-3.5, etc.)
    Drop-in replacement for AnswerGenerator.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI generator.

        Args:
            model_name: OpenAI model name (defaults to OPENAI_MODEL from settings)
            max_tokens: Maximum tokens to generate (defaults to OPENAI_MAX_TOKENS from settings, default 100 for demo)
            temperature: Generation temperature (defaults to OPENAI_TEMPERATURE from settings)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        # Use settings if not provided, with defaults for demo
        self.model_name = model_name or settings.openai_model
        self.max_tokens = max_tokens if max_tokens is not None else settings.openai_max_tokens
        self.temperature = temperature if temperature is not None else settings.openai_temperature

        # Initialize OpenAI client (only if API key is available)
        api_key_value = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key_value:
            self.client = OpenAI(api_key=api_key_value)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided. Client not initialized.")

        logger.info(f"Initialized OpenAI generator with model: {self.model_name}, max_tokens: {self.max_tokens}")

    def generate_answer(
        self,
        query: str,
        evidence_plan: Dict,
        formatted_evidence: str
    ) -> Dict:
        """
        Generate citation-grounded answer using OpenAI API.

        Args:
            query: User query
            evidence_plan: Evidence plan from EvidencePlanner
            formatted_evidence: Formatted evidence string

        Returns:
            Dictionary with answer and metadata
        """
        logger.debug(f"Generating answer for query: {query}")

        # Check evidence sufficiency
        if not evidence_plan['is_sufficient']:
            return self._create_abstention_response(query, "insufficient_evidence")

        # Query-scope gate: for borderline relevance, check if the question is specifically
        # answerable from the Constitution before invoking the full generation pipeline.
        # Runs only when max rerank score < 3.0 (clearly in-corpus queries score 4.9+
        # and skip this check; out-of-domain queries below -0.5 are already caught by
        # EvidencePlanner Layer 0). This gate targets rights-adjacent queries that pass
        # evidence planning but ask about statutory topics not covered by the Constitution.
        max_rerank = max(
            (p.get('rerank_score', 0) for p in evidence_plan.get('passages', [])),
            default=0
        )
        if max_rerank < 3.0:
            if not self._is_constitutionally_answerable(query):
                return self._create_abstention_response(query, "out_of_scope")

        # Build prompt
        prompt = self._build_prompt(query, formatted_evidence)

        # Generate with OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a constitutional law expert specialising in the Constitution of the Democratic Socialist Republic of Sri Lanka (1978). Answer questions accurately and concisely, grounded strictly in the evidence provided to you, using clear and accessible language for users who may not have legal expertise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            answer_text = response.choices[0].message.content.strip()

            # Ensure disclaimer is present regardless of model compliance
            disclaimer = (
                "\n\n*This information is for general reference only and does not "
                "constitute legal advice. Consult a qualified Sri Lankan legal "
                "practitioner for advice on your specific situation.*"
            )
            if "does not constitute legal advice" not in answer_text:
                answer_text += disclaimer

            # Parse citations
            citations = self._extract_citations(answer_text, evidence_plan)

            return {
                'query': query,
                'answer': answer_text,
                'citations': citations,
                'evidence_used': evidence_plan['filtered_passages'],
                'abstained': False,
                'abstention_reason': None,
                'model': self.model_name,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._create_abstention_response(query, "generation_error")

    def _build_prompt(self, query: str, evidence: str) -> str:
        """Build prompt for OpenAI API."""
        prompt = f"""Answer the following constitutional law question using ONLY the evidence passages provided below. If a claim cannot be supported by the evidence passages, state that the evidence is insufficient — do not draw on background knowledge to fill the gap. Your response must be grounded exclusively in the EVIDENCE section

TASK
Begin with a direct answer to the question, then explain the supporting constitutional provisions. Provide a legally accurate answer that:
- Addresses ONLY what the question asks — do not include tangentially related provisions, background content, or legal implications not stated in the evidence
- Do not infer unstated legal consequences from the constitutional text — state only what the evidence explicitly provides
- Do not assume that a provision is irrelevant simply because it was not included in the evidence — if coverage appears incomplete, acknowledge it
- Cites every factual claim with an inline citation, including all relevant balancing clauses and qualifications found in BOTH the primary and supporting evidence passages
- Uses direct quotes (with quotation marks) for key constitutional language
- Preserves exact statutory modal verbs — if the evidence uses "shall", "may", or "must", reproduce those words exactly; never substitute "will", "can", or "should"
- Writes in prose paragraphs — do not use bullet points or numbered lists
- If two evidence passages conflict, describe the conflict explicitly rather than reconciling them — do not choose one interpretation over another without justification from the evidence
- Acknowledges gaps explicitly if the evidence does not fully address the question
- States ambiguity rather than selecting one interpretation without justification when the evidence admits more than one reading
- For enumeration questions ("what are the rights / grounds / requirements"): synthesise from ALL evidence passages provided — both primary and supporting — not just the primary passage, and cite each relevant passage individually

CITATION FORMAT (mandatory for every factual claim)
Use this exact format: [Act Name, Article X]

The Act name must exactly match the "Act:" field shown in the evidence — do NOT substitute, paraphrase, or guess a different source name. The Constitution of Sri Lanka uses ARTICLES — always write [Constitution, Article X], never [Constitution, Section X]. Note: even if an evidence passage is internally labelled "Section X", the citation must still use Article X for the Constitution.

Rules:
- One citation per factual claim — never cite a range (e.g. Article 7–13 is wrong; cite each individually)
- Use numeric identifiers only (9, 14, 14(1)(a)) — never Roman numerals or chapter titles
- If multiple evidence passages are relevant, cite each one individually
- Use [Act Name, Text] or [Act Name, General] only when no specific article number can be identified from the evidence

✅ CORRECT: [Constitution, Article 9]
✅ CORRECT: [Constitution, Article 14(1)(a)]
✅ CORRECT: [Constitution, Text]     (only when no article number can be identified)
✅ CORRECT: [Constitution, General]  (alternative to Text for document-level passages)
❌ WRONG: [Constitution, Section 9]   ← The Constitution uses ARTICLES, never Sections
❌ WRONG: [Constitution, Section 14]  ← Always write Article, never Section
❌ WRONG: [Constitution, Chapter II]  ← Never use chapter titles
❌ WRONG: [Constitution, Article 7–13] ← Never cite ranges; cite each article individually
❌ WRONG: [Constitution]              ← Always include the article number or Text

READ BEFORE ANSWERING
Read every passage in the EVIDENCE section below — both PRIMARY EVIDENCE and SUPPORTING EVIDENCE — before you begin writing your answer. You may omit a supporting passage from your citations only if it is genuinely irrelevant to the question asked.

EVIDENCE:
{evidence}

QUESTION: {query}

BEFORE OUTPUTTING YOUR ANSWER
Run through this checklist:
1. Citations — every factual claim has an inline citation traceable to a specific passage in the EVIDENCE section; remove any that cannot be traced
2. Disclaimer — the disclaimer appears on a new line at the end of your answer
3. Scope — your answer addresses only what the question asked; remove any tangential content

Reminder: every factual claim must carry a citation in the format [Act Name, Article X]. The Constitution uses ARTICLES, never Sections.

FORMAT EXAMPLES:

Factual query:
Article 9 mandates that "the Republic of Sri Lanka shall give to Buddhism the foremost place" [Constitution, Article 9]. The State must protect and foster the Buddha Sasana while assuring the rights of all religions under Articles 10 and 14(1)(e) [Constitution, Article 9]. The freedom of thought, conscience and religion is separately guaranteed [Constitution, Article 10].

Enumeration query ("what are the..."):
The Constitution guarantees several fundamental rights. Freedom of thought, conscience and religion is protected [Constitution, Article 10]. Freedom of speech and expression is guaranteed [Constitution, Article 14(1)(a)]. The right to equality before the law is enshrined [Constitution, Article 12(1)].

End your answer with the following disclaimer on a new line:
*This information is for general reference only and does not constitute legal advice. Consult a qualified Sri Lankan legal practitioner for advice on your specific situation.*"""
        return prompt

    def _is_constitutionally_answerable(self, query: str) -> bool:
        """
        Binary scope classifier: returns True if the query is specifically answerable
        from the Sri Lankan Constitution, False if it requires other legislation.

        Used as a pre-generation gate for queries with borderline relevance scores
        (max cross-encoder rerank score < 3.0). At this score range, the evidence
        planner may admit rights-adjacent queries (e.g. bail/liberty, copyright/expression)
        whose specific answer is governed by statutory law, not the Constitution.

        A separate low-cost API call (max_tokens=5, temperature=0) is used so the
        classifier decision is independent of the generation prompt.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": (
                        "Answer YES or NO only (no other words).\n\n"
                        "Is this question specifically answerable from the Constitution "
                        "of Sri Lanka (1978)?\n\n"
                        "The Constitution answers: presidential term and powers, "
                        "Parliament structure and elections, voting qualifications, "
                        "fundamental rights declarations (what rights exist), "
                        "constitutional amendment procedures, emergency proclamation, "
                        "judicial appointments, Attorney General and other constitutional "
                        "officers' constitutional functions, Cabinet formation.\n\n"
                        "The Constitution does NOT answer: criminal penalties, bail "
                        "conditions, drug trafficking sentences, divorce grounds or "
                        "procedures, copyright penalties, consumer protection fines, "
                        "income tax rates, company registration procedures, labour law "
                        "notice periods, minimum wage, immigration procedures, traffic "
                        "speed limits or fines, licensing requirements, will formalities.\n\n"
                        f"Question: {query}"
                    )
                }],
                max_tokens=5,
                temperature=0
            )
            answer = resp.choices[0].message.content.strip().upper()
            is_constitutional = not answer.startswith("NO")
            logger.info(
                f"Scope classifier for '{query[:60]}': {answer!r} → "
                f"{'answerable' if is_constitutional else 'out_of_scope'}"
            )
            return is_constitutional
        except Exception as e:
            logger.warning(f"Scope classifier error: {e} — defaulting to answerable")
            return True  # fail-open: attempt generation rather than silently abstaining

    def _extract_citations(
        self,
        answer_text: str,
        evidence_plan: Dict
    ) -> List[Dict]:
        """Extract citations from generated answer."""
        import re

        citations = []

        # First check for invalid range citations (e.g., Article 7-13)
        range_pattern = r'\[([^,\]]+),\s*(?:Article|Section|Chapter)?\s*([0-9]+-[0-9]+)\]'
        range_matches = re.findall(range_pattern, answer_text, re.IGNORECASE)
        if range_matches:
            logger.warning(f"Found invalid range citations: {range_matches}")

        # Match: [Act Name, Article/Section/Chapter X] or [Act Name, X] or [Act Name, Text]
        # Pattern 1: Numeric citations: [Act Name, Article X] or [Act Name, X]
        # Capture the type word (Article/Section/Chapter) so it can be preserved in output.
        citation_pattern = r'\[([^,\]]+),\s*(Article|Section|Chapter)?\s*([0-9]+[A-Z]?(?:\([0-9]+\))?(?:\([a-z]\))?)\]'
        # Pattern 2: Text citations: [Act Name, Text] or [Act Name, General]
        text_citation_pattern = r'\[([^,\]]+),\s*(Text|General)\]'

        matches = list(re.finditer(citation_pattern, answer_text, re.IGNORECASE))
        text_matches = list(re.finditer(text_citation_pattern, answer_text, re.IGNORECASE))
        
        # Combine matches, avoiding duplicates
        all_matches = matches + text_matches

        for match in all_matches:
            act_name = match.group(1).strip()

            # citation_pattern has 3 groups: (act_name, type_word, number)
            # text_citation_pattern has 2 groups: (act_name, Text/General)
            if len(match.groups()) == 3:
                citation_type = match.group(2)  # "Article", "Section", "Chapter", or None
                section = match.group(3).strip()
            elif len(match.groups()) == 2:
                citation_type = None
                section = match.group(2).strip()
            else:
                continue

            # Determine the correct display label:
            # - If the answer explicitly said "Article", preserve it.
            # - If act is the Constitution and no explicit type, default to "Article".
            # - Otherwise default to "Section".
            if citation_type:
                section_label = citation_type
            elif 'constitution' in act_name.lower():
                section_label = 'Article'
            else:
                section_label = 'Section'
            
            # Handle "Text" or "General" citations - match to highest-scored passage from same act
            # Prefer passages with higher scores (more relevant)
            if section.lower() in ['text', 'general']:
                matching_passage = None
                best_score = -1
                for passage in evidence_plan['passages']:
                    passage_act = passage.get('act_name', '')
                    act_matches = (
                        not passage_act  # empty act_name = Constitution passages
                        or act_name.lower() in passage_act.lower()
                        or passage_act.lower() in act_name.lower()
                        or 'constitution' in act_name.lower() and not passage_act
                    )
                    if act_matches:
                        # Prefer passages with higher scores (more relevant to query)
                        score = passage.get('normalized_score', passage.get('score', 0))
                        if score > best_score:
                            best_score = score
                            matching_passage = passage
                
                if matching_passage:
                    citations.append({
                        'act_name': act_name,
                        'section': section,
                        'section_label': section_label,
                        'has_matching_evidence': True,
                        'passage_id': matching_passage['passage_id']
                    })
                continue

            # Find matching passage for numeric citations - be flexible with matching
            matching_passage = None
            section_str = str(section)
            for passage in evidence_plan['passages']:
                passage_act = passage.get('act_name', '')
                meta = passage.get('metadata', {})
                passage_art = str(meta.get('article_number', '') or '')
                passage_sec = str(meta.get('section_number', '') or '')

                # Act name match: empty act_name in passage means Constitution corpus
                act_matches = (
                    not passage_act  # empty act_name = Constitution passages
                    or act_name.lower() in passage_act.lower()
                    or passage_act.lower() in act_name.lower()
                    or 'constitution' in act_name.lower() and not passage_act
                )
                if not act_matches:
                    continue

                # Section match: check metadata first (most reliable)
                if section_str == passage_art or section_str == passage_sec:
                    matching_passage = passage
                    break

                # Fallback: check passage text for explicit mention
                passage_text = passage.get('text', '').lower()
                section_mentions = [
                    f'article {section}',
                    f'section {section}',
                    f'chapter {section}',
                ]
                if any(s in passage_text for s in section_mentions):
                    matching_passage = passage
                    break

            # Last resort: highest-scored passage from same act (prefer more relevant)
            if not matching_passage:
                best_score = -1
                for passage in evidence_plan['passages']:
                    passage_act = passage.get('act_name', '')
                    if not passage_act or act_name.lower() in passage_act.lower():
                        score = passage.get('normalized_score', passage.get('score', 0))
                        if score > best_score:
                            best_score = score
                            matching_passage = passage

            citations.append({
                'act_name': act_name,
                'section': section,
                'section_label': section_label,
                'has_matching_evidence': matching_passage is not None,
                'passage_id': matching_passage['passage_id'] if matching_passage else None
            })

        # Also handle document-level citations: [Act Name] or [Act Name, Text] (for explanatory passages)
        # This handles cases where the LLM cites without article numbers for document-level comparisons
        doc_level_pattern = r'\[([^\]]+)\](?!\s*\[)'  # Match [Act Name] but not if followed by another [
        
        # First, remove already matched citations to avoid duplicates
        answer_without_citations = answer_text
        for citation in citations:
            # Remove matched citations from text
            pattern_to_remove = re.escape(f"[{citation['act_name']}, {citation['section']}]")
            answer_without_citations = re.sub(pattern_to_remove, '', answer_without_citations, flags=re.IGNORECASE)
        
        # Find document-level citations (without article/section numbers)
        doc_matches = re.finditer(r'\[([^\]]+)\]', answer_without_citations)
        for match in doc_matches:
            citation_text = match.group(1).strip()
            
            # Skip if it looks like it has a comma (might be a partial match)
            if ',' in citation_text and any(char.isdigit() for char in citation_text):
                continue
            
            # Check if this matches any explanatory/comparison passages
            matching_passage = None
            for passage in evidence_plan['passages']:
                meta = passage.get('metadata', {})
                content_type = meta.get('content_type', '')
                passage_act = passage.get('act_name', '')
                
                # Check if this is an explanatory/comparison passage
                is_explanatory = (
                    content_type in ['comparison', 'explanatory'] or
                    'explanatory' in passage.get('title', '').lower() or
                    'comparison' in passage.get('title', '').lower()
                )
                
                # Match act name - be flexible for explanatory passages
                # For explanatory/comparison passages, match if citation mentions any compared act
                act_matches = False
                if is_explanatory:
                    # Check if citation text mentions any of the compared acts
                    compares = meta.get('compares', [])
                    if compares:
                        for compared_act in compares:
                            if compared_act.lower() in citation_text.lower() or citation_text.lower() in compared_act.lower():
                                act_matches = True
                                break
                    # Also match if passage_act is empty or matches
                    if not act_matches:
                        act_matches = (
                            not passage_act  # empty act_name = Constitution passages
                            or citation_text.lower() in passage_act.lower()
                            or passage_act.lower() in citation_text.lower()
                            or 'constitution' in citation_text.lower() and not passage_act
                        )
                else:
                    act_matches = (
                        not passage_act  # empty act_name = Constitution passages
                        or citation_text.lower() in passage_act.lower()
                        or passage_act.lower() in citation_text.lower()
                        or 'constitution' in citation_text.lower() and not passage_act
                    )
                
                if is_explanatory and act_matches:
                    matching_passage = passage
                    break
            
            # If no explanatory passage found, try matching by act name only
            if not matching_passage:
                for passage in evidence_plan['passages']:
                    passage_act = passage.get('act_name', '')
                    meta = passage.get('metadata', {})
                    passage_art = str(meta.get('article_number', '') or '')
                    passage_sec = str(meta.get('section_number', '') or '')
                    
                    # Only match if passage has no article/section (document-level)
                    if not passage_art and not passage_sec:
                        act_matches = (
                            not passage_act
                            or citation_text.lower() in passage_act.lower()
                            or passage_act.lower() in citation_text.lower()
                            or ('constitution' in citation_text.lower() and not passage_act)
                        )
                        if act_matches:
                            matching_passage = passage
                            break
            
            if matching_passage:
                # Extract clean act name from citation text
                # For "Constitution of the Democratic Socialist Republic of Sri Lanka", use "Constitution"
                act_name = citation_text
                if 'constitution' in citation_text.lower():
                    # Extract just "Constitution" or the first significant word
                    words = citation_text.split()
                    if words:
                        # Use first word if it's a known act name, otherwise use "Constitution"
                        first_word = words[0]
                        if first_word.lower() in ['constitution', 'civil', 'penal', 'code']:
                            act_name = first_word
                        else:
                            act_name = 'Constitution'  # Default for Constitution-related citations
                elif 'civil procedure' in citation_text.lower():
                    act_name = 'Civil Procedure Code'
                elif 'penal' in citation_text.lower():
                    act_name = 'Penal Code'
                
                citations.append({
                    'act_name': act_name,
                    'section': 'Text',  # Use 'Text' as placeholder for document-level citations
                    'has_matching_evidence': True,
                    'passage_id': matching_passage['passage_id']
                })

        return citations

    def _create_abstention_response(
        self,
        query: str,
        reason: str
    ) -> Dict:
        """Create abstention response."""
        abstention_messages = {
            'insufficient_evidence': (
                "I cannot provide a reliable answer to this question based on "
                "the available legal materials. The retrieved passages do not "
                "contain sufficient evidence to address your query."
            ),
            'out_of_scope': (
                "This question falls outside the scope of the Sri Lankan "
                "Constitution and cannot be answered from the available corpus "
                "materials. Please consult a qualified Sri Lankan legal "
                "practitioner or refer to the relevant statutory source."
            ),
            'generation_error': (
                "An error occurred during answer generation. "
                "Please try rephrasing your question."
            )
        }

        message = abstention_messages.get(
            reason,
            "I cannot provide an answer at this time."
        )

        return {
            'query': query,
            'answer': message,
            'citations': [],
            'evidence_used': 0,
            'abstained': True,
            'abstention_reason': reason,
            'model': self.model_name
        }
