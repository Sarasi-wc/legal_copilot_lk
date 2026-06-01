"""
Query normalization module for preprocessing user queries.
Handles expansion, reformulation, and entity extraction.
"""

import re
from typing import Dict, List, Optional

from src.utils import get_logger

logger = get_logger(__name__)


class QueryNormalizer:
    """
    Normalizes and preprocesses legal queries for better retrieval.
    """

    def __init__(self):
        """Initialize query normalizer."""

        # Legal abbreviations and expansions
        self.abbreviations = {
            'sec': 'section',
            'sec.': 'section',
            's.': 'section',
            'ch': 'chapter',
            'ch.': 'chapter',
            'pt': 'part',
            'pt.': 'part',
            'para': 'paragraph',
            'para.': 'paragraph',
            'act': 'Act',
            'no': 'number',
            'no.': 'number'
        }

        # Legal domain terms
        self.legal_terms = [
            'criminal', 'civil', 'contract', 'tort', 'property',
            'evidence', 'procedure', 'penal', 'administrative',
            'constitutional', 'labor', 'labour'
        ]

        # Concept → statutory synonym expansions.
        # Keys are lowercase phrases found in user queries; values are
        # additional terms appended to the retrieval query so that BM25
        # and dense search can match the actual statutory language used
        # in the Constitution of Sri Lanka, which often differs from lay usage.
        self.legal_expansions = {
            # ── REPUBLIC & SOVEREIGNTY ──────────────────────────────────────
            'republic': ['Democratic Socialist Republic', 'unitary State', 'sovereignty of the People'],
            'sovereignty': ['sovereignty of the People', 'inalienable', 'legislative power', 'executive power', 'judicial power'],
            'supreme law': ['supreme law of Sri Lanka', 'inconsistent with the Constitution', 'void'],
            'separation of powers': ['legislative power', 'executive power', 'judicial power', 'vested in'],
            'national flag': ['Lion Flag', 'national flag of Sri Lanka', 'golden lion'],
            'national anthem': ['national anthem of Sri Lanka', 'Sri Lanka Matha'],
            'national day': ['fourth day of February', 'National Day'],

            # ── PRESIDENT ────────────────────────────────────────────────────
            'president': [
                'Head of State', 'Head of the Executive', 'Commander-in-Chief',
                'elected by the People', 'executive power of the People',
            ],
            'president election': [
                'election of the President', 'elected by the People', 'preferential vote',
                'more than one-half of the valid votes', 'Presidential election',
            ],
            'president term': [
                'term of office of the President', 'six years', 'holds office',
                'twice elected', 'not eligible for re-election',
            ],
            'president powers': [
                'executive power', 'Commander-in-Chief', 'appoint', 'summon prorogue dissolve',
                'assent to Bills', 'grant pardon',
            ],
            'presidential immunity': [
                'shall not be subject to the jurisdiction', 'immunity of the President',
                'while in office', 'not be instituted or continued',
            ],
            'impeachment': [
                'Article 38', 'removal from office of President',
                'resolution passed by not less than two-thirds of the whole number of Members',
                'misconduct or incapacity or corruption or abuse of power',
                'referred to the Supreme Court', 'intentional violation of the Constitution',
            ],
            'removal of president': [
                'Article 38',
                'mental or physical infirmity',
                'has been guilty of intentional violation',
                'the Supreme Court shall after due inquiry',
                'resolution of Parliament passed by not less than two-thirds',
                'misconduct or incapacity or corruption or abuse of power',
            ],
            'removing the president': [
                'Article 38',
                'mental or physical infirmity',
                'has been guilty of intentional violation',
                'the Supreme Court shall after due inquiry',
                'resolution of Parliament passed by not less than two-thirds',
            ],
            'acting president': [
                'Prime Minister shall act', 'functions of the President', 'absence or incapacity',
                'Speaker shall act as President',
            ],

            # ── PRIME MINISTER & CABINET ─────────────────────────────────────
            'prime minister': [
                'Prime Minister', 'Head of the Cabinet', 'appointed by the President',
                'member of Parliament who commands the confidence',
            ],
            'cabinet': [
                'Cabinet of Ministers', 'Council of Ministers', 'collectively responsible',
                'collective responsibility', 'Cabinet shall be charged with',
            ],
            'minister': [
                'Minister of the Cabinet', 'assigned to a Minister', 'appointed by the President',
                'on the advice of the Prime Minister',
            ],
            'government': [
                'Cabinet of Ministers', 'executive power', 'policy of the Government',
                'Council of Ministers', 'Prime Minister',
            ],

            # ── PARLIAMENT ───────────────────────────────────────────────────
            'parliament': [
                'member of Parliament', 'sitting of Parliament', 'dissolution of Parliament',
                'Parliament shall consist', 'elected members',
            ],
            'dissolution of parliament': [
                'summon prorogue and dissolve Parliament', 'Article 70',
                'by Proclamation dissolve Parliament', 'expiration of one year',
                'dissolution of Parliament by the President',
            ],
            'dissolve parliament': [
                'summon prorogue and dissolve Parliament', 'Article 70',
                'by Proclamation dissolve Parliament', 'expiration of one year',
            ],
            'prorogation': [
                'prorogued by the President', 'prorogation of Parliament',
                'Parliament stands prorogued',
            ],
            'quorum': [
                'fewer than twenty', 'number of members present', 'sitting of Parliament',
                'quorum of Parliament',
            ],
            'speaker': [
                'Speaker of Parliament', 'elected by the members of Parliament',
                'Speaker shall preside', 'Deputy Speaker',
            ],
            'member of parliament': [
                'elected on the basis of proportional representation',
                'disqualified to be elected', 'cease to be a member',
                'elected member of Parliament',
            ],
            'parliamentary session': [
                'meets once at least in every year', 'session of Parliament',
                'summoned to meet', 'first meeting of Parliament',
            ],
            'bill': [
                'Bill passed by Parliament', 'presented to the President for assent',
                'certified by the Speaker', 'urgent Bill',
            ],
            'constitutional amendment': [
                'Bill for the amendment or for the repeal and replacement',
                'two-thirds of the whole number of Members of Parliament',
                'approved by the People at a Referendum',
                'submitted to the People at a Referendum',
                'Article 83', 'Article 84',
                'Speaker certificate Article 79', 'bill becomes law Article 80',
            ],
            'amend the constitution': [
                'Bill for the amendment or for the repeal and replacement',
                'two-thirds of the whole number of Members of Parliament',
                'approved by the People at a Referendum', 'Article 83',
                'Speaker certificate Article 79', 'bill becomes law Article 80',
            ],

            # ── JUDICIARY ────────────────────────────────────────────────────
            'supreme court': [
                'jurisdiction of the Supreme Court', 'final court of appeal',
                'constitutional jurisdiction', 'fundamental rights jurisdiction',
                'Chief Justice',
            ],
            'court of appeal': [
                'jurisdiction of the Court of Appeal', 'appellate jurisdiction',
                'Court of Appeal shall', 'Judges of the Court of Appeal',
            ],
            'high court': [
                'High Court', 'trial by jury', 'indictment', 'jurisdiction of the High Court',
            ],
            'judicial appointments': [
                'appointed by the President', 'on the advice of the Constitutional Council',
                'Chief Justice', 'Judges of the Supreme Court',
            ],
            'judicial service commission': [
                'Judicial Service Commission', 'appointment transfer dismissal',
                'judicial officers', 'Commission shall consist',
            ],
            'attorney general': [
                'Attorney-General', 'legal advisor to the Government',
                'public prosecutions', 'appear in courts',
            ],

            # ── FUNDAMENTAL RIGHTS ───────────────────────────────────────────
            # "Article 10 Article 11 ..." triggers multi-article boost in
            # ArticleBoostedRetriever, injecting Arts 10-14 as exact matches.
            'fundamental rights': [
                'Article 10 Article 11 Article 12 Article 13 Article 14',
                'every person is entitled to freedom of thought conscience',
                'no person shall be subjected to torture cruel inhuman',
                'all persons are equal before the law no citizen shall be discriminated',
                'no person shall be arrested except according to procedure',
                'every citizen is entitled to freedom of speech and expression',
            ],
            'right to equality': ['equal protection', 'discrimination', 'no citizen shall be discriminated'],
            'freedom of speech': ['freedom of speech and expression', 'publication', 'freedom of assembly'],
            'freedom of religion': ['freedom of thought conscience', 'religion or belief', 'Article 10'],
            'right to life': ['deprived of life', 'liberty of the person', 'Article 13'],
            'right to education': ['entitled to the right of education', 'medium of instruction'],
            'freedom of movement': ['freedom of movement', 'choose his place of residence', 'Article 14'],
            'right to information': ['right to access to information', 'Article 14A'],
            'enforcement of fundamental rights': [
                'application to the Supreme Court', 'infringement of a fundamental right',
                'Article 126', 'relief or redress',
            ],

            # ── CITIZENSHIP ──────────────────────────────────────────────────
            'citizenship': [
                'citizen of Sri Lanka', 'citizenship by descent', 'citizenship by registration',
                'renunciation of citizenship',
            ],
            'citizen': [
                'citizen of Sri Lanka', 'dual citizenship', 'born in Sri Lanka',
                'parent was a citizen',
            ],

            # ── OFFICIAL LANGUAGE ────────────────────────────────────────────
            'official language': ['official language of Sri Lanka', 'Sinhala', 'Tamil', 'link language', 'Article 18', 'Article 22'],
            'language of administration': ['language of administration', 'Sinhala', 'Tamil', 'Article 18'],
            'national language': ['national languages of Sri Lanka', 'Sinhala', 'Tamil', 'Article 19'],

            # ── RELIGION ─────────────────────────────────────────────────────
            'buddhism': [
                'foremost place', 'Buddha Sasana', 'duty of the State to protect and foster',
                'Article 9',
            ],

            # ── ELECTIONS ────────────────────────────────────────────────────
            'elections': [
                'Commissioner of Elections', 'election of members', 'general election',
                'electoral list', 'proportional representation',
            ],
            'voting age': ['elector', 'eighteen years', 'qualified to be registered', 'electoral list'],
            'vote': ['elector', 'electoral', 'qualify to vote', 'eighteen years'],
            'voter': ['elector', 'qualified to be registered', 'electoral list'],
            'election commission': [
                'Commissioner of Elections', 'Elections Commission', 'conduct of elections',
                'delimitation of electoral districts',
            ],

            # ── FINANCE ──────────────────────────────────────────────────────
            'consolidated fund': [
                'Consolidated Fund of the Republic', 'charged on the Consolidated Fund',
                'appropriation', 'moneys received',
            ],
            'budget': [
                'Annual Financial Statement', 'Appropriation Bill', 'estimates of expenditure',
                'presented to Parliament', 'Minister of Finance',
            ],
            'taxation': [
                'tax', 'duty', 'levy', 'no taxation shall be imposed', 'Act of Parliament',
                'imposition of taxation',
            ],
            'auditor general': [
                'Auditor-General', 'audit of accounts', 'public accounts',
                'report to Parliament',
            ],
            'finance commission': [
                'Finance Commission', 'allocation of funds', 'Provincial Councils',
                'equitable share',
            ],

            # ── EMERGENCY ────────────────────────────────────────────────────
            'state of emergency': [
                'Proclamation of a state of emergency', 'emergency regulations',
                'public security', 'Article 155',
            ],
            'emergency powers': [
                'emergency regulations', 'public security ordinance',
                'approved by Parliament', 'cease to have effect',
            ],

            # ── PROVINCIAL COUNCILS ──────────────────────────────────────────
            'provincial council': [
                'Provincial Council', 'devolution of power', 'Governor of the Province',
                'Chief Minister', 'Board of Ministers',
            ],
            'devolution': [
                'devolution of power', 'Provincial Council', 'List I List II List III',
                'Concurrent List',
            ],

            # ── OMBUDSMAN ────────────────────────────────────────────────────
            'ombudsman': [
                'Parliamentary Commissioner for Administration', 'maladministration',
                'investigation of complaints', 'report to Parliament',
            ],

            # ── CONSTITUTIONAL COUNCIL ───────────────────────────────────────
            'constitutional council': [
                'Constitutional Council', 'independent commissions', 'appointment of',
                'National Police Commission', 'Human Rights Commission',
            ],

            # ── PUBLIC SERVICE ───────────────────────────────────────────────
            'public service commission': [
                'Public Service Commission', 'appointment transfer dismissal',
                'public officers', 'public service',
            ],
            'police commission': [
                'National Police Commission', 'appointment of police officers',
                'Inspector-General of Police',
            ],
        }

        # Article number → topic terms for bare "What is Article X" queries.
        # When a query mentions only an article number with no topical context,
        # these terms are appended so dense retrieval has something to match on.
        self.article_topics = {
            '1':   ['Democratic Socialist Republic free sovereign independent'],
            '2':   ['unitary State'],
            '3':   ['sovereignty of the People inalienable'],
            '4':   ['legislative power executive power judicial power fundamental rights'],
            '9':   ['Buddhism foremost place Buddha Sasana protect and foster'],
            '10':  ['freedom of thought conscience religion'],
            '11':  ['torture cruel inhuman degrading treatment punishment'],
            '12':  ['equality before the law equal protection discrimination'],
            '13':  ['freedom from arbitrary arrest detention personal liberty'],
            '14':  ['freedom of speech expression publication assembly association movement'],
            '14A': ['right to access to information'],
            '15':  ['restrictions on fundamental rights prescribed by law'],
            '16':  ['existing written law inconsistent with fundamental rights'],
            '17':  ['remedies for infringement of fundamental rights Supreme Court'],
            '18':  ['official language Sinhala'],
            '19':  ['national languages Sinhala Tamil'],
            '20':  ['Tamil language rights administration'],
            '22':  ['Tamil official language administration record proceedings'],
            '30':  ['President Head of State executive Commander-in-Chief elected'],
            '33':  ['powers duties functions of the President'],
            '38':  ['removal of President impeachment two-thirds majority'],
            '42':  ['Cabinet of Ministers collectively responsible Parliament'],
            '43':  ['appointment of Ministers Prime Minister'],
            '55':  ['Public Service Commission appointment transfer dismissal'],
            '70':  ['dissolution of Parliament summon prorogue'],
            '83':  ['entrenched provisions two-thirds majority referendum'],
            '89':  ['disqualification electors register'],
            '91':  ['disqualification Members of Parliament'],
            '99':  ['proportional representation electoral districts seats'],
            '105': ['Supreme Court Court of Appeal jurisdiction'],
            '107': ['removal of judges address Parliament'],
            '111': ['Court of Appeal jurisdiction appellate'],
            '126': ['fundamental rights application Supreme Court relief redress'],
            '129': ['advisory opinion Supreme Court President'],
            '138': ['Court of Appeal jurisdiction'],
            '154': ['Ombudsman Parliamentary Commissioner administration'],
            '155': ['public security emergency regulations'],
        }

    def normalize(self, query: str) -> Dict:
        """
        Normalize a query for retrieval.

        Args:
            query: Raw query string

        Returns:
            Dictionary with normalized query and metadata
        """
        logger.debug(f"Normalizing query: {query}")

        # Clean query
        cleaned = self._clean_query(query)

        # Expand abbreviations
        expanded = self._expand_abbreviations(cleaned)

        # Extract entities
        entities = self._extract_entities(expanded)

        # Detect query type
        query_type = self._detect_query_type(expanded)

        # Extract section references
        section_refs = self._extract_section_references(expanded)

        # Extract Act references
        act_refs = self._extract_act_references(expanded)

        # Expand statutory synonyms for retrieval
        retrieval_query = self._expand_legal_terms(expanded)

        # Append topic terms for bare "Article X" references
        retrieval_query = self._expand_bare_article_references(retrieval_query)

        result = {
            'original': query,
            'cleaned': cleaned,
            'normalized': expanded,
            'retrieval_query': retrieval_query,
            'entities': entities,
            'query_type': query_type,
            'section_refs': section_refs,
            'act_refs': act_refs
        }

        if retrieval_query != expanded:
            logger.debug(f"Expanded retrieval query: {retrieval_query}")

        logger.debug(f"Normalized result: {result}")
        return result

    def _clean_query(self, query: str) -> str:
        """Clean and standardize query text."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)

        # Trim
        query = query.strip()

        return query

    def _expand_abbreviations(self, query: str) -> str:
        """Expand common legal abbreviations."""
        words = query.split()
        expanded = []

        for word in words:
            lower_word = word.lower()
            if lower_word in self.abbreviations:
                expanded.append(self.abbreviations[lower_word])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def _expand_legal_terms(self, query: str) -> str:
        """
        Append statutory synonyms for concepts present in the query.

        This bridges the vocabulary gap between lay user language (e.g.
        "cheating", "voting age") and the actual statutory language used
        in Sri Lankan legislation (e.g. "dishonestly induces", "elector").
        The original query is preserved; expansions are appended so that
        BM25 and dense search see both the user's terms and the statute terms.
        """
        query_lower = query.lower()
        extra_terms: List[str] = []
        for concept, synonyms in self.legal_expansions.items():
            if concept in query_lower:
                extra_terms.extend(synonyms)

        if not extra_terms:
            return query

        # Deduplicate while preserving order
        seen = set()
        unique_extras = []
        for t in extra_terms:
            if t not in seen:
                seen.add(t)
                unique_extras.append(t)

        return query + ' ' + ' '.join(unique_extras)

    def _expand_bare_article_references(self, query: str) -> str:
        """
        Append topic terms when a query references an article number with little
        other topical content. Without this, dense retrieval has nothing to embed
        against for queries like "What is Article 12?" — the article number alone
        provides no semantic signal for the bi-encoder.
        """
        article_matches = re.findall(r'\bArticle\s+(\d+[A-Z]?)\b', query, re.IGNORECASE)
        if not article_matches:
            return query

        extra_terms = []
        for art_num in article_matches:
            if art_num in self.article_topics:
                extra_terms.extend(self.article_topics[art_num])

        if not extra_terms:
            return query

        return query + ' ' + ' '.join(extra_terms)

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal entities from query."""
        entities = {
            'sections': [],
            'acts': [],
            'years': [],
            'domains': []
        }

        # Extract section numbers
        section_pattern = r'\b[Ss]ection\s+(\d+[A-Z]?(?:\([a-z0-9]+\))?)\b'
        sections = re.findall(section_pattern, query)
        entities['sections'] = sections

        # Extract Act numbers
        act_pattern = r'Act\s+No\.\s*(\d+)\s+of\s+(\d{4})'
        acts = re.findall(act_pattern, query)
        entities['acts'] = [f"Act No. {num} of {year}" for num, year in acts]

        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, query)
        entities['years'] = years

        # Extract legal domains
        query_lower = query.lower()
        for term in self.legal_terms:
            if term in query_lower:
                entities['domains'].append(term)

        return entities

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type based on keywords and structure.

        Types:
            - factual: What/Who questions seeking facts
            - procedural: How/When questions about procedures
            - interpretive: Meaning/interpretation questions
            - cross_referenced: Questions involving multiple provisions
        """
        query_lower = query.lower()

        # Factual questions
        if re.search(r'\b(what|who|which|where)\b', query_lower):
            return 'factual'

        # Procedural questions
        if re.search(r'\b(how|when|process|procedure|step)\b', query_lower):
            return 'procedural'

        # Interpretive questions
        if re.search(r'\b(mean|means|interpret|definition|define)\b', query_lower):
            return 'interpretive'

        # Cross-reference questions
        if len(re.findall(r'[Ss]ection\s+\d+', query)) > 1:
            return 'cross_referenced'

        return 'general'

    def _extract_section_references(self, query: str) -> List[str]:
        """Extract section references from query."""
        pattern = r'[Ss]ection\s+(\d+[A-Z]?(?:\([a-z0-9]+\))?)'
        return re.findall(pattern, query)

    def _extract_act_references(self, query: str) -> List[Dict]:
        """Extract Act references from query."""
        pattern = r'([^(]+?)\s*\(Act\s+No\.\s*(\d+)\s+of\s+(\d{4})\)'
        matches = re.findall(pattern, query)

        return [
            {
                'name': name.strip(),
                'number': number,
                'year': int(year)
            }
            for name, number, year in matches
        ]
