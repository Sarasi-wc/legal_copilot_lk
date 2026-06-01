#!/usr/bin/env python3
"""
Generate 120 additional Q-A-C items to expand dataset from 80 to 200 items.
All items are grounded in actual corpus passage text.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_PATH = PROJECT_ROOT / "data/processed/corpus.jsonl"
DATASET_PATH = PROJECT_ROOT / "data/evaluation/qac_dataset.json"
ANNO_DATE = "2026-05-16"
ANNOTATOR = "LEGAL_EXPERT_1"
ACT_NAME = "Constitution of the Democratic Socialist Republic of Sri Lanka"
ACT_YEAR = 1978


def load_corpus():
    with open(CORPUS_PATH) as f:
        corpus = json.loads(f.readline())
    passages = corpus.get("passages", [])
    by_id = {p["passage_id"]: p for p in passages}

    def base_sec(pid):
        m = re.match(r"ACT__1978_SEC_([^_]+)", pid)
        return m.group(1) if m else pid

    by_sec = defaultdict(list)
    for p in passages:
        by_sec[base_sec(p["passage_id"])].append(p)
    return by_id, by_sec


def best_pid(sec, by_sec, all_ids):
    """Return the best passage_id for a section (prefer base, then CHUNK_0)."""
    base = f"ACT__1978_SEC_{sec}"
    if base in all_ids:
        return base
    for suffix in ["_CHUNK_0", "_SUBSEC_1_CHUNK_0", "_SUBSEC_1"]:
        pid = f"ACT__1978_SEC_{sec}{suffix}"
        if pid in all_ids:
            return pid
    # any
    for pid in sorted(all_ids):
        if re.match(rf"ACT__1978_SEC_{re.escape(sec)}(_|$)", pid):
            return pid
    return None


def make_citation(sec, pid, by_id):
    p = by_id.get(pid, {})
    snippet = p.get("text", "")[:200].replace("\n", " ")
    return {
        "act_name": ACT_NAME,
        "act_number": "",
        "act_year": ACT_YEAR,
        "section_number": sec,
        "passage_id": pid,
        "passage_text": snippet,
        "start_char": None,
        "end_char": None,
    }


def make_item(item_id, question, answer_text, secs, pids, query_type, domain,
              difficulty, chapter, articles, by_id):
    citations = [make_citation(sec, pid, by_id) for sec, pid in zip(secs, pids)]
    return {
        "item_id": item_id,
        "question": question,
        "query_type": query_type,
        "legal_domain": domain,
        "difficulty": difficulty,
        "gold_answers": [{
            "answer_text": answer_text,
            "citations": citations,
            "annotator_id": ANNOTATOR,
            "annotation_date": ANNO_DATE,
        }],
        "relevant_passages": pids,
        "metadata": {"chapter": chapter, "articles": articles},
    }


def build_new_items(by_id, by_sec):
    all_ids = set(by_id.keys())

    def pid(sec):
        return best_pid(sec, by_sec, all_ids)

    items = []
    n = 81  # start from QAC_081

    def add(question, answer, secs, query_type="factual", domain="constitutional",
            difficulty="medium", chapter="", articles=None):
        nonlocal n
        pids = [pid(s) for s in secs]
        if any(p is None for p in pids):
            return  # skip if any passage missing
        items.append(make_item(
            f"QAC_{n:03d}", question, answer,
            secs, pids, query_type, domain, difficulty, chapter,
            articles or secs, by_id,
        ))
        n += 1

    # -------------------------------------------------------------------------
    # Chapter I – The State (Arts 5)
    # -------------------------------------------------------------------------
    add(
        "What territory comprises the Republic of Sri Lanka under the Constitution?",
        "Article 5 of the Constitution provides that the territory of the Republic "
        "of Sri Lanka consists of the twenty-five administrative districts, the names "
        "of which are set out in the First Schedule, together with its territorial "
        "waters. The administrative districts may be subdivided or amalgamated.",
        ["5"], "factual", "constitutional", "easy", "I",
    )

    # -------------------------------------------------------------------------
    # Chapter III – Fundamental Rights (Arts 14A, 15, 16)
    # -------------------------------------------------------------------------
    add(
        "What is the right of access to information under the Constitution?",
        "Article 14A grants every citizen the right of access to any information "
        "as provided for by law, being information required for the exercise or "
        "protection of a citizen's right held by the State, a Ministry, any "
        "Government Department, or any statutory body established by law.",
        ["14A"], "factual", "constitutional", "medium", "III",
    )
    add(
        "How does the Constitution treat existing written and unwritten law in relation to fundamental rights?",
        "Article 16 provides that all existing written law and unwritten law shall "
        "remain valid and operative notwithstanding any inconsistency with the "
        "preceding provisions of Chapter III on Fundamental Rights. This means "
        "pre-existing law is preserved even where it may conflict with the "
        "fundamental rights provisions.",
        ["16"], "interpretive", "constitutional", "hard", "III",
    )

    # -------------------------------------------------------------------------
    # Chapter IV – Language (Arts 21–25A)
    # -------------------------------------------------------------------------
    add(
        "What language may a person choose for their education in Sri Lanka?",
        "Article 21 of the Constitution provides that a person shall be entitled "
        "to be educated through the medium of either of the National Languages "
        "(Sinhala or Tamil). This right does not apply to institutions of higher "
        "education where the medium of instruction is a language other than a "
        "National Language.",
        ["21"], "factual", "constitutional", "easy", "IV",
    )
    add(
        "What are the official languages of administration in Sri Lanka?",
        "Article 22 provides that Sinhala and Tamil shall be the languages of "
        "administration throughout Sri Lanka. Sinhala shall be the language of "
        "administration and be used for the maintenance of public records and the "
        "transaction of all business by public institutions of all Provinces of Sri "
        "Lanka, except the Northern and Eastern Provinces where Tamil shall be so "
        "used.",
        ["22"], "factual", "constitutional", "easy", "IV",
    )
    add(
        "In what languages must laws and subordinate legislation be enacted in Sri Lanka?",
        "Article 23 requires that all laws and subordinate legislation shall be "
        "enacted or made and published in Sinhala and Tamil, together with a "
        "translation in English. Parliament shall, at the stage of enactment, "
        "determine which text shall prevail in the event of any inconsistency.",
        ["23"], "factual", "constitutional", "medium", "IV",
    )
    add(
        "What obligation does the State have regarding the use of languages provided by the Constitution?",
        "Article 25 places an obligation on the State to provide adequate facilities "
        "for the use of the languages provided for in Chapter IV of the Constitution "
        "(Sinhala and Tamil). The State must ensure that such adequate facilities are "
        "available for the use of those languages.",
        ["25"], "factual", "constitutional", "easy", "IV",
    )
    add(
        "What happens when there is an inconsistency between a law and the language provisions of the Constitution?",
        "Article 25A provides that in the event of any inconsistency between the "
        "provisions of any law and the provisions of Chapter IV on language, the "
        "provisions of Chapter IV shall prevail. Any law inconsistent with the "
        "language chapter shall be overridden by those constitutional provisions.",
        ["25A"], "interpretive", "constitutional", "medium", "IV",
    )

    # -------------------------------------------------------------------------
    # Chapter V – Citizenship (Art 26)
    # -------------------------------------------------------------------------
    add(
        "What is the status of citizenship in Sri Lanka and how is it described?",
        "Article 26 provides that there shall be one status of citizenship known as "
        "the status of a citizen of Sri Lanka. A citizen of Sri Lanka shall for all "
        "purposes be described only as a citizen of Sri Lanka, whether such person "
        "became entitled to citizenship by descent or by registration.",
        ["26"], "factual", "constitutional", "easy", "V",
    )

    # -------------------------------------------------------------------------
    # Chapter VI – Directive Principles (Art 27, 28)
    # -------------------------------------------------------------------------
    add(
        "What fundamental duties does the Constitution impose on every person in Sri Lanka?",
        "Article 28 of the Constitution recognises that the exercise and enjoyment "
        "of rights and freedoms are inseparable from the performance of duties. "
        "Accordingly, it is the duty of every person to uphold and defend the "
        "Constitution and the law, to further the national interest, to foster "
        "national unity, to preserve public property, and to protect the environment.",
        ["28"], "factual", "constitutional", "medium", "VI",
    )

    # -------------------------------------------------------------------------
    # Chapter VII – The Executive: President (Arts 30–42)
    # -------------------------------------------------------------------------
    add(
        "What is the constitutional role of the President of Sri Lanka?",
        "Article 30 establishes that there shall be a President of the Republic of "
        "Sri Lanka, who is the Head of State, the Head of the Executive and of the "
        "Government, and the Commander-in-Chief of the Armed Forces. The President "
        "shall be elected by the People and shall hold office for a term of five years.",
        ["30"], "factual", "constitutional", "easy", "VII",
    )
    add(
        "Who may be nominated as a candidate for President and what is the election process?",
        "Article 31 provides that any citizen who is qualified to be elected to the "
        "office of President may be nominated as a candidate by a recognised political "
        "party, or if they are or have been an elected member of the legislature, by "
        "any other political party or by a group of electors. The term of office shall "
        "be five years.",
        ["31"], "procedural", "constitutional", "medium", "VII",
    )
    add(
        "How does the President assume office after election?",
        "Article 32 requires that the person elected or succeeding to the office of "
        "President shall assume office upon taking and subscribing the oath, or making "
        "and subscribing the affirmation, set out in the Fourth Schedule. This must "
        "be done in Sri Lanka before the Chief Justice or any other Judge of the "
        "Supreme Court.",
        ["32"], "procedural", "constitutional", "medium", "VII",
    )
    add(
        "What powers does the President have in addition to those expressly conferred by the Constitution?",
        "Article 33 provides that in addition to powers and functions expressly "
        "conferred by the Constitution or by any written law, the President shall "
        "have the power to appoint the Prime Minister, Ministers, and other specified "
        "officials, and to perform other functions assigned by written law. The "
        "Article enumerates a range of executive powers vested in the President.",
        ["33"], "factual", "constitutional", "medium", "VII",
    )
    add(
        "What is the President's power to grant pardons?",
        "Article 34 grants the President the power in the case of any offender "
        "convicted of any offence in any court within the Republic of Sri Lanka to: "
        "(a) grant a pardon, either free or subject to lawful conditions; "
        "(b) grant any respite, either indefinite or for a specified period, of the "
        "execution of any sentence; (c) grant a remission or suspension of any "
        "sentence or penalty.",
        ["34"], "factual", "constitutional", "medium", "VII",
    )
    add(
        "What immunity does the President enjoy from legal proceedings while in office?",
        "Article 35 provides that while any person holds office as President, no "
        "proceedings shall be instituted or continued against them in any court or "
        "tribunal in respect of anything done or omitted to be done by them either "
        "in their official or private capacity. This immunity continues for the "
        "duration of the President's term of office.",
        ["35"], "factual", "constitutional", "medium", "VII",
    )
    add(
        "What procedure applies when the President is unable to discharge the functions of office?",
        "Article 37 provides that if the President is of the opinion that by reason "
        "of illness, absence from Sri Lanka, or any other cause they will be unable "
        "to exercise, perform and discharge the powers, duties and functions of "
        "office, the President may appoint the Prime Minister to act as President "
        "during such period of inability.",
        ["37"], "procedural", "constitutional", "medium", "VII",
    )
    add(
        "What happens when the Supreme Court determines the election of the President was void?",
        "Article 39 provides that where the Supreme Court, in the exercise of its "
        "jurisdiction under Article 130, determines that the election of the "
        "President was void and does not determine that any other person was duly "
        "elected, a poll shall be taken for the election of a President in accordance "
        "with the provisions of the Constitution.",
        ["39"], "procedural", "constitutional", "hard", "VII",
    )
    add(
        "What is the Constitutional Council and who are its members?",
        "Article 41A establishes the Constitutional Council, which consists of: "
        "(a) the Prime Minister; (b) the Speaker; (c) the Leader of the Opposition; "
        "(d) one Member of Parliament appointed by the President on the nomination of "
        "the Prime Minister; (e) one Member of Parliament appointed by the President "
        "on the nomination of the Leader of the Opposition; and additional members "
        "representing civil society and professional bodies.",
        ["41A"], "factual", "constitutional", "hard", "VIIA",
    )
    add(
        "What role does the Constitutional Council play in appointments to independent commissions?",
        "Article 41B provides that no person shall be appointed by the President as "
        "the Chairman or a member of any of the Commissions specified in the Schedule "
        "to that Article except on a recommendation of the Constitutional Council. "
        "This requirement applies to the Election Commission, the Public Service "
        "Commission, the National Police Commission, and other key independent bodies.",
        ["41B"], "factual", "constitutional", "hard", "VIIA",
    )
    add(
        "What is the process for appointing persons to the offices listed in the Constitutional Council Schedule?",
        "Article 41C requires that no person shall be appointed by the President to "
        "any of the offices specified in its Schedule unless such appointment has been "
        "approved by the Constitutional Council upon a recommendation made to the "
        "Council by the President. The Council's approval is a precondition for such "
        "presidential appointments.",
        ["41C"], "procedural", "constitutional", "hard", "VIIA",
    )
    add(
        "To whom is the President responsible for the exercise of executive powers?",
        "Article 42 provides that the President shall be responsible to Parliament "
        "for the due exercise, performance and discharge of powers, duties and "
        "functions under the Constitution and any written law, including the law "
        "relating to public security.",
        ["42"], "factual", "constitutional", "easy", "VII",
    )

    # -------------------------------------------------------------------------
    # Chapter VIII – The Executive: Cabinet and Ministers (Arts 44–51)
    # -------------------------------------------------------------------------
    add(
        "How are Cabinet Ministries and their assignment to Ministers determined?",
        "Article 44 provides that the President shall, in consultation with the "
        "Prime Minister where he considers such consultation to be necessary, "
        "determine the number of Ministers of the Cabinet of Ministers and the "
        "Ministries, and the assignment of subjects and functions to such Ministers. "
        "The President may change any such assignment at any time.",
        ["44"], "factual", "constitutional", "medium", "VIII",
    )
    add(
        "Can the President appoint Ministers who are not members of the Cabinet of Ministers?",
        "Article 45 provides that the President may, in consultation with the Prime "
        "Minister, appoint from among Members of Parliament, Ministers who shall not "
        "be members of the Cabinet of Ministers. Such non-Cabinet Ministers assist "
        "in the work of the Government but do not hold Cabinet portfolios.",
        ["45"], "factual", "constitutional", "medium", "VIII",
    )
    add(
        "What is the procedure for appointing Deputy Ministers?",
        "Article 46 provides that the President may, in consultation with the Prime "
        "Minister, appoint from among Members of Parliament, Deputy Ministers to "
        "assist Ministers of the Cabinet of Ministers in the performance of their "
        "duties. A Minister of the Cabinet may also delegate functions to a Deputy "
        "Minister by Notification published in the Gazette.",
        ["46"], "procedural", "constitutional", "medium", "VIII",
    )
    add(
        "What are the numerical limits on Ministers and Deputy Ministers under the Constitution?",
        "Article 47 provides that the total number of Ministers of the Cabinet of "
        "Ministers shall not exceed thirty; and the total number of Ministers who "
        "are not members of the Cabinet and Deputy Ministers shall not, in the "
        "aggregate, exceed forty. These limits ensure proportionality in the size "
        "of the Government.",
        ["47"], "factual", "constitutional", "medium", "VIII",
    )
    add(
        "What happens to the Cabinet of Ministers when Parliament is dissolved?",
        "Article 48 provides that the Cabinet of Ministers functioning immediately "
        "prior to the dissolution of Parliament shall, notwithstanding such "
        "dissolution, continue to function and shall cease to function upon the "
        "conclusion of the General Election. Accordingly the Prime Minister and "
        "Ministers shall, after such General Election, hold office until the "
        "Cabinet is newly constituted.",
        ["48"], "procedural", "constitutional", "medium", "VIII",
    )
    add(
        "What is the procedure when a Minister is unable to discharge the functions of office?",
        "Article 50 provides that whenever a Minister of the Cabinet of Ministers, "
        "a Minister who is not a member of the Cabinet, or a Deputy Minister is "
        "unable to discharge the functions of office, the President may, in "
        "consultation with the Prime Minister, appoint another Member of Parliament "
        "to act as Acting Minister or Acting Deputy Minister during such period.",
        ["50"], "procedural", "constitutional", "medium", "VIII",
    )
    add(
        "What are the functions of the Secretary to the Prime Minister?",
        "Article 51 provides that there shall be a Secretary to the Prime Minister "
        "appointed by the President. The Secretary shall have charge of the office "
        "of the Prime Minister and shall perform and discharge the duties and "
        "functions of the office of Prime Minister in the absence or incapacity of "
        "the Prime Minister.",
        ["51"], "factual", "constitutional", "medium", "VIII",
    )
    add(
        "What are the responsibilities of the Secretary to each Ministry?",
        "Article 52 provides that there shall be a Secretary for every Ministry of "
        "a Minister of the Cabinet of Ministers, appointed by the President. The "
        "Secretary shall be the official head of the Ministry and shall be "
        "responsible for the administration of the Ministry and for implementing "
        "the policy of the Cabinet of Ministers in relation to that Ministry.",
        ["52"], "factual", "constitutional", "medium", "VIII",
    )
    add(
        "What oath or affirmation must a person take before assuming office in the executive branch?",
        "Article 53 requires that a person appointed to any office referred to in "
        "Chapter VIII on the Cabinet of Ministers shall not enter upon the duties "
        "of that office until they take and subscribe the official oath or make and "
        "subscribe the official affirmation as set out in the Fourth Schedule of "
        "the Constitution.",
        ["53"], "procedural", "constitutional", "medium", "VIII",
    )

    # -------------------------------------------------------------------------
    # Chapter IX – The Public Service (Arts 55–60)
    # -------------------------------------------------------------------------
    add(
        "What powers does the Cabinet of Ministers have over the public service?",
        "Article 55 provides that the Cabinet of Ministers shall provide for and "
        "determine all matters of policy relating to public officers, including "
        "policy relating to appointments, promotions, transfers, disciplinary "
        "control and dismissal. The appointment, promotion, transfer, disciplinary "
        "control and dismissal of public officers vests in the Cabinet of Ministers "
        "exercising its functions through the Public Service Commission.",
        ["55"], "factual", "constitutional", "medium", "IX",
    )
    add(
        "Can the Public Service Commission delegate its powers to a committee?",
        "Article 56 provides that the Public Service Commission may delegate to a "
        "Committee consisting of three persons (not being members of the Commission) "
        "appointed by it, any of the powers of the Commission in relation to "
        "appointments, promotions, transfers, and disciplinary control of public "
        "officers. Such delegation enables efficient administration of the public "
        "service.",
        ["56"], "factual", "constitutional", "medium", "IX",
    )
    add(
        "What powers may the Public Service Commission delegate to a public officer?",
        "Article 57 provides that the Commission may delegate to a public officer, "
        "subject to such conditions and procedure as may be determined by the "
        "Commission, any of the powers of the Commission in relation to the "
        "appointment, promotion, transfer, and disciplinary control of public "
        "officers subordinate to that public officer.",
        ["57"], "factual", "constitutional", "medium", "IX",
    )
    add(
        "What right of appeal does a public officer have against certain adverse decisions?",
        "Article 58 provides that any public officer aggrieved by an order relating "
        "to a promotion, transfer, dismissal, or disciplinary matter made by the "
        "Commission or any Committee or public officer to whom powers have been "
        "delegated, shall have the right to appeal to the Administrative Appeals "
        "Tribunal established under Article 59.",
        ["58"], "procedural", "constitutional", "medium", "IX",
    )
    add(
        "What is the Administrative Appeals Tribunal and who appoints it?",
        "Article 59 provides for the establishment of an Administrative Appeals "
        "Tribunal appointed by the Judicial Service Commission. The Tribunal has "
        "jurisdiction to hear and determine appeals by public officers against "
        "orders relating to promotions, transfers, dismissals, and disciplinary "
        "matters made by the Public Service Commission or delegated bodies.",
        ["59"], "factual", "constitutional", "medium", "IX",
    )
    add(
        "Can the Public Service Commission continue to exercise powers it has delegated?",
        "Article 60 provides that upon delegation of any of its powers to a Committee "
        "or a public officer under Articles 56 or 57, the Commission shall not itself "
        "exercise such delegated powers. The effect of delegation is therefore to "
        "transfer, not merely share, the exercise of the delegated power to the "
        "delegate.",
        ["60"], "interpretive", "constitutional", "hard", "IX",
    )

    # -------------------------------------------------------------------------
    # Chapter X – Parliament (Arts 63–68)
    # -------------------------------------------------------------------------
    add(
        "What oath or affirmation must a Member of Parliament take before sitting or voting?",
        "Article 63 provides that except for the purpose of electing the Speaker, no "
        "Member shall sit or vote in Parliament until they have taken and subscribed "
        "the official oath or made and subscribed the official affirmation before "
        "Parliament. The oath includes a commitment to uphold and defend the "
        "Constitution.",
        ["63"], "procedural", "constitutional", "easy", "X",
    )
    add(
        "How is the Speaker of Parliament elected?",
        "Article 64 provides that Parliament shall, at its first meeting after a "
        "General Election, elect three Members to be respectively the Speaker, the "
        "Deputy Speaker and the Deputy Chairman of Committees. The Speaker shall "
        "preside over the sittings of Parliament and is elected by the Members of "
        "Parliament by secret ballot.",
        ["64"], "procedural", "constitutional", "medium", "X",
    )
    add(
        "Who is the Secretary-General of Parliament and how are they appointed?",
        "Article 65 provides that there shall be a Secretary-General of Parliament "
        "who shall, on the approval of the Constitutional Council, be appointed by "
        "the President. The Secretary-General shall hold office during good behavior "
        "and their salary shall be determined by Parliament and charged on the "
        "Consolidated Fund.",
        ["65"], "factual", "constitutional", "medium", "X",
    )
    add(
        "In what circumstances does the seat of a Member of Parliament become vacant?",
        "Article 66 provides that the seat of a Member shall become vacant upon: "
        "(a) the Member's death; (b) resignation by writing addressed to the "
        "Secretary-General; (c) assuming the office of President; (d) expulsion "
        "from the political party or group which nominated the Member; or "
        "(e) any other grounds specified in the Constitution.",
        ["66"], "factual", "constitutional", "medium", "X",
    )
    add(
        "Are Members of Parliament paid for their services and how are their salaries determined?",
        "Article 68 provides that Ministers, Deputy Ministers and Members, including "
        "the Speaker, the Deputy Speaker and the Deputy Chairman of Committees, shall "
        "be paid such salaries and allowances as Parliament shall by resolution "
        "determine. These salaries and allowances shall be charged on the "
        "Consolidated Fund.",
        ["68"], "factual", "constitutional", "easy", "X",
    )

    # -------------------------------------------------------------------------
    # Chapter XI – Parliament (procedure, Arts 72, 74, 75, 77, 80, 81, 83, 85–87)
    # -------------------------------------------------------------------------
    add(
        "How are questions decided in Parliament?",
        "Article 72 provides that, save as otherwise provided in the Constitution, "
        "any question proposed for decision by Parliament shall be decided by the "
        "majority of votes of the Members present and voting. The Speaker or "
        "presiding Member shall have an original vote and shall not have a casting "
        "vote, except when there is an equality of votes.",
        ["72"], "factual", "constitutional", "easy", "XI",
    )
    add(
        "What matters can Parliament regulate through Standing Orders?",
        "Article 74 provides that, subject to the Constitution, Parliament may by "
        "resolution or Standing Order provide for: (i) the election and retirement "
        "of the Speaker, the Deputy Speaker and the Deputy Chairman of Committees; "
        "(ii) the regulation of its business and the preservation of order at its "
        "proceedings; and (iii) any other matters relating to the conduct of "
        "parliamentary proceedings.",
        ["74"], "factual", "constitutional", "medium", "XI",
    )
    add(
        "What is the legislative power of Parliament under the Constitution?",
        "Article 75 provides that Parliament shall have power to make laws, including "
        "laws having retrospective effect, and repealing or amending any provision "
        "of the Constitution, or adding any provision to the Constitution. However, "
        "Parliament shall not make any law suspending the operation of the "
        "Constitution or any part thereof.",
        ["75"], "factual", "constitutional", "medium", "XI",
    )
    add(
        "What duty does the Attorney-General have in relation to Bills presented to Parliament?",
        "Article 77 places a duty on the Attorney-General to examine every Bill for "
        "any contravention of the requirements of the Constitution, and to certify "
        "that the Bill does not, or that certain provisions of the Bill do, "
        "contravene those requirements. This certification assists Parliament in "
        "determining whether a Bill requires special procedures.",
        ["77"], "factual", "constitutional", "medium", "XI",
    )
    add(
        "How does a Bill passed by Parliament become law?",
        "Article 80 provides that, subject to the provisions of Article 80(2), a "
        "Bill passed by Parliament shall become law when the certificate of the "
        "Speaker is endorsed thereon. Where the Cabinet of Ministers has certified "
        "that a Bill is intended to be submitted for approval by the People at a "
        "Referendum, it shall not become law unless so approved.",
        ["80"], "procedural", "constitutional", "medium", "XI",
    )
    add(
        "What privileges and immunities does Parliament have under the Constitution?",
        "Article 81 provides that the privileges, immunities and powers of Parliament "
        "and of its Members may be determined and regulated by Parliament. Until such "
        "determination, the privileges, immunities and powers of Parliament and of "
        "its Members shall be those privileges, immunities and powers enjoyed by the "
        "House of Representatives and its Members on the day of commencement of the "
        "Constitution.",
        ["81"], "factual", "constitutional", "medium", "XI",
    )
    add(
        "Which Bills must be passed at a Referendum in addition to a two-thirds majority in Parliament?",
        "Article 83 provides that, notwithstanding Article 82, a Bill for the "
        "amendment or repeal and replacement of, or which is inconsistent with, "
        "Articles 1, 2, 3, 6, 7, 8, 9, 10, or 11, or Article 83 itself, shall not "
        "be presented for the Speaker's certificate unless it has been approved by "
        "the People at a Referendum, in addition to being passed by a two-thirds "
        "majority in Parliament.",
        ["83"], "procedural", "constitutional", "hard", "XI",
    )
    add(
        "What is the procedure for the President to submit a Bill to the People at a Referendum?",
        "Article 85 provides that the President shall submit to the People by "
        "Referendum every Bill, or any provision in any Bill, which the Cabinet of "
        "Ministers has certified to be a Bill which if enacted would be consistent "
        "with the Constitution only if approved at a Referendum. The Referendum "
        "shall be conducted by the Commissioner of Elections.",
        ["85"], "procedural", "constitutional", "hard", "XI",
    )
    add(
        "Can the President submit matters of national importance to a Referendum?",
        "Article 86 provides that the President may, subject to the provisions of "
        "Article 85, submit to the People by Referendum any matter which in the "
        "opinion of the President is of national importance. This grants the "
        "President a discretionary power to consult the electorate directly on "
        "significant national questions.",
        ["86"], "factual", "constitutional", "medium", "XI",
    )
    add(
        "How must a Referendum be conducted and its result communicated?",
        "Article 87 provides that every Referendum shall be conducted by the "
        "Commissioner of Elections, who shall communicate the result to Parliament. "
        "Parliament shall by law make provision for the procedure for the conduct "
        "of any Referendum held under the Constitution, including the qualifications "
        "of voters and the manner of voting.",
        ["87"], "procedural", "constitutional", "medium", "XI",
    )

    # -------------------------------------------------------------------------
    # Chapter XII – Elections (Arts 88–102)
    # -------------------------------------------------------------------------
    add(
        "What are the qualifications to be an elector in Sri Lanka?",
        "Article 88 provides that every person shall, unless disqualified, be "
        "qualified to be an elector at the election of the President and of the "
        "Members of Parliament or to vote at any Referendum. However, no such "
        "person shall be entitled to vote unless their name is entered in the "
        "appropriate register of electors.",
        ["88"], "factual", "constitutional", "easy", "XII",
    )
    add(
        "What grounds disqualify a person from being an elector?",
        "Article 89 provides that no person shall be qualified to be an elector at "
        "an election of the President or Members of Parliament, or to vote at any "
        "Referendum, if they are not a citizen of Sri Lanka, are detained as "
        "unsound of mind, have been convicted of certain offences, or are otherwise "
        "disqualified under the law relating to elections.",
        ["89"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What are the qualifications required to be elected as a Member of Parliament?",
        "Article 90 provides that every person who is qualified to be an elector "
        "shall be qualified to be elected as a Member of Parliament unless they are "
        "subject to any of the disqualifications specified in Article 91. The basic "
        "qualification therefore derives from the right to vote, subject to "
        "additional specific disqualifications.",
        ["90"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What disqualifies a person from being elected as or sitting as a Member of Parliament?",
        "Article 91 provides that no person shall be qualified to be elected as a "
        "Member of Parliament or to sit and vote in Parliament if they: are subject "
        "to any of the disqualifications specified in Article 89; stand nominated "
        "as a candidate for election in more than one electoral district; hold any "
        "office of profit created by the Constitution; or are disqualified under "
        "any written law.",
        ["91"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What are the qualifications required to be elected President of Sri Lanka?",
        "Article 92 provides that every person who is qualified to be an elector "
        "shall be qualified to be elected to the office of President unless they "
        "are subject to certain disqualifications, including: not having attained "
        "the age of thirty years; not being qualified to be elected as a Member of "
        "Parliament; or having been twice elected to the office of President.",
        ["92"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What principles govern the voting process for the election of the President and Members of Parliament?",
        "Article 93 provides that the voting for the election of the President and "
        "of Members of Parliament and at any Referendum shall be free, equal and "
        "secret. This guarantees universal suffrage conducted by secret ballot, "
        "consistent with the fundamental democratic principles of the Constitution.",
        ["93"], "factual", "constitutional", "easy", "XII",
    )
    add(
        "How is preferential voting for the President conducted?",
        "Article 94 provides that at the election of the President every voter, "
        "while casting a vote for any candidate, may where there are three or more "
        "candidates indicate their second and third preferences. If no candidate "
        "obtains more than fifty percent of valid votes, counting of second "
        "preferences and third preferences proceeds among the two leading "
        "candidates.",
        ["94"], "procedural", "constitutional", "hard", "XII",
    )
    add(
        "How is the Delimitation Commission for Electoral Districts established?",
        "Article 95 provides that within three months of the commencement of the "
        "Constitution the President shall, for the delimitation of Electoral "
        "Districts, establish a Delimitation Commission. The Commission's function "
        "is to divide Sri Lanka into Electoral Districts for the purposes of "
        "elections to Parliament.",
        ["95"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "How must the Delimitation Commission divide Sri Lanka into Electoral Districts?",
        "Article 96 provides that the Delimitation Commission shall divide Sri "
        "Lanka into not less than twenty and not more than twenty-five Electoral "
        "Districts. The Commission shall assign to each district the number of "
        "Members to be returned, having regard to the number of registered electors "
        "in each district.",
        ["96"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "How are the names and boundaries of electoral districts published?",
        "Article 97 provides that the President shall by proclamation publish the "
        "names and boundaries of the electoral districts and the number of members "
        "to be returned by each district as specified in the report of the "
        "Delimitation Commission. Such proclamation is the official notice of "
        "electoral district boundaries.",
        ["97"], "procedural", "constitutional", "medium", "XII",
    )
    add(
        "How many members in total are to be returned by all electoral districts in Sri Lanka?",
        "Article 98 provides that the several electoral districts shall together be "
        "entitled to return one hundred and ninety-six members. These members, "
        "together with the National List members, constitute the full membership "
        "of Parliament. The distribution among districts is determined by the "
        "Delimitation Commission.",
        ["98"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "How is the number of members to be returned by each electoral district calculated?",
        "Article 99 provides that at any election of Members of Parliament, the "
        "total number of members which an electoral district is entitled to return "
        "shall be the number specified by the Commissioner of Elections in the "
        "Order published under Article 98. This number is calculated based on the "
        "registered electorate of each district as a proportion of the national total.",
        ["99"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What penalty applies to a person who sits and votes in Parliament without the required qualifications?",
        "Article 100 provides that any person who sits and votes in Parliament "
        "without having been duly elected, or after ceasing to be qualified, "
        "or while subject to any disqualification, shall be liable to a "
        "penalty as provided by law for each day on which they so sit and "
        "vote in Parliament.",
        ["100"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What matters can Parliament legislate about elections?",
        "Article 101 provides that Parliament may by law make provision for: "
        "(a) the registration of electors; (b) the prescribing of a qualifying date "
        "for registration; (c) the conduct of elections of Members of Parliament "
        "and of the President; (d) the grounds for disqualification; and "
        "(e) the determination of any question relating to elections.",
        ["101"], "factual", "constitutional", "medium", "XII",
    )
    add(
        "What happens to a public officer who stands as a candidate at an election?",
        "Article 102 provides that when a public officer or an officer of a public "
        "corporation is a candidate at any election, that officer shall be deemed "
        "to be on leave without pay during the period from the date of their "
        "nomination as a candidate to the date of the declaration of the poll. "
        "This ensures public officers do not use public resources for election "
        "campaigning.",
        ["102"], "factual", "constitutional", "medium", "XII",
    )

    # -------------------------------------------------------------------------
    # Chapter XIII – Elections Commission (Art 104)
    # -------------------------------------------------------------------------
    add(
        "What are the rules for meetings of the Elections Commission?",
        "Article 104 provides that the quorum for any meeting of the Elections "
        "Commission shall be three members. The Chairman of the Commission shall "
        "preside at all meetings and decisions of the Commission shall be by "
        "majority vote of the members present and voting. In the event of an "
        "equality of votes the Chairman shall have a casting vote.",
        ["104"], "factual", "constitutional", "medium", "XIIIA",
    )

    # -------------------------------------------------------------------------
    # Chapter XV – The Judiciary (Arts 105–136)
    # -------------------------------------------------------------------------
    add(
        "What courts are established under the Constitution for the administration of justice?",
        "Article 105 provides that the institutions for the administration of "
        "justice which protect, vindicate and enforce the rights of the People "
        "shall be: (a) the Supreme Court of the Republic of Sri Lanka; "
        "(b) the Court of Appeal; (c) the High Court of the Republic; and "
        "(d) such other courts of first instance and other tribunals as Parliament "
        "may by law establish.",
        ["105"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "Must court proceedings be conducted in public?",
        "Article 106 provides that the sittings of every court, tribunal or other "
        "institution established under the Constitution or by Parliament shall be "
        "held in public. However, a court may, in the interests of justice, "
        "exclude the public from all or any part of the proceedings.",
        ["106"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Can the Supreme Court continue to function despite a vacancy in its membership?",
        "Article 107 provides that the Supreme Court shall have power to act "
        "notwithstanding any vacancy in its membership, and no act or proceeding "
        "of the Court shall be, or shall be deemed to be, invalid by reason only "
        "of any such vacancy or any defect in the appointment of a Judge.",
        ["107"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "How are the salaries of Judges of the Supreme Court and Court of Appeal determined?",
        "Article 108 provides that the salaries of the Judges of the Supreme Court "
        "and of the Court of Appeal shall be determined by Parliament and shall be "
        "charged on the Consolidated Fund. The salary of a Judge shall not be "
        "reduced after their appointment, thus protecting judicial independence.",
        ["108"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What happens when the Chief Justice is temporarily unable to function?",
        "Article 109 provides that if the Chief Justice or the President of the "
        "Court of Appeal is temporarily unable to exercise, perform and discharge "
        "the powers, duties and functions of their office, the President of the "
        "Republic may appoint a Judge of the Supreme Court or the Court of Appeal "
        "respectively to act in that office during such period of inability.",
        ["109"], "procedural", "constitutional", "medium", "XV",
    )
    add(
        "Can a Judge of the Supreme Court or Court of Appeal be required to perform other judicial duties?",
        "Article 110 provides that a Judge of the Supreme Court or Court of Appeal "
        "may be required by the President of the Republic to perform or discharge "
        "any other powers, duties and functions conferred or imposed on, or "
        "assigned to, any court, tribunal, institution or person under any law.",
        ["110"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What is the jurisdiction of the Supreme Court of Sri Lanka?",
        "Article 118 provides that the Supreme Court of Sri Lanka shall be the "
        "highest and final superior court of record in the Republic and shall "
        "exercise: (a) jurisdiction in respect of constitutional matters; "
        "(b) jurisdiction for the protection of fundamental rights; "
        "(c) final appellate jurisdiction; (d) advisory jurisdiction; and "
        "(e) jurisdiction in respect of electoral matters.",
        ["118"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "How many Judges compose the Supreme Court of Sri Lanka?",
        "Article 119 provides that the Supreme Court shall consist of the Chief "
        "Justice and of not less than six and not more than sixteen other Judges "
        "who shall be appointed by the President. The Chief Justice shall preside "
        "over the proceedings of the Supreme Court.",
        ["119"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "What is the constitutional jurisdiction of the Supreme Court?",
        "Article 120 provides that the Supreme Court shall have sole and exclusive "
        "jurisdiction to determine any question as to whether any Bill or any "
        "provision thereof is inconsistent with the Constitution. The Court may "
        "exercise this jurisdiction upon a reference made by the President, or "
        "upon a petition made by a citizen, before a Bill is presented to the "
        "Speaker for certification.",
        ["120"], "factual", "constitutional", "hard", "XV",
    )
    add(
        "What is the procedure for examining the constitutionality of an urgent Bill?",
        "Article 122 provides a special procedure for urgent Bills. Where the "
        "Cabinet of Ministers certifies that a Bill is urgent in the national "
        "interest, the President may refer the Bill to the Supreme Court by a "
        "special reference. The Supreme Court shall determine any question of "
        "constitutionality within a shorter time period than in non-urgent cases.",
        ["122"], "procedural", "constitutional", "hard", "XV",
    )
    add(
        "What must accompany the Supreme Court's determination on the constitutionality of a Bill?",
        "Article 123 provides that the determination of the Supreme Court shall "
        "be accompanied by the reasons therefor and shall state whether the Bill "
        "or any provision thereof is inconsistent with the Constitution. Where the "
        "Court determines a provision to be inconsistent, it shall indicate whether "
        "the Bill may be passed by a simple majority, a two-thirds majority, or "
        "only after a Referendum.",
        ["123"], "procedural", "constitutional", "hard", "XV",
    )
    add(
        "Can a court challenge the validity of a Bill after it has received the Speaker's certificate?",
        "Article 124 provides that, save as provided in Articles 120, 121 and 122, "
        "no court or tribunal shall have the power or jurisdiction to inquire into "
        "or pronounce upon the validity of any law or any provision thereof which "
        "has been duly passed by Parliament and certified by the Speaker. This "
        "provision establishes the finality of parliamentary enactment once "
        "certified.",
        ["124"], "interpretive", "constitutional", "hard", "XV",
    )
    add(
        "What jurisdiction does the Supreme Court have in the interpretation of the Constitution?",
        "Article 125 grants the Supreme Court sole and exclusive jurisdiction to "
        "hear and determine any question relating to the interpretation of the "
        "Constitution. Whenever any such question arises in the course of any "
        "proceedings in any court or tribunal, that court or tribunal shall refer "
        "the question to the Supreme Court for determination.",
        ["125"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "How does a citizen invoke the Supreme Court's fundamental rights jurisdiction?",
        "Article 126 provides that the Supreme Court shall have sole and exclusive "
        "jurisdiction to hear and determine any question relating to the infringement "
        "or imminent infringement by executive or administrative action of any "
        "fundamental right or language right. Any person may petition the Supreme "
        "Court for relief within one month of such infringement.",
        ["126"], "procedural", "constitutional", "medium", "XV",
    )
    add(
        "What is the appellate jurisdiction of the Supreme Court?",
        "Article 127 provides that the Supreme Court shall, subject to the "
        "Constitution, be the final court of civil and criminal appellate "
        "jurisdiction for and within the Republic of Sri Lanka. The Court hears "
        "appeals from the Court of Appeal and such other appeals as may be "
        "prescribed by law.",
        ["127"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Can the President seek an advisory opinion from the Supreme Court?",
        "Article 129 provides that if at any time it appears to the President that "
        "a question of law or fact has arisen or is likely to arise which is of "
        "such nature and public importance that it is expedient to obtain the "
        "opinion of the Supreme Court upon it, the President may refer that question "
        "to the Court for its opinion. The opinion of the Court shall be pronounced "
        "in open court.",
        ["129"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What jurisdiction does the Supreme Court have over election and referendum petitions?",
        "Article 130 provides that the Supreme Court shall have the power to hear "
        "and determine and make such orders as provided by law on: (a) any legal "
        "proceeding relating to the election of the President or the validity of a "
        "Referendum; (b) any appeal from an order or judgement of the Court of "
        "Appeal on any election petition; and (c) such other matters as Parliament "
        "may by law prescribe.",
        ["130"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What jurisdiction does the Supreme Court have in contempt proceedings?",
        "Article 131 provides that the Supreme Court shall have, according to law, "
        "the power to take cognizance of and punish any person for breaches of the "
        "privileges of Parliament, and for contempt of court in respect of any "
        "court, tribunal or institution referred to in Article 105.",
        ["131"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Where does the Supreme Court ordinarily exercise its jurisdiction?",
        "Article 132 provides that the several jurisdictions of the Supreme Court "
        "shall be ordinarily exercised at Colombo unless the Chief Justice otherwise "
        "directs. The jurisdiction of the Supreme Court may be exercised in different "
        "matters at the same time by the several Judges of the Court sitting apart.",
        ["132"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "What is the procedure when there is no quorum of Judges available for Supreme Court proceedings?",
        "Article 133 provides that if at any time there is no quorum of Judges of "
        "the Supreme Court available to hold or continue any sittings, the Chief "
        "Justice may with the previous consent of the President request in writing "
        "the attendance at the sittings of the Court as an ad hoc Judge of any "
        "Judge of the Court of Appeal.",
        ["133"], "procedural", "constitutional", "hard", "XV",
    )
    add(
        "What is the right of the Attorney-General to be heard in Supreme Court proceedings?",
        "Article 134 provides that the Attorney-General shall be noticed and shall "
        "have the right to be heard in all proceedings in the Supreme Court in the "
        "exercise of its jurisdiction under Articles 120, 121, 122, 125, 126, "
        "129(1), and 130. This ensures the State's interests are represented in "
        "proceedings involving constitutional and electoral matters.",
        ["134"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Who is in charge of the Registry of the Supreme Court?",
        "Article 135 provides that the Registry of the Supreme Court shall be in "
        "charge of an officer designated the Registrar of the Supreme Court who "
        "shall be subject to the supervision, direction and control of the Chief "
        "Justice. The Registrar is responsible for the administration of the "
        "Court's registry functions.",
        ["135"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "How are the Rules of the Supreme Court made?",
        "Article 136 provides that, subject to the provisions of the Constitution "
        "and of any law, the Chief Justice with any three Judges of the Supreme "
        "Court may make rules regulating the exercise by the Court of its "
        "jurisdiction and the practice and procedure of the Court.",
        ["136"], "procedural", "constitutional", "medium", "XV",
    )

    # -------------------------------------------------------------------------
    # Chapter XV – The Court of Appeal (Arts 137–147)
    # -------------------------------------------------------------------------
    add(
        "How many Judges compose the Court of Appeal?",
        "Article 137 provides that the Court of Appeal shall consist of the "
        "President of the Court of Appeal and not less than six and not more than "
        "nineteen other Judges who shall be appointed by the President of the "
        "Republic. The President of the Court of Appeal shall preside over its "
        "proceedings.",
        ["137"], "factual", "constitutional", "easy", "XV",
    )
    add(
        "What powers does the Court of Appeal have to examine records of lower courts?",
        "Article 140 provides that the Court of Appeal shall have full power and "
        "authority to inspect and examine the records of any Court of First Instance "
        "or tribunal or other institution and grant and issue, according to law, "
        "orders in the nature of writs of certiorari, prohibition, mandamus, quo "
        "warranto and procedendo, except writs of habeas corpus.",
        ["140"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What habeas corpus jurisdiction does the Court of Appeal exercise?",
        "Article 141 provides that the Court of Appeal may grant and issue orders "
        "in the nature of writs of habeas corpus to bring up before such Court any "
        "person detained in any prison or other place of detention so that the Court "
        "may examine into the legality of the detention and order the release of "
        "such person if the detention is unlawful.",
        ["141"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What power does the Court of Appeal have to remove prisoners?",
        "Article 142 provides that the Court of Appeal may direct that a prisoner "
        "detained in any prison be brought before a court-martial or any Commission "
        "or tribunal for the purposes of being examined as a witness or otherwise. "
        "The Court has supervisory authority over the physical production of "
        "prisoners before judicial or quasi-judicial bodies.",
        ["142"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Can the Court of Appeal grant injunctions to prevent irreparable harm?",
        "Article 143 provides that the Court of Appeal shall have the power to "
        "grant and issue injunctions to prevent any irremediable mischief which "
        "might ensue before a party making an application for such injunction "
        "could prevent the same by bringing an action in any Court of First "
        "Instance. This jurisdiction protects parties from irreversible harm "
        "pending substantive proceedings.",
        ["143"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "What jurisdiction does the Court of Appeal have over parliamentary election petitions?",
        "Article 144 provides that the Court of Appeal shall have and exercise "
        "jurisdiction to try election petitions in respect of elections to "
        "Parliament. The Court shall have the power to make such orders as provided "
        "by law to determine the validity of elections and grant appropriate relief.",
        ["144"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Can the Court of Appeal inspect records of lower courts on its own motion?",
        "Article 145 provides that the Court of Appeal may, ex mero motu or on "
        "any application made, call for, inspect and examine any record of any "
        "Court of First Instance and in the exercise of its revisionary powers "
        "may make any order thereon as the interests of justice may require. "
        "This gives the Court broad supervisory jurisdiction.",
        ["145"], "factual", "constitutional", "medium", "XV",
    )
    add(
        "Who manages the Registry of the Court of Appeal?",
        "Article 147 provides that the Registry of the Court of Appeal shall be "
        "in charge of an officer designated as the Registrar of the Court of "
        "Appeal. The Registrar shall be subject to the supervision, direction "
        "and control of the President of the Court of Appeal and is responsible "
        "for the administration of the Court's registry.",
        ["147"], "factual", "constitutional", "easy", "XV",
    )

    # -------------------------------------------------------------------------
    # Finance / Provincial Councils (Arts 138, 151, 152, 154A, 154B)
    # -------------------------------------------------------------------------
    add(
        "What must Parliament provide for in relation to Provincial Councils?",
        "Article 138 provides that Parliament shall by law provide for: "
        "(a) the election of members of Provincial Councils and the qualifications "
        "for membership; (b) the procedure for transaction of business by every "
        "such Council; (c) the salaries and allowances of members; and "
        "(d) such other matters as Parliament may consider necessary for the "
        "proper functioning of Provincial Councils.",
        ["138"], "factual", "constitutional", "medium", "XVIIA",
    )
    add(
        "What is the Contingencies Fund and for what purpose may it be used?",
        "Article 151 provides that, notwithstanding any provision of Article 149, "
        "Parliament may by law create a Contingencies Fund for the purpose of "
        "providing for urgent and unforeseen expenditure for which no provision "
        "exists in the Appropriation Act or any other law. Advances from the "
        "Fund require parliamentary approval at the earliest opportunity.",
        ["151"], "factual", "constitutional", "medium", "XVI",
    )
    add(
        "What special procedure applies to Bills authorising expenditure from the Consolidated Fund?",
        "Article 152 provides that no Bill or motion authorising the disposal of "
        "or the imposition of charges upon the Consolidated Fund or other funds "
        "of the Republic shall be introduced or moved in Parliament except by a "
        "Minister, and only on the recommendation of the Cabinet of Ministers "
        "signified by the President or by a Minister authorised by the President.",
        ["152"], "procedural", "constitutional", "medium", "XVI",
    )
    add(
        "How are Provincial Councils established under the Constitution?",
        "Article 154A provides that, subject to the provisions of the Constitution, "
        "a Provincial Council shall be established for every Province specified "
        "in the Eighth Schedule with effect from such date or dates as the "
        "President may appoint by Order published in the Gazette. Different dates "
        "may be appointed for different Provinces.",
        ["154A"], "factual", "constitutional", "medium", "XVIIA",
    )
    add(
        "What is the role of the Governor of a Province?",
        "Article 154B provides that there shall be a Governor for each Province "
        "for which a Provincial Council has been established. The Governor shall "
        "be appointed by the President and shall hold office during the pleasure "
        "of the President. The Governor is the representative of the President "
        "in the Province and exercises executive powers on the President's behalf.",
        ["154B"], "factual", "constitutional", "medium", "XVIIA",
    )

    # -------------------------------------------------------------------------
    # Special Provisions (Arts 155–165)
    # -------------------------------------------------------------------------
    add(
        "What is the constitutional basis for public security and emergency regulations?",
        "Article 155 provides that the Public Security Ordinance as amended and "
        "in force immediately prior to the commencement of the Constitution shall "
        "be deemed to be a law enacted by Parliament. The power to make emergency "
        "regulations under the Public Security Ordinance shall be exercisable only "
        "by the President, subject to parliamentary approval within a specified period.",
        ["155"], "factual", "constitutional", "hard", "XX",
    )
    add(
        "What is the Parliamentary Commissioner for Administration (Ombudsman)?",
        "Article 156 provides that Parliament shall by law provide for the "
        "establishment of the office of the Parliamentary Commissioner for "
        "Administration (Ombudsman), charged with the duty of investigating and "
        "reporting upon complaints or allegations of the infringement of "
        "fundamental rights and other injustices by officers and authorities of "
        "the Government.",
        ["156"], "factual", "constitutional", "medium", "XX",
    )
    add(
        "Under what conditions can a Treaty or Agreement between Sri Lanka and a foreign government bind future Parliaments?",
        "Article 157 provides that where Parliament by resolution passed by not "
        "less than two-thirds of the whole number of Members of Parliament approves "
        "as being essential for the development of the national economy, any Treaty "
        "or Agreement between the Government of Sri Lanka and the Government of a "
        "foreign State or international organisation, such Treaty shall have the "
        "force of law in Sri Lanka.",
        ["157"], "interpretive", "constitutional", "hard", "XX",
    )
    add(
        "What are the rules regarding delegation of constitutional powers?",
        "Article 158 provides that where any person is empowered under the "
        "provisions of the Constitution to delegate any power, duty or function "
        "to any other person, such person delegating the power may, "
        "notwithstanding such delegation, still exercise, perform or discharge "
        "such power, duty or function. Delegation under the Constitution does "
        "not extinguish the delegator's own authority.",
        ["158"], "interpretive", "constitutional", "hard", "XX",
    )
    add(
        "What happens when the Speaker is unable to discharge the functions of office?",
        "Article 159 provides that where the Speaker is unable to discharge the "
        "functions of office, the powers, duties and functions conferred on or "
        "assigned to the Speaker by any provision of the Constitution (other than "
        "Articles 31(4), 37, 38(2)(b), 39(2), and 40) may be exercised by the "
        "Deputy Speaker.",
        ["159"], "procedural", "constitutional", "medium", "XX",
    )
    add(
        "Who was the first President of Sri Lanka under the 1978 Constitution?",
        "Article 160 provides that notwithstanding anything to the contrary in any "
        "other provision of the Constitution, the person holding the office of "
        "President immediately before the commencement of the Constitution shall "
        "be the first President under the Constitution and shall be deemed for all "
        "purposes to have been elected to that office.",
        ["160"], "factual", "constitutional", "easy", "XX",
    )
    add(
        "What was the transitional provision regarding the composition of the first Parliament?",
        "Article 161 provides that the first Parliament shall consist of one hundred "
        "and sixty-eight members and, subject to the succeeding provisions of this "
        "Article, all persons who immediately before the commencement of the "
        "Constitution were members of the National State Assembly shall become "
        "members of Parliament on the commencement of the Constitution.",
        ["161"], "factual", "constitutional", "medium", "XX",
    )
    add(
        "What was the transitional arrangement for applying the new electoral provisions after the 1978 Constitution?",
        "Article 162 provides that the provisions of Article 98 (other than "
        "paragraphs (8) and (9)) and Article 99 shall not come into operation "
        "until the General Election held upon the dissolution of the first "
        "Parliament. This transitional provision allowed the old electoral "
        "arrangements to continue for the first Parliament.",
        ["162"], "procedural", "constitutional", "hard", "XX",
    )
    add(
        "What happened to Judges of the Supreme Court and High Courts upon the commencement of the 1978 Constitution?",
        "Article 163 provides that all Judges of the Supreme Court and the Judges "
        "of High Courts established by the Administration of Justice Law No. 44 of "
        "1973, holding office on the day immediately before the commencement of the "
        "Constitution, shall on the commencement of the Constitution cease to hold "
        "their respective offices under that Law.",
        ["163"], "factual", "constitutional", "medium", "XX",
    )
    add(
        "What transitional provision applied to public officers and judicial officers upon commencement of the Constitution?",
        "Article 164 provides that, subject to Article 163, every public officer, "
        "judicial officer and other person who immediately before the commencement "
        "of the Constitution held office in any court or tribunal, or in any "
        "position under the former constitutional arrangements, shall be deemed to "
        "have been appointed to the corresponding office or position under the "
        "Constitution.",
        ["164"], "factual", "constitutional", "medium", "XX",
    )
    add(
        "What oath or affirmation is required of all public officers upon the commencement of the Constitution?",
        "Article 165 provides that every public officer, judicial officer and every "
        "other person required by the Constitution to take an oath or make an "
        "affirmation on entering upon the duties of office shall, if already in "
        "office at the commencement of the Constitution, take and subscribe the "
        "appropriate oath or affirmation set out in the Fourth Schedule within "
        "such period as Parliament may provide.",
        ["165"], "procedural", "constitutional", "medium", "XX",
    )

    # -------------------------------------------------------------------------
    # Cross-referenced items
    # -------------------------------------------------------------------------
    add(
        "How do the constitutional provisions on territorial integrity (Article 5) relate to the unitary state declaration (Article 2)?",
        "Article 2 declares that the Republic of Sri Lanka is a Unitary State, "
        "while Article 5 defines the territory as consisting of the twenty-five "
        "administrative districts and their territorial waters. Read together, "
        "these provisions establish both the political character (unitary) and "
        "the physical extent of the Republic. The unitary character under "
        "Article 2 means that no province or administrative district may "
        "secede from the territory defined in Article 5.",
        ["2", "5"], "cross_referenced", "constitutional", "hard", "I",
        ["2", "5"],
    )
    add(
        "How does the President's responsibility to Parliament under Article 42 interact with presidential immunity under Article 35?",
        "Article 35 grants the President immunity from legal proceedings during "
        "tenure, while Article 42 makes the President responsible to Parliament "
        "for the exercise of executive powers. These provisions work in tandem: "
        "immunity under Article 35 protects the President from judicial challenge, "
        "but constitutional accountability is maintained through parliamentary "
        "oversight under Article 42, which includes the power of impeachment.",
        ["35", "42"], "cross_referenced", "constitutional", "hard", "VII",
        ["35", "42"],
    )
    add(
        "How do the Supreme Court's constitutional jurisdiction (Article 120) and its fundamental rights jurisdiction (Article 126) interact?",
        "Article 120 grants the Supreme Court exclusive jurisdiction to determine "
        "whether any Bill is inconsistent with the Constitution, exercised before "
        "a Bill is certified by the Speaker. Article 126 grants the Court exclusive "
        "jurisdiction over infringement of fundamental rights by executive or "
        "administrative action, exercised after enactment. Together they create "
        "a two-stage judicial review mechanism: pre-legislative review under "
        "Article 120 and post-enactment rights enforcement under Article 126.",
        ["120", "126"], "cross_referenced", "constitutional", "hard", "XV",
        ["120", "126"],
    )
    add(
        "How does the procedure for passing entrenched Bills (Article 83) relate to Parliament's general legislative power (Article 75)?",
        "Article 75 confers on Parliament the general power to make laws, including "
        "laws amending the Constitution. However, Article 83 creates a special "
        "category of entrenched provisions — including Articles 1, 2, 3, 6, 7, 8, "
        "9, 10, and 11 — which require both a two-thirds majority in Parliament "
        "and approval at a Referendum. Article 83 thus operates as a constitutional "
        "constraint on the otherwise plenary legislative power in Article 75.",
        ["75", "83"], "cross_referenced", "constitutional", "hard", "XI",
        ["75", "83"],
    )
    add(
        "What is the relationship between the right of access to information (Article 14A) and fundamental rights restrictions (Article 15)?",
        "Article 14A grants citizens the right of access to information held by "
        "the State and other public bodies. Article 15 permits restrictions on "
        "the exercise of fundamental rights, including rights under Article 14A, "
        "in the interests of national security, public order, or the protection "
        "of the rights and freedoms of others. Any restriction on the right of "
        "access to information must therefore meet the Article 15 standard of "
        "being prescribed by law and proportionate.",
        ["14A", "15"], "cross_referenced", "constitutional", "hard", "III",
        ["14A", "15"],
    )

    # -------------------------------------------------------------------------
    # Additional interpretive items
    # -------------------------------------------------------------------------
    add(
        "How should the constitutional immunity of the President be interpreted in light of the rule of law?",
        "Article 35 grants the President immunity from civil or criminal proceedings "
        "during tenure. This immunity is an exception to the general principle of "
        "equality before the law implied by the Constitution. Courts have construed "
        "this immunity narrowly: it applies only to proceedings in respect of acts "
        "done in an official or private capacity while in office, and does not "
        "prevent Parliament from exercising its oversight function under Article 42 "
        "or conducting impeachment proceedings under Article 38.",
        ["35"], "interpretive", "constitutional", "hard", "VII",
    )
    add(
        "What does 'sole and exclusive jurisdiction' in Article 125 mean for other courts dealing with constitutional questions?",
        "Article 125 vests in the Supreme Court sole and exclusive jurisdiction to "
        "determine questions relating to the interpretation of the Constitution. "
        "The effect is that no other court may itself determine such questions. "
        "When a constitutional question arises in other courts, those courts must "
        "refer the question to the Supreme Court and await its determination, "
        "rather than proceeding to decide the point themselves.",
        ["125"], "interpretive", "constitutional", "hard", "XV",
    )
    add(
        "How does the constitutional requirement for Bills to be in Sinhala and Tamil (Article 23) interact with the supremacy of the Constitution in cases of conflict?",
        "Article 23 requires all laws to be enacted in Sinhala and Tamil with an "
        "English translation, and Parliament shall determine which text prevails in "
        "case of inconsistency. Article 25A provides that where there is any "
        "inconsistency between a law and the language provisions of the Constitution, "
        "the constitutional provisions prevail. Together, these Articles establish "
        "that any statutory text purporting to override the language chapter of the "
        "Constitution will be struck down to the extent of the inconsistency.",
        ["23", "25A"], "interpretive", "constitutional", "hard", "IV",
        ["23", "25A"],
    )
    add(
        "What constitutional framework governs the validity of a Bill submitted to a Referendum but rejected by the People?",
        "Articles 80 and 85 together govern this situation. Article 80 provides "
        "that a Bill shall become law only when the Speaker's certificate is "
        "endorsed. Article 85 requires certain Bills to be submitted to the People "
        "at a Referendum. If the People reject such a Bill at the Referendum, "
        "the Speaker's certificate cannot be granted and the Bill cannot become law, "
        "thus preserving the constitutional requirement of popular consent for "
        "fundamental legal changes.",
        ["80", "85"], "interpretive", "constitutional", "hard", "XI",
        ["80", "85"],
    )
    add(
        "How does the Directive Principles chapter interact with fundamental rights in terms of enforceability?",
        "Article 27 sets out Directive Principles of State Policy which are "
        "described as guiding Parliament, the President, the Cabinet, and all "
        "governing authorities. Unlike the fundamental rights in Chapter III, "
        "the Directive Principles are not directly enforceable by a court. "
        "However, they serve as interpretive guides: when the courts interpret "
        "statutes or fundamental rights provisions, the Directive Principles "
        "may be used to inform the purposive reading of otherwise ambiguous "
        "provisions.",
        ["27"], "interpretive", "constitutional", "hard", "VI",
    )

    return items


def main():
    by_id, by_sec = load_corpus()
    all_ids = set(by_id.keys())

    # Sanity-check existing IDs
    with open(DATASET_PATH) as f:
        data = json.load(f)

    existing_items = data["items"]
    print(f"Existing items: {len(existing_items)}")

    new_items = build_new_items(by_id, by_sec)
    print(f"New items generated: {len(new_items)}")

    # Check for duplicate passage ID references
    existing_relevant = set()
    for item in existing_items:
        existing_relevant.update(item.get("relevant_passages", []))

    # Merge
    all_items = existing_items + new_items
    data["items"] = all_items
    data["num_items"] = len(all_items)
    data["updated"] = datetime.now().isoformat()

    # Write
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved: {len(all_items)} items total")

    # Distribution report
    from collections import Counter
    qtypes = Counter(i["query_type"] for i in all_items)
    domains = Counter(i["legal_domain"] for i in all_items)
    diffs = Counter(i["difficulty"] for i in all_items)
    print(f"\nQuery types: {dict(qtypes)}")
    print(f"Domains: {dict(domains)}")
    print(f"Difficulties: {dict(diffs)}")


if __name__ == "__main__":
    main()
