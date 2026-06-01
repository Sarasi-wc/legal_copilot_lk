"""
Comprehensive QAC dataset fix script.
Applies 54 fixes identified during legal review of AI-generated items.

Changes:
  - Removes 3 duplicates: QAC_083, QAC_084, QAC_129
  - Renumbers all items sequentially
  - Fixes answer_text for 38+ items
  - Fixes passage_ids for 5 items
  - Fixes query_type for 3 items
  - Fixes metadata.articles for 1 item
"""
import json
import copy
import shutil
from pathlib import Path

DATASET_PATH = Path("data/evaluation/qac_dataset.json")
BACKUP_PATH = Path("data/evaluation/qac_dataset.json.bak8")

# ── helpers ──────────────────────────────────────────────────────────────────

def get_item(items, item_id):
    for item in items:
        if item["item_id"] == item_id:
            return item
    raise KeyError(f"{item_id} not found")

def replace_in_answer(item, old, new):
    ans = item["gold_answers"][0]["answer_text"]
    if old not in ans:
        print(f"  WARNING: '{old[:60]}' not found in {item['item_id']} answer")
        return
    item["gold_answers"][0]["answer_text"] = ans.replace(old, new, 1)

def replace_passage_id(item, old_pid, new_pid, new_section_number=None, new_passage_text=None):
    """Fix passage_id in relevant_passages, citations, and optionally section_number."""
    # Fix relevant_passages
    rp = item.get("relevant_passages", [])
    if old_pid in rp:
        idx = rp.index(old_pid)
        rp[idx] = new_pid
    else:
        print(f"  WARNING: {old_pid} not in relevant_passages for {item['item_id']}")

    # Fix citations
    for cit in item["gold_answers"][0].get("citations", []):
        if cit.get("passage_id") == old_pid:
            cit["passage_id"] = new_pid
            if new_section_number:
                cit["section_number"] = new_section_number
            if new_passage_text:
                cit["passage_text"] = new_passage_text

def add_citation(item, section_number, passage_id, passage_text):
    """Add a new citation entry."""
    new_cit = {
        "act_name": "Constitution of the Democratic Socialist Republic of Sri Lanka",
        "act_number": "",
        "act_year": 1978,
        "section_number": section_number,
        "passage_id": passage_id,
        "passage_text": passage_text,
        "start_char": None,
        "end_char": None
    }
    item["gold_answers"][0].setdefault("citations", []).append(new_cit)
    rp = item.setdefault("relevant_passages", [])
    if passage_id not in rp:
        rp.append(passage_id)

# ── load ─────────────────────────────────────────────────────────────────────

with open(DATASET_PATH) as f:
    data = json.load(f)

# backup
shutil.copy(DATASET_PATH, BACKUP_PATH)
print(f"Backup written to {BACKUP_PATH}")

items = data["items"]
print(f"Loaded {len(items)} items")

# ── 1. remove duplicates ──────────────────────────────────────────────────────

REMOVE = {"QAC_083", "QAC_084", "QAC_129"}
items = [it for it in items if it["item_id"] not in REMOVE]
print(f"After removing {len(REMOVE)} duplicates: {len(items)} items")

# ── 2. renumber ───────────────────────────────────────────────────────────────

for i, item in enumerate(items, start=1):
    item["item_id"] = f"QAC_{i:03d}"

print("Renumbered all items 001–334")

# rebuild lookup by new ids
idx_map = {item["item_id"]: item for item in items}

# ── helper to get item by content (original id) ───────────────────────────────
# After removing 083/084 (before 085), items 085→083, 086→084, 087→085...
# After removing 129 (originally at 129, which is now 127 after -2 shift from 083/084):
#   original 129 is now at position 129-2=127, but it's in REMOVE so gone
#   original 130 becomes 128, etc.
# Summary of offsets after removals:
#   original < 083 -> same (no change)
#   original 085-128 -> original - 2
#   original 130-337 -> original - 3
# NOTE: QAC_129 original was at position 129, removed; items after shift by 3 total after that.

def remap(original_num):
    """Map original QAC number to new QAC number after removals."""
    if original_num < 83:
        return original_num
    elif original_num == 83 or original_num == 84:
        return None  # removed
    elif 85 <= original_num <= 128:
        return original_num - 2
    elif original_num == 129:
        return None  # removed
    else:
        return original_num - 3

def get_orig(original_num):
    new_num = remap(original_num)
    if new_num is None:
        raise ValueError(f"QAC_{original_num:03d} was removed")
    new_id = f"QAC_{new_num:03d}"
    return idx_map[new_id]

# ── 3. query_type fixes ───────────────────────────────────────────────────────

get_orig(187)["query_type"] = "factual"
get_orig(204)["query_type"] = "factual"
get_orig(209)["query_type"] = "factual"
print("Fixed 3 query_types")

# ── 4. metadata.articles fix ─────────────────────────────────────────────────

get_orig(172)["metadata"]["articles"] = ["154Q"]
print("Fixed QAC_172 metadata.articles 138->154Q")

# ── 5. passage_id fixes ───────────────────────────────────────────────────────

# QAC_277: add Art 21 citation
item277 = get_orig(277)
add_citation(
    item277,
    section_number="21",
    passage_id="ACT__1978_SEC_21",
    passage_text=(
        "(1) A person shall be entitled to be educated through the medium of "
        "either of the National Languages: Provided that the provisions of this "
        "paragraph shall not apply to an institution of higher education where "
        "the medium of instruction is a language other than a National Language."
    )
)
item277["metadata"].setdefault("articles", [])
if "21" not in item277["metadata"].get("articles", []):
    item277["metadata"]["articles"] = list(item277["metadata"].get("articles", [])) + ["21"]

# QAC_292: SEC_27_CHUNK_0 -> SEC_27_CHUNK_3
item292 = get_orig(292)
replace_passage_id(
    item292,
    "ACT__1978_SEC_27_CHUNK_0",
    "ACT__1978_SEC_27_CHUNK_3",
    new_section_number="27",
    new_passage_text=(
        "(11) The State shall create the necessary economic and social "
        "environment to enable people of all religious faiths to make a "
        "reality of their religious principles. (12) The State shall recognize "
        "and protect the family as the basic unit of society. (13) The State "
        "shall promote with special care the interests of children and youth, "
        "so as to ensure their full development, physical, mental, moral, "
        "religious and social."
    )
)

# QAC_320: SEC_107 -> SEC_106_CHUNK_1
item320 = get_orig(320)
replace_passage_id(
    item320,
    "ACT__1978_SEC_107",
    "ACT__1978_SEC_106_CHUNK_1",
    new_section_number="107",
    new_passage_text=(
        "(1) The Chief Justice, the President of the Court of Appeal and every "
        "other judge of the Supreme Court and the Court of Appeal shall be "
        "appointed by the President subject to the approval of the "
        "Constitutional Council, by warrant under his hand. (2) Every such "
        "Judge shall hold office during good behaviour and shall not be removed "
        "except by an order of the President made after an address of "
        "Parliament supported by a majority of the total number of Members of "
        "Parliament (including those not present) has been presented to the "
        "President for such removal on the ground of proved misbehaviour or "
        "incapacity."
    )
)

# QAC_325: SEC_154G -> SEC_154C (second citation)
item325 = get_orig(325)
replace_passage_id(
    item325,
    "ACT__1978_SEC_154G",
    "ACT__1978_SEC_154C",
    new_section_number="154C",
    new_passage_text=(
        "Executive power extending to the matters with respect to which a "
        "Provincial Council has power to make statutes shall be exercised by "
        "the Governor of the Province for which that Provincial Council is "
        "established, either directly or through Ministers of the Board of "
        "Ministers, or through officers subordinate to him, in accordance with "
        "Article 154F."
    )
)
# also fix metadata.articles
item325["metadata"]["articles"] = ["154B", "154C"]

print("Fixed 4 passage_id issues (QAC_277 add, QAC_292 CHUNK_0->CHUNK_3, QAC_320 SEC_107->SEC_106_CHUNK_1, QAC_325 154G->154C)")

# ── 6. answer_text fixes ─────────────────────────────────────────────────────

# QAC_085: nomination by elector
replace_in_answer(
    get_orig(85),
    "by a group of electors",
    "by an elector whose name has been entered in any register of electors"
)

# QAC_088: pardon sub-items (b)(c) wrong
replace_in_answer(
    get_orig(88),
    "either indefinite or for a specified period, of the execution of any sentence; "
    "(c) grant a remission or suspension of any sentence or penalty.",
    "either indefinite for such period as the President may think fit, of the "
    "execution of any sentence passed on such offender; (c) substitute a less "
    "severe form of punishment for any punishment imposed on such offender; or "
    "(d) remit the whole or any part of any punishment imposed or of any penalty "
    "or forfeiture otherwise due to the Republic on account of such offence."
)

# QAC_100: Cabinet continues after dissolution
replace_in_answer(
    get_orig(100),
    "Accordingly the Prime Minister and Ministers shall, after such General "
    "Election, hold office until the Cabinet is newly constituted.",
    "Accordingly, the Prime Minister and the Ministers of the Cabinet of "
    "Ministers shall continue to function unless they cease to hold office as "
    "provided by Article 47, and shall comply with the criteria set out by the "
    "Commissioner-General of Elections and shall not cause any undue influence "
    "on the General Election."
)

# QAC_102: Secretary to PM duties (wrong – says PM substitute)
replace_in_answer(
    get_orig(102),
    "The Secretary shall have charge of the office of the Prime Minister and "
    "shall perform and discharge the duties and functions of the office of Prime "
    "Minister in the absence or incapacity of the Prime Minister.",
    "The Secretary shall have charge of the office of the Prime Minister and "
    "shall perform and discharge the duties and functions of his office, subject "
    "to the directions of the Prime Minister."
)

# QAC_103: Secretary to Ministry duties (wrong – 'official head' not in Art 52)
replace_in_answer(
    get_orig(103),
    "The Secretary shall be the official head of the Ministry and shall be "
    "responsible for the administration of the Ministry and for implementing the "
    "policy of the Cabinet of Ministers in relation to that Ministry.",
    "The Secretary to a Ministry shall, subject to the direction and control of "
    "his Minister, exercise supervision over the departments of Government and "
    "other institutions in charge of the Minister. Notwithstanding the dissolution "
    "of the Cabinet of Ministers, the Secretary to a Ministry shall continue to "
    "hold office until a new Secretary is appointed."
)

# QAC_104: Fourth Schedule only (missing Seventh Schedule)
replace_in_answer(
    get_orig(104),
    "as set out in the Fourth Schedule of the Constitution.",
    "as set out in the Fourth Schedule and Seventh Schedule of the Constitution."
)

# QAC_108: Article 58 appeals go to Commission, not AAT
replace_in_answer(
    get_orig(108),
    "shall have the right to appeal to the Administrative Appeals Tribunal "
    "established under Article 59.",
    "may appeal to the Public Service Commission against such order, and the "
    "Commission shall have the power to alter, vary, rescind or confirm the order."
)

# QAC_109: AAT reviews Commission decisions, not public officer orders directly
replace_in_answer(
    get_orig(109),
    "The Tribunal has jurisdiction to hear and determine appeals by public officers "
    "against orders relating to promotions, transfers, dismissals, and disciplinary "
    "matters made by the Public Service Commission or delegated bodies.",
    "The Administrative Appeals Tribunal shall have the power to alter, vary or "
    "rescind any order or decision made by the Public Service Commission. The "
    "constitution, powers and procedure of the Tribunal, including time limits for "
    "preferring appeals, shall be provided for by law."
)

# QAC_112: Art 64 doesn't say 'preside over sittings' or 'secret ballot'
replace_in_answer(
    get_orig(112),
    "The Speaker shall preside over the sittings of Parliament and is elected by "
    "the Members of Parliament by secret ballot.",
    "Whenever the office of Speaker, Deputy Speaker or Deputy Chairman of "
    "Committees becomes vacant otherwise than by dissolution, Parliament shall at "
    "its first meeting elect another Member to fill the vacancy."
)

# QAC_114: (d) disqualification, not expulsion from party
replace_in_answer(
    get_orig(114),
    "(d) expulsion from the political party or group which nominated the Member; or "
    "(e) any other grounds specified in the Constitution.",
    "(d) becoming subject to any disqualification specified in Article 89 or 91; "
    "(e) becoming a member of the Public Service or an employee of a public "
    "corporation without ceasing to be so before sitting in Parliament; or "
    "(f) absence from sittings for a continuous period of three months without "
    "leave of Parliament."
)

# QAC_115: 'remuneration or allowance' not 'salaries and allowances'; 'by law or by resolution'
replace_in_answer(
    get_orig(115),
    "shall be paid such salaries and allowances as Parliament shall by resolution "
    "determine.",
    "shall be paid such remuneration or allowance as may be provided by Parliament, "
    "by law or by resolution."
)

# QAC_116: Speaker voting – completely inverted
replace_in_answer(
    get_orig(116),
    "The Speaker or presiding Member shall have an original vote and shall not have "
    "a casting vote, except when there is an equality of votes.",
    "The person presiding shall not vote in the first instance but shall have and "
    "exercise a casting vote in the event of an equality of votes."
)

# QAC_119: AG's duty is specific to Art 82 paras (1)&(2), communicates to President
replace_in_answer(
    get_orig(119),
    "Article 77 places a duty on the Attorney-General to examine every Bill for any "
    "contravention of the requirements of the Constitution, and to certify that the "
    "Bill does not, or that certain provisions of the Bill do, contravene those "
    "requirements. This certification assists Parliament in determining whether a "
    "Bill requires special procedures.",
    "Article 77 places a duty on the Attorney-General to examine every Bill for any "
    "contravention of the requirements of paragraphs (1) and (2) of Article 82 and "
    "for any provision which cannot be validly passed except by the special majority "
    "prescribed by the Constitution. If the Attorney-General is of the opinion that "
    "a Bill contravenes any such requirements, he shall communicate such opinion to "
    "the President (or, in the case of an amendment to a Bill, to the Speaker)."
)

# QAC_125: result to President, not Parliament
replace_in_answer(
    get_orig(125),
    "communicate the result to Parliament.",
    "communicate the result thereof to the President."
)

# QAC_139: sub-items (b)–(e) completely wrong
replace_in_answer(
    get_orig(139),
    "(b) the prescribing of a qualifying date for registration; "
    "(c) the conduct of elections of Members of Parliament and of the President; "
    "(d) the grounds for disqualification; and "
    "(e) the determination of any question relating to elections.",
    "(b) the prescribing of a qualifying date on which a person should be resident "
    "in any Electoral District to be entered in the register of electors of that "
    "Electoral District; (c) the prescribing of a qualifying date on which a person "
    "should have attained the age of eighteen years to qualify for registration as "
    "an elector; (d) the preparation and revision of registers of electors; "
    "(e) the procedure for the election of Members of Parliament; "
    "(f) the creation of offences relating to such elections; and "
    "(g) the grounds for avoiding such elections and the manner of holding fresh "
    "elections."
)

# QAC_140: 'on leave' not 'on leave without pay'; 'conclusion of the election' not 'declaration of the poll'
replace_in_answer(
    get_orig(140),
    "shall be deemed to be on leave without pay during the period from the date of "
    "their nomination as a candidate to the date of the declaration of the poll.",
    "shall be deemed to be on leave from the date on which he stands nominated as a "
    "candidate until the conclusion of the election. Such a public officer or "
    "officer of a public corporation shall not during such period exercise, perform "
    "or discharge any of the powers, duties or functions of his office."
)

# QAC_146: 'shall, on approval of Constitutional Council' not 'may'
replace_in_answer(
    get_orig(146),
    "the President of the Republic may appoint a Judge of the Supreme Court or the "
    "Court of Appeal respectively to act in that office during such period of "
    "inability.",
    "the President shall, on the approval of the Constitutional Council, appoint "
    "another Judge of the Supreme Court, or of the Court of Appeal, as the case "
    "may be, to act in the office of Chief Justice, or the President of the Court "
    "of Appeal, respectively, during such period."
)

# QAC_150: urgent Bills must be for national security or disaster management, not general 'national interest'
replace_in_answer(
    get_orig(150),
    "Where the Cabinet of Ministers certifies that a Bill is urgent in the national "
    "interest, the President may refer the Bill to the Supreme Court by a special "
    "reference. The Supreme Court shall determine any question of constitutionality "
    "within a shorter time period than in non-urgent cases.",
    "Where the Cabinet of Ministers considers that a Bill is urgent in the interest "
    "of national security or for the purpose of any matter relating to disaster "
    "management, and the Bill bears an endorsement to that effect under the hand of "
    "the Secretary to the Cabinet, the President shall refer the Bill to the Supreme "
    "Court by a special written reference. The Supreme Court shall make its "
    "determination within twenty-four hours (or up to three days as specified by "
    "the President) and shall communicate its determination only to the President "
    "and the Speaker."
)

# QAC_156: missing 'of such' before 'public importance'
replace_in_answer(
    get_orig(156),
    "which is of such nature and public importance",
    "which is of such nature and of such public importance"
)
# Also remove fabricated 'pronounced in open court'
replace_in_answer(
    get_orig(156),
    " The opinion of the Court shall be pronounced in open court.",
    ""
)

# QAC_157: '(b) any appeal... on any election petition' -> 'in an election petition case'
#           and remove fabricated (c)
replace_in_answer(
    get_orig(157),
    "(b) any appeal from an order or judgement of the Court of Appeal on any "
    "election petition; and (c) such other matters as Parliament may by law "
    "prescribe.",
    "(b) any appeal from an order or judgment of the Court of Appeal in an election "
    "petition case. The hearing of a proceeding relating to the election of the "
    "President or the validity of a referendum shall be by at least five Judges of "
    "the Supreme Court."
)

# QAC_172: answer text is correct; metadata already fixed above
# (passage_id stays ACT__1978_SEC_138 – OCR mapping for Art 154Q)

# QAC_173: Contingencies Fund description and advance procedure wrong
replace_in_answer(
    get_orig(173),
    "for the purpose of providing for urgent and unforeseen expenditure for which "
    "no provision exists in the Appropriation Act or any other law. "
    "Advances from the Fund require parliamentary approval at the earliest "
    "opportunity.",
    "for the purpose of providing for urgent and unforeseen expenditure. "
    "The Minister in charge of Finance, if satisfied that there is need for such "
    "expenditure and that no provision exists for it, may, with the consent of the "
    "President, authorise advances from the Contingencies Fund. Such advances must "
    "be followed by provision being made for such expenditure by Parliament at the "
    "earliest opportunity."
)

# QAC_174: 'recommendation' vs 'approval' / scope of bills covered
replace_in_answer(
    get_orig(174),
    "no Bill or motion authorising the disposal of or the imposition of charges "
    "upon the Consolidated Fund or other funds of the Republic shall be introduced "
    "or moved in Parliament except by a Minister, and only on the recommendation of "
    "the Cabinet of Ministers signified by the President or by a Minister authorised "
    "by the President.",
    "no Bill or motion authorising the disposal of or the imposition of charges "
    "upon the Consolidated Fund or other funds of the Republic, or the imposition "
    "of any tax or the repeal, augmentation or reduction of any tax for the time "
    "being in force, shall be introduced in Parliament except by a Minister, and "
    "unless such Bill or motion has been approved by the Cabinet of Ministers."
)

# QAC_177: emergency regulations description
replace_in_answer(
    get_orig(177),
    "The power to make emergency regulations under the Public Security Ordinance "
    "shall be exercisable only by the President, subject to parliamentary approval "
    "within a specified period.",
    "The power to make emergency regulations under the Public Security Ordinance or "
    "the law for the time being in force relating to public security shall include "
    "the power to make regulations having the legal effect of overriding, "
    "suspending or amending any law."
)

# QAC_186: over-generalises court/tribunal qualification (should cite Art 105(2))
replace_in_answer(
    get_orig(186),
    "held office in any court or tribunal, or in any position under the former "
    "constitutional arrangements, shall be deemed to have been appointed to the "
    "corresponding office or position under the Constitution.",
    "held office in any court or tribunal deemed by virtue of paragraph (2) of "
    "Article 105 to be a court or tribunal created and established by Parliament, "
    "or was in the service of the Republic, any local authority, or any public "
    "corporation, shall be deemed to have been appointed to the corresponding "
    "office or position under the Constitution."
)

# QAC_198: 'by virtue of any other written law' -> 'by virtue of any prior service'
replace_in_answer(
    get_orig(198),
    "any other pension to which the person is entitled by virtue of any other "
    "written law.",
    "any other pension to which such person is entitled by virtue of any prior "
    "service."
)

# QAC_202: duties assigned 'by Parliament' -> 'by the Constitution, or by any other written law'
replace_in_answer(
    get_orig(202),
    "such other duties and functions as may be imposed or assigned to it by "
    "Parliament.",
    "such other duties and functions as may be imposed or assigned to the Council "
    "by the Constitution, or by any other written law."
)

# QAC_205: wrong exception clause ('assumed the office himself' vs 'dissolved Parliament under Art 70')
replace_in_answer(
    get_orig(205),
    "unless the President has in the meantime assumed the office himself, stand "
    "dissolved.",
    "unless the President has, in the exercise of his powers under Article 70, "
    "dissolved Parliament, stand dissolved."
)

# QAC_206: minor wording ('their lawful duty' vs 'such persons lawful duty'; missing 'by himself or by or with any other person')
replace_in_answer(
    get_orig(206),
    "every person who, otherwise than in the course of their lawful duty, directly "
    "or indirectly influences or attempts to influence or interferes with any "
    "decision of the Public Service Commission, or a Committee or officer to whom "
    "powers have been delegated, commits an offence punishable under any law "
    "enacted by Parliament.",
    "every person who, otherwise than in the course of such person's lawful duty, "
    "directly or indirectly by himself or by or with any other person, in any "
    "manner whatsoever influences or attempts to influence or interferes with any "
    "decision of the Public Service Commission, or a Committee or a public officer "
    "to whom the Commission has delegated any power, or to so influence any member "
    "of the Commission or a Committee, shall be guilty of an offence."
)

# QAC_212: 'Commissioners of the High Court' not 'additional Judges'; 'the subject of Justice'
replace_in_answer(
    get_orig(212),
    "Article 111A provides that where the Minister in charge of Justice represents "
    "to the President that it is expedient that the number of Judges exercising the "
    "jurisdiction and powers of the High Court in any judicial zone should be "
    "temporarily increased, the President may, on the recommendation of the "
    "Judicial Service Commission, appoint additional Judges of the High Court for "
    "that judicial zone for a specified period.",
    "Article 111A provides that where the Minister in charge of the subject of "
    "Justice represents to the President that it is expedient that the number of "
    "Judges exercising the jurisdiction and powers of the High Court in any judicial "
    "zone should be temporarily increased, the President may, on the recommendation "
    "of the Judicial Service Commission, by warrant appoint one or more "
    "Commissioners of the High Court to exercise the jurisdiction and powers of "
    "the High Court within such judicial zone."
)

# QAC_215: (c) 'any immovable property'; (d) 'licence, registration or other authorization, by or under any written law'; remove fabricated (e)(f)
replace_in_answer(
    get_orig(215),
    "(c) the right to own immovable property; "
    "(d) the right to engage in any trade or profession which requires a licence or "
    "permit from the State; (e) the right to remain in Sri Lanka; and "
    "(f) the right to vote at any election or referendum.",
    "(c) the right to own any immovable property; "
    "(d) the right to engage in any trade or profession which requires a licence, "
    "registration or other authorization, by or under any written law."
)

# QAC_221: Governor's reservation power is correct but 'inconsistent with national law' is imprecise
replace_in_answer(
    get_orig(221),
    "The Governor shall reserve any statute inconsistent with national law for the "
    "consideration of the President.",
    "The Governor shall reserve for the consideration of the President any statute "
    "which in his opinion would, if assented to, be inconsistent with the "
    "provisions of any law made by Parliament with respect to a matter in the "
    "Concurrent List."
)

# QAC_224: wrong qualifications – 'finance' and 'engineering' are not listed; should be 'accountancy'
replace_in_answer(
    get_orig(224),
    "of whom at least three must be persons who have had experience in finance, law, "
    "engineering, procurement, or public administration.",
    "of whom at least three members shall be persons who have had proven experience "
    "in procurement, accountancy, law or public administration."
)

# QAC_238: Article 12(4) -> 12(3)
replace_in_answer(get_orig(238), "Article 12(4) provides", "Article 12(3) provides")

# QAC_254: Article 14(1)(j) -> 14(1)(i)
replace_in_answer(get_orig(254), "Article 14(1)(j) provides", "Article 14(1)(i) provides")

# QAC_255: Art 15(3) only covers 'racial and religious harmony' – remove added grounds
replace_in_answer(
    get_orig(255),
    "in the interests of racial and religious harmony or in relation to national "
    "security, public order, the protection of public health or morality, or for "
    "the purpose of securing due recognition and respect for the rights and "
    "freedoms of others.",
    "in the interests of racial and religious harmony only."
)

# QAC_258: Art 15(6) only says 'national economy' for freedom of movement; fix 14(1)(j)->14(1)(i)
replace_in_answer(
    get_orig(258),
    "may be restricted by law in the interests of national security, public order, "
    "or for the purpose of securing due recognition and respect for the rights and "
    "freedoms of others, or for the protection of public health.",
    "may be restricted by law in the interests of national economy."
)
replace_in_answer(
    get_orig(258),
    "The freedom to return to Sri Lanka (Article 14(1)(j)) is similarly subject to "
    "restrictions in the interests of national security.",
    "The freedom to return to Sri Lanka (Article 14(1)(i)) is subject to the "
    "broader restrictions applicable under Article 15(7), which covers all "
    "fundamental rights and permits restrictions in the interests of national "
    "security, public order, and the protection of public health or morality."
)

# QAC_259: 14(1)(j) -> 14(1)(i); 15(6) is 'national economy' not 'national security, public order'
replace_in_answer(
    get_orig(259),
    "Article 14(1)(j) recognises every citizen's freedom to return to Sri Lanka. "
    "Article 15(6) permits restrictions on freedom of movement in the interests of "
    "national security, public order, or to protect the rights and freedoms of "
    "others, but such restrictions must be prescribed by law.",
    "Article 14(1)(i) recognises every citizen's freedom to return to Sri Lanka. "
    "Article 15(6) permits restrictions on the freedom of movement (Article "
    "14(1)(h)) only in the interests of national economy, while Article 15(7) "
    "permits broader restrictions on all fundamental rights, including the freedom "
    "to return, in the interests of national security, public order, the protection "
    "of public health or morality, or to secure due recognition of the rights and "
    "freedoms of others; but all such restrictions must be prescribed by law."
)

# QAC_260: Art 15(4) only covers 'racial and religious harmony or national economy'
replace_in_answer(
    get_orig(260),
    "in the interests of racial and religious harmony, national security, public "
    "order, the protection of public health or morality, or in relation to the "
    "national economy.",
    "in the interests of racial and religious harmony or in relation to the national "
    "economy."
)

# QAC_272: Art 15(2) does not include 'national security' or 'rights of others'
replace_in_answer(
    get_orig(272),
    "in the interests of racial and religious harmony, national security, use of "
    "the parliamentary privilege, contempt of court, defamation, or incitement to "
    "an offence, or for the purpose of securing due recognition and respect for "
    "the rights and freedoms of others.",
    "in the interests of racial and religious harmony or in relation to "
    "parliamentary privilege, contempt of court, defamation or incitement to an "
    "offence."
)

# QAC_277: add Art 21 reference
replace_in_answer(
    get_orig(277),
    "The right to receive education and sit for examinations in a chosen medium of "
    "instruction flows from the language rights in Chapter IV of the Constitution.",
    "Article 21 directly provides that every person is entitled to be educated "
    "through the medium of either of the National Languages."
)

# QAC_292: Article 27(2)(e) -> 27(13); fix content
replace_in_answer(
    get_orig(292),
    "Article 27(2)(e) directs the State to ensure that children and young persons "
    "are protected from exploitation and from moral and material abandonment.",
    "Article 27(13) directs the State to promote with special care the interests "
    "of children and youth, so as to ensure their full development, physical, "
    "mental, moral, religious and social."
)

# QAC_300: wrong article (40(1)(a) vs 40(3)(c)); wrong content (Parliament elects vs PM acts)
replace_in_answer(
    get_orig(300),
    "Article 40(1)(a) provides that if the office of President becomes vacant prior "
    "to the expiration of the term, Parliament shall elect as President one of its "
    "Members who is qualified to be elected to the office of President. The person "
    "elected holds office only for the unexpired period of the term of the President "
    "vacating office.",
    "Article 40(3)(c) provides that if, at any time between the close of the poll "
    "at a presidential election and the declaration of the result, a vacancy in the "
    "office of President occurs by reason of the death of a candidate, the Prime "
    "Minister shall act in the office of President during the period between the "
    "occurrence of such vacancy and the assumption of office by the new President, "
    "and shall appoint one of the other Ministers of the Cabinet to act in the "
    "office of Prime Minister. If the office of Prime Minister is then vacant or "
    "the Prime Minister is unable to act, the Speaker shall act in the office of "
    "President."
)

# QAC_308: simplistic 'consult AG' – actual process is Judge's report -> AG advice -> Minister of Justice -> President
replace_in_answer(
    get_orig(308),
    "However, the proviso states that before granting a pardon or substituting a "
    "lesser punishment for an offender sentenced to death, the President shall "
    "consult the Attorney-General.",
    "However, before exercising the pardon power in relation to a person condemned "
    "to death, the President shall cause a report to be made by the Judge who tried "
    "the case, forward that report to the Attorney-General for advice, and then "
    "have the report with the Attorney-General's advice sent to the Minister in "
    "charge of the subject of Justice, who shall forward it with his recommendation "
    "to the President."
)

# QAC_320: answer text fixes (articles 107-111 -> 107; remove 'retirement age'; fix appointment description)
replace_in_answer(
    get_orig(320),
    "Articles 107–111 provide for the appointment, tenure and removal of judges. "
    "Judges of the Supreme Court and Court of Appeal hold office during good "
    "behaviour until a specified retirement age and may only be removed by an "
    "address of Parliament.",
    "Article 107 provides that the Chief Justice, the President of the Court of "
    "Appeal and every other judge of the Supreme Court and the Court of Appeal "
    "shall be appointed by the President subject to the approval of the "
    "Constitutional Council, by warrant under his hand. Every such judge holds "
    "office during good behaviour and may only be removed by an order of the "
    "President made after an address of Parliament supported by a majority of the "
    "total number of Members of Parliament."
)

# ── 7. verify counts and save ─────────────────────────────────────────────────

data["items"] = items
data["num_items"] = len(items)

with open(DATASET_PATH, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(items)} items to {DATASET_PATH}")
print("Done. Backup at", BACKUP_PATH)
