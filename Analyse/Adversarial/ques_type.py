import json
import pandas as pd
import numpy as np
import re


# ----------------------------------------------
#  Question Type Extraction
# ----------------------------------------------

def get_q_type(q):
    q = q.lower()

    if re.search(r"\bwho\b", q):
        return "who"
    if re.search(r"\bif\b", q):
        return "if"
    if re.search(r"\bwhere\b", q):
        return "where"
    if re.search(r"\bwhen\b", q) or re.search(r"\bwhat year\b", q) or re.search(r"\bin which year\b", q):
        return "when"
    if re.search(r"\bwhy\b", q) or re.search(r"\bwhat caused\b", q):
        return "why"
    if re.search(r"\bhow\b", q):
        return "how"
    if re.search(r"\bwhat\b", q) or re.search(r"\bwhich\b", q):
        return "what"

    return "other"

# ---------------------------------------------------
# Question Decomposition Categories
# ---------------------------------------------------

def classify_decomposition_type(q):

    q = q.lower()

    # --- temporal questions ---
    if any(kw in q for kw in [
        "when", "what year", "which year", "what date", "in what year",
        "how long", "how many years", "how old", "since when", "until when"
    ]):
        return "temporal"

    # --- causal questions ---
    if any(kw in q for kw in [
        "why", "cause", "caused", "what led to", "reason for", "due to",
        "how did this happen", "effect of", "result of"
    ]):
        return "causal"

    # --- counting questions ---
    if any(kw in q for kw in [
        "how many", "how much", "number of", "count", "total"
    ]):
        return "counting"

    # --- (relational) ---
    if any(kw in q for kw in [
        "both", "either", "first", "second", "before", "after",
        "relationship", "compare", "between",
        "combined", "together", "or" "," ,"because", "and"
    ]):
        return "compositional"

    # --- entity lookup questions ---
    if any(kw in q for kw in [
        "who", "where", "what", "which", "name the"
    ]):
        return "entity lookup"

    # fallback
    return "other"



# ----------------------------------------------
#  Multi-hop Question Detection
# ----------------------------------------------


Q_WORDS = r"(if |who|what|when|where|why|how|which)"

def count_Q_WORDS(q):
    # print("counting Q words in question: ", re.findall(Q_WORDS, q.lower()))
    return len(re.findall(Q_WORDS, q.lower()))

def detect_multi_questions(q):
    q = q.lower().strip()

    wh_count = count_Q_WORDS(q)
    CONNECTOR_PATTERNS = [" and ", " & ", " then ",  ", and", ";", " also ", " besides ", " as well as ", " if "," or "," that ", " that is ", " that are ", " that were ", " that are not ", " that aren't ", " that weren't "]

    has_connector = any(re.search(p, q) for p in CONNECTOR_PATTERNS)


    # split into clauses
    clauses = re.split(r"\band\b|\b or\b|,|;|if", q)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 3]
    
    # count clauses that contain a WH word
    clause_wh_count = sum(1 for c in clauses if re.search(Q_WORDS, c))

    # strong signals
    if clause_wh_count >= 2:
        # print( q, clauses, clause_wh_count,)
        return True
    if wh_count >= 1 and has_connector:
        # print(q, wh_count, has_connector)

        return True
    if wh_count >= 2:
        # print(q, wh_count)
        return True
    # else :
    #     print(q, "wh_count: ", wh_count,"has_connector: ",  has_connector,"clause wh count: ", clause_wh_count)
    return False


