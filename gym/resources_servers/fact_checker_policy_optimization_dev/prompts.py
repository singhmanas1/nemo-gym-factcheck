import re
from typing import Optional

from resources_servers.fact_checker_policy_optimization.prompts import (  # noqa: F401
    FACT_CHECKER_POLICY_PROMPT_TEMPLATE,
    SEARCH_WIKI_TOOL,
    WIKI_SUMMARY_PROMPT_TEMPLATE,
)


def extract_query_and_response(checker_prompt: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract the user query and policy response from a checker prompt.

    The checker prompt is built by FACT_CHECKER_POLICY_PROMPT_TEMPLATE and
    has this structure:

        [Beginning of Input Conversation]
        [Begin of user Message]
        <user question>
        [End of user Message]
        [End of Input Conversation]

        [Beginning of Response 1]
        <policy response>
        [End of Response 1]

    Returns (query, response), either of which may be None if not found.
    """
    # Extract input conversation block
    conv_match = re.search(
        r"\[Beginning of Input Conversation\](.*?)\[End of Input Conversation\]",
        checker_prompt,
        flags=re.DOTALL,
    )
    query: Optional[str] = None
    if conv_match:
        conv_block = conv_match.group(1)
        # Strip [Begin of X Message] / [End of X Message] wrappers and collect text
        msg_texts = re.findall(
            r"\[Begin of \w+ Message\](.*?)\[End of \w+ Message\]",
            conv_block,
            flags=re.DOTALL,
        )
        query = "\n".join(t.strip() for t in msg_texts).strip() or None

    # Extract response block
    resp_match = re.search(
        r"\[Beginning of Response 1\](.*?)\[End of Response 1\]",
        checker_prompt,
        flags=re.DOTALL,
    )
    response: Optional[str] = resp_match.group(1).strip() if resp_match else None

    return query, response


WIKI_SUMMARY_WITH_CONTEXT_PROMPT_TEMPLATE = """You are a research assistant helping a factuality judge verify whether a model response is accurate.

You will be given:
- The original user request
- The model response being fact-checked
- A search query that was used to retrieve evidence relevant to that response
- The retrieved documents

---

Original user request:
{original_query}

---

Model response being fact-checked:
{original_response}

---

Search query:
{query}

---

Retrieved documents:
{wiki_content}

---

Your task: read the retrieved documents carefully and extract ALL information that is relevant to the search query. Then, for each specific factual claim in the model response that the retrieved documents can speak to, clearly state whether the evidence supports, contradicts, or is silent on that claim.

Instructions:
1. Use ONLY information from the retrieved documents. Do not add outside knowledge.
2. Extract detailed facts related to the search query — names, dates, roles, relationships, places, numbers, titles. Be thorough; do not discard details that might be relevant.
3. For each claim in the model response that touches on the search query topic, explicitly state what the evidence says about it.
4. If the evidence contradicts a claim in the response, state the contradiction clearly and precisely.
5. If the evidence confirms a claim, state that clearly.
6. If the evidence is silent or ambiguous on a claim, say so.
7. Do not summarize the documents generically — focus entirely on what helps verify or refute the response.

Return your output in exactly this format:

[Extracted Evidence]
- [<source url>] <detailed fact from the documents>
- [<source url>] <detailed fact from the documents>
(include as many bullets as needed — do not truncate relevant facts)

[Claims Confirmed By Evidence]
- "<exact or paraphrased claim from the response>" — confirmed by: <specific evidence>

[Claims Contradicted By Evidence]
- "<exact or paraphrased claim from the response>" — contradicted by: <specific evidence>

[Claims Not Addressed By Evidence]
- "<exact or paraphrased claim from the response>" — no relevant evidence found

If the retrieved documents contain no information relevant to the search query or the response, return:

[Extracted Evidence]
- none

[Claims Confirmed By Evidence]
- none

[Claims Contradicted By Evidence]
- none

[Claims Not Addressed By Evidence]
- All claims related to this search query are unaddressed by the retrieved documents.
"""
