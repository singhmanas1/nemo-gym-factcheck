SEARCH_WIKI_TOOL = [
    {
        "type": "function",
        "name": "search_wiki",
        "description": "Search Wikipedia for the given query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


FACT_CHECKER_POLICY_PROMPT_TEMPLATE = """You are a generative reward model that checks responses for factual inaccuracies. You will be given one response to an input conversation.

Based on the response, you need to first identify if the response is factually correct or not.
If the response is not factually accurate, you must list all factual inaccuracies in the response. Each factual inaccuracy should be a separate item in the list.

Please call the search_wiki tool to search for information from Wikipedia, in order to check that the response is factually correct.

Your output should have the following format:

[The Begin of Analysis on Response 1]
Analysis of Response 1.
[The End of Analysis on Response 1]

[Beginning of Factuality Prediction]
YES or NO. If the response is factually correct, respond with YES. If the response is not factually correct, respond with NO.
[End of Factuality Prediction]

[Beginning of Factual Errors]
<factual_inaccuracy_1>
<factual_inaccuracy_2>
...
[End of Factual Errors]

Now, here is the input conversation:

[Beginning of Input Conversation]
{input_conversation}
[End of Input Conversation]

[Beginning of Response 1]
{response}
[End of Response 1]

Please provide your analysis and decide if the response is factually correct or not. You can use the search_wiki tool to search for information from Wikipedia, in order to check that the response is factually correct.
"""


WIKI_SUMMARY_PROMPT_TEMPLATE = """You are preparing Wikipedia evidence for a factuality judge.

You will receive multiple retrieved Wikipedia articles. Your job is to turn them into a compact evidence packet that helps a downstream judge verify claims related to the search query.

Rules:
1. Use ONLY the provided Wikipedia content.
2. Do NOT add outside knowledge, assumptions, or speculation.
3. Extract only facts that are directly useful for checking claims related to the query.
4. Prefer concrete, atomic facts: names, dates, places, titles, numbers, definitions, and explicit relationships.
5. Remove irrelevant details, repetition, stylistic filler, and unnecessary background.
6. Combine evidence across articles only when it improves verification.
7. If the articles disagree, are ambiguous, or fail to answer the query, state that clearly.
8. Do NOT write a general summary of the articles.
9. Keep the output short and evidence-focused.

Inputs:

[Search Query]
{query}

[Wikipedia Articles]
{wiki_content}

Return output in exactly this format:

[Query-Relevant Evidence]
- [<article title>] <atomic fact>
- [<article title>] <atomic fact>
- [<article title>] <atomic fact>

[Supported By Retrieved Evidence]
- <claim or point that the retrieved evidence clearly supports>

[Not Established By Retrieved Evidence]
- <claim or point that the retrieved evidence does not establish, leaves ambiguous, or contradicts>

Requirements:
- Include 1 to 5 evidence bullets, ordered from most useful to least useful.
- Each bullet must contain only one atomic fact.
- Always include the article title.
- Prefer evidence that directly answers the query over broad background facts.
- If the articles do not provide useful evidence, say so explicitly.

If there is no useful evidence, return:

[Query-Relevant Evidence]
- none

[Supported By Retrieved Evidence]
- The retrieved Wikipedia articles do not clearly answer the query.

[Not Established By Retrieved Evidence]
- The retrieved content is too weak, broad, conflicting, or unrelated to verify the query.
"""
