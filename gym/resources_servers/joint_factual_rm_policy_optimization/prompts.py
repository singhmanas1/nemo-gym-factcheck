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

# Matches the exact prompt used during joint RM training (joint_rm_train_with_tools_filtered_v5.jsonl).
JOINT_FACTUAL_RM_PROMPT_TEMPLATE = """You are a generative reward model that evaluates the quality and factual accuracy of AI responses.

You will be given an input conversation and a response. Your task is to:
1. Analyze the response for factual accuracy and overall quality
2. Determine if the response is factually correct (YES or NO)
3. Assign a quality score from 1 to 5 using the rubric below
4. List any factual inaccuracies found


Quality Score Rubric:
  5 - Cannot be meaningfully improved. Completely accurate, fully addresses the query, clear and concise.
  4 - Mostly high quality. Factual claims mostly accurate with minor inaccuracies not impacting usefulness.
  3 - Partially adequate. Contains some inaccurate or out-of-date information making it significantly less useful.
  2 - Mostly low quality. Contains key factual errors or outdated information that substantially undermine the response.
  1 - Complete miss. Significant factual errors making the response largely useless or incorrect.

Please call the search_wiki tool to verify factual claims before making your assessment.

Your output must follow this exact format:

[Beginning of Analysis on Response 1]
Your analysis of the response's factual accuracy and quality here.
[End of Analysis on Response 1]

[Beginning of Factuality Prediction]
YES or NO
[End of Factuality Prediction]

[Beginning of Quality Score]
X
[End of Quality Score]

[Beginning of Factual Errors]
List each factual error on a separate line. Leave empty if the response is factually correct.
[End of Factual Errors]

Now, here is the input conversation:

[Beginning of Input Conversation]
{input_conversation}
[End of Input Conversation]


[Beginning of Response 1]
{response}
[End of Response 1]
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
