SEARCH_WEB_TOOL = [
    {
        "type": "function",
        "name": "search_wiki",
        "description": "Search the web for the given query",
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

# Matches the prompt used during Tavily joint RM training (rlhf_veriscore_rm_train_tavily.jsonl).
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
