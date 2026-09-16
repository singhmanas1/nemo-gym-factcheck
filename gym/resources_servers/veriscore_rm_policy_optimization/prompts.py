QUALITY_RM_PROMPT_TEMPLATE = """You are a generative reward model that evaluates the quality and factual accuracy of AI responses.

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


VERISCORE_ATOMIZATION_PROMPT_TEMPLATE = """You are trying to verify how factual a piece of text is. To do so, you need to break down a sentence and extract important fine-grained facts mentioned in the sentence. Each fact should be verifiable against reliable external world knowledge. Any story, personal experiences, hypotheticals, subjective statements, suggestions, advice, instructions, and other such content should not be included in the list. Biographical, historical, scientific, and other such texts are not personal experiences or stories. You should extract verifiable facts from them. Each fact should describe either one single event or one single state with necessary time and location information. Quotations should be extracted verbatim with the source when available. Listed references should be ignored.

Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. Focus on named entities and numbers in the sentence, and extract relevant information from the sentence. Other sentences are only context for recovering pronouns, definite phrases, and so on. Each fact should be understandable on its own and require no additional context. This means that all entities must be referred to by name rather than by pronoun. Use the name of entities rather than definite noun phrases whenever possible. If a definite noun phrase is used, add modifiers such as an embedded clause or prepositional phrase. Each fact must be situated within relevant temporal and location context whenever needed. Keep each fact to one sentence with zero or at most one embedded clause. You do not need to justify what you extract.

If there is no verifiable fact in the sentence, write "No verifiable claim."

Here are some examples:

Text: The sweet potato or sweetpotato (Ipomoea batatas) is a dicotyledonous plant that belongs to the bindweed or morning glory family, Convolvulaceae. <SOS>Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.<EOS> The young shoots and leaves are sometimes eaten as greens.
Sentence to be focused on: Its large, starchy, sweet-tasting tuberous roots are used as a root vegetable.
Facts:
- Sweet potatoes' roots are large.
- Sweet potatoes' roots are starchy.
- Sweet potatoes' roots are sweet-tasting.
- Sweet potatoes' roots are tuberous.
- Sweet potatoes' roots are used as a root vegetable.

Text: <SOS>After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects.<EOS> He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished.
Sentence to be focused on: After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects.
Facts:
- Michelangelo achieved the success of the David in 1504.
- After 1504, Michelangelo's work consisted almost entirely of vast projects.

Text: After the success of the David in 1504, Michelangelo's work consisted almost entirely of vast projects. He was attracted to these ambitious tasks while at the same time rejecting the use of assistants, so that most of these projects were impractical and remained unfinished. <SOS>In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.<EOS> Both murals recorded military victories by the city.
Sentence to be focused on: In 1504 he agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall to form a pair with another just begun by Leonardo da Vinci.
Facts:
- In 1504, Michelangelo agreed to paint a huge fresco for the Sala del Gran Consiglio of the Florence city hall.
- Around 1504, Leonardo da Vinci just began a mural for the Florence city hall.

Text: I (27f) and my fiance "Leo" (27m) decided to let my FSIL "Maya" (32f) stay at our house because she needed space from her husband due to some relationship struggles they're having. We planned to test wedding cake samples along with Maya. <SOS>However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.<EOS>
Sentence to be focused on: However, when I came home from work to see Leo yelling at Maya, the box the samples came in wide open on the living room table, and Maya arguing with him.
Facts:
No verifiable claim.

Text: <SOS>Major depressive disorder (MDD), also known as depression, is a mental disorder.<EOS>
Sentence to be focused on: Major depressive disorder (MDD), also known as depression, is a mental disorder.
Facts:
- Major depressive disorder is also known as depression.
- Major depressive disorder is a mental disorder.

Text: The 1937 Fox vault fire was a major fire in a 20th Century Fox film storage facility in Little Ferry, New Jersey on 9 July 1937. It was caused by the spontaneous combustion of nitrate film stored in inadequately-ventilated vaults. The fire resulted in one death and two injuries, and destroyed all of the film present. <SOS>This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.<EOS>
Sentence to be focused on: This fire was responsible for the loss of most of the silent films produced by Fox Film Corporation before 1932.
Facts:
- Fox Film Corporation produced silent films before 1932.
- The 1937 Fox vault fire caused the loss of most of the silent films produced by Fox Film Corporation before 1932.

Text: <SOS>Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.<EOS> When he said "you can't get your youth back," he meant it.
Sentence to be focused on: Garnett had spent well over a decade with the Minnesota Timberwolves, and while he stayed loyal to that team, he found little success there.
Facts:
- Kevin Garnett spent over a decade with the Minnesota Timberwolves.
- Kevin Garnett was loyal to the Minnesota Timberwolves.
- Kevin Garnett found little success with the Minnesota Timberwolves.

Extract *verifiable atomic* facts.

Text: {snippet}
Sentence to be focused on: {sentence}
Facts:"""


VERISCORE_JUDGE_PROMPT_TEMPLATE = """You are verifying whether a claim is supported by retrieved evidence.

Use only the provided evidence. Do not use outside knowledge.

Labels:
- Supported: the evidence directly supports the claim.
- Contradicted: the evidence directly contradicts the claim.
- Inconclusive: the evidence is missing, ambiguous, unrelated, or insufficient.

Claim:
{claim}

Evidence:
{evidence}

Write a short explanation in one or two sentences. Then end with exactly one final verdict line:
###Supported###
###Contradicted###
###Inconclusive###
"""
