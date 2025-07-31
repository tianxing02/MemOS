COT_DECOMPOSE_PROMPT = """
I am an 8-year-old student who needs help analyzing and breaking down complex questions. Your task is to help me understand whether a question is complex enough to be broken down into smaller parts.

Requirements:
1. First, determine if the question is a decomposable problem. If it is a decomposable problem, set 'is_complex' to True.
2. If the question needs to be decomposed, break it down into 1-3 sub-questions. The number should be controlled by the model based on the complexity of the question.
3. For decomposable questions, break them down into sub-questions and put them in the 'sub_questions' list. Each sub-question should contain only one question content without any additional notes.
4. If the question is not a decomposable problem, set 'is_complex' to False and set 'sub_questions' to an empty list.
5. You must return ONLY a valid JSON object. Do not include any other text, explanations, or formatting.

Here are some examples:

Question: Who is the current head coach of the gymnastics team in the capital of the country that Lang Ping represents?
Answer: {{"is_complex": true, "sub_questions": ["Which country does Lang Ping represent in volleyball?", "What is the capital of this country?", "Who is the current head coach of the gymnastics team in this capital?"]}}

Question: Which country's cultural heritage is the Great Wall?
Answer: {{"is_complex": false, "sub_questions": []}}

Question: How did the trade relationship between Madagascar and China develop, and how does this relationship affect the market expansion of the essential oil industry on Nosy Be Island?
Answer: {{"is_complex": true, "sub_questions": ["How did the trade relationship between Madagascar and China develop?", "How does this trade relationship affect the market expansion of the essential oil industry on Nosy Be Island?"]}}

Please analyze the following question and respond with ONLY a valid JSON object:
Question: {query}
Answer:"""

PRO_MODE_WELCOME_MESSAGE = """
============================================================
🚀 MemOS PRO Mode Activated!
============================================================
✅ Chain of Thought (CoT) enhancement is now enabled by default
✅ Complex queries will be automatically decomposed and enhanced

🌐 To enable Internet search capabilities:
   1. Go to your cube's textual memory configuration
   2. Set the backend to 'google' in the internet_retriever section
   3. Configure the following parameters:
      - api_key: Your Google Search API key
      - cse_id: Your Custom Search Engine ID
      - num_results: Number of search results (default: 5)

📝 Example configuration at cube config for tree_text_memory :
   internet_retriever:
     backend: 'google'
     config:
       api_key: 'your_google_api_key_here'
       cse_id: 'your_custom_search_engine_id'
       num_results: 5
details: https://github.com/memos-ai/memos/blob/main/examples/core_memories/tree_textual_w_internet_memoy.py
============================================================
"""

SYNTHESIS_PROMPT = """
exclude memory information, synthesizing information from multiple sources to provide comprehensive answers.
I will give you chain of thought for sub-questions and their answers.
Sub-questions and their answers:
{qa_text}

Please synthesize these answers into a comprehensive response that:
1. Addresses the original question completely
2. Integrates information from all sub-questions
3. Provides clear reasoning and connections
4. Is well-structured and easy to understand
5. Maintains a natural conversational tone"""

MEMOS_PRODUCT_BASE_PROMPT = (
    "You are a knowledgeable and helpful AI assistant with access to user memories. "
    "When responding to user queries, you should reference relevant memories using the provided memory IDs. "
    "Use the reference format: [1-n:memoriesID] "
    "where refid is a sequential number starting from 1 and increments for each reference in your response, "
    "and memoriesID is the specific memory ID provided in the available memories list. "
    "For example: [1:abc123], [2:def456], [3:ghi789], [4:jkl101], [5:mno112] "
    "Do not use connect format like [1:abc123,2:def456]"
    "Only reference memories that are directly relevant to the user's question. "
    "Make your responses natural and conversational while incorporating memory references when appropriate."
)

MEMOS_PRODUCT_ENHANCE_PROMPT = """
# Memory-Enhanced AI Assistant Prompt

You are a knowledgeable and helpful AI assistant with access to two types of memory sources:

## Memory Types
- **PersonalMemory**: User-specific memories and information stored from previous interactions
- **OuterMemory**: External information retrieved from the internet and other sources

## Memory Reference Guidelines

### Reference Format
When citing memories in your responses, use the following format:
- `[refid:memoriesID]` where:
  - `refid` is a sequential number starting from 1 and incrementing for each reference
  - `memoriesID` is the specific memory ID from the available memories list

### Reference Examples
- Correct: `[1:abc123]`, `[2:def456]`, `[3:ghi789]`, `[4:jkl101]`, `[5:mno112]`
- Incorrect: `[1:abc123,2:def456]` (do not use connected format)

## Response Guidelines

### Memory Selection
- Intelligently choose which memories (PersonalMemory or OuterMemory) are most relevant to the user's query
- Only reference memories that are directly relevant to the user's question
- Prioritize the most appropriate memory type based on the context and nature of the query

### Response Style
- Make your responses natural and conversational
- Seamlessly incorporate memory references when appropriate
- Ensure the flow of conversation remains smooth despite memory citations
- Balance factual accuracy with engaging dialogue

## Key Principles
- Reference only relevant memories to avoid information overload
- Maintain conversational tone while being informative
- Use memory references to enhance, not disrupt, the user experience
"""
QUERY_REWRITING_PROMPT = """
I'm in discussion with my friend about a question, and we have already talked about something before that. Please help me analyze the logic between the question and the former dialogue, and rewrite the question we are discussing about.

Requirements:
1. First, determine whether the question is related to the former dialogue. If so, set "former_dialogue_related" to True.
2. If "former_dialogue_related" is set to True, meaning the question is related to the former dialogue, rewrite the question according to the keyword in the dialogue and put it in the "rewritten_question" item. If "former_dialogue_related" is set to False, set "rewritten_question" to an empty string.
3. If you decided to rewrite the question, keep in mind that the rewritten question needs to be concise and accurate.
4. You must return ONLY a valid JSON object. Do not include any other text, explanations, or formatting.

Here are some examples:

Former dialogue:
————How's the weather in ShangHai today?
————It's great. The weather in Shanghai is sunny right now. The lowest temperature is 27℃, the highest temperature can reach 33℃, the air quality is excellent, the pm2.5 index is 13, the humidity is 60%, and the northerly wind is at level 1.
Current question: What should I wear today?
Answer: {{"former_dialogue_related": True, "rewritten_question": "Considering the weather in Shanghai today, what should I wear?"}}

Former dialogue:
————I need a brief introduction to Oxford-Cambridge boat race.
————The race originated from a challenge in 1829 between Charles Merivale of Cambridge University and Charles Wordsworth of Oxford University. Oxford won the first race. The event became an annual tradition in 1856, with interruptions only during the World Wars and the 2020 COVID-19 pandemic. The women's race was added in 1927. The team members are full-time students of the two universities, including both novice rowers and experienced athletes such as Olympic champions and world champions.
————What is the international community's attitude towards the 2024 US election?
————The international community approached the 2024 U.S. election with a blend of pragmatism, anxiety, and strategic recalibration. Allies sought to mitigate risks from Trump's policies while maintaining cooperation, while adversaries like China and Russia capitalized on perceived U.S. decline to advance their agendas. Developing nations increasingly resisted U.S. dominance, advocating for a multipolar world. Ultimately, the election underscored the need for global actors to adapt to a more fragmented and unpredictable international order shaped by U.S. domestic politics.
Current question: In March 2025, after a magnitude 7.9 earthquake struck Myanmar, what assistance did the Chinese government provide?
Answer: {{"former_dialogue_related": False, "rewritten_question": ""}}

Former dialogue:
————I am an entry-level learner of large language models. Please recommend me three papers suitable for reading.
————For an entry-level learner of large language models (LLMs), here are three foundational papers that provide essential insights into the core concepts, architectures, and advancements in the field: "Attention Is All You Need", "Improving Language Understanding by Generative Pre-Training (GPT-1)", and "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". These papers will equip you with the foundational knowledge needed to explore more advanced topics in LLMs, such as scaling laws, instruction tuning, and multi-modal learning.
Current question: Of these three papers, which one do you recommend I start reading?
Answer: {{"former_dialogue_related": True, "rewritten_question": "Among the three papers \"Attention Is All You Need\", \"Improving Language Understanding by Generative Pre-Training (GPT-1)\" and \"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\", which one do you recommend I start reading?"}}

Former dialogue:
{dialogue}
Current question: {query}
Answer:"""
