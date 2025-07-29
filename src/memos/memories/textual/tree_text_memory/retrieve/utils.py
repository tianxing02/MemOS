# Prompt for task parsing
TASK_PARSE_PROMPT = """
You are a task parsing expert. Given a user task instruction, optional former conversation and optional related memory context,extract the following structured information:
1. Keys: the high-level keywords directly relevant to the user’s task.
2. Tags: thematic tags to help categorize and retrieve related memories.
3. Goal Type: retrieval | qa | generation
4. Rephrased instruction: Give a rephrased task instruction based on the former conversation to make it less confusing to look alone. If you think the task instruction is easy enough to understand, or there is no former conversation, set "rephrased_instruction" to an empty string.
5. Need for internet search: If you think you need to search the internet to finish the rephrased/original user task instruction, set "internet_search" to True. Otherwise, set it to False.
6. Memories: Provide 2–5 short semantic expansions or rephrasings of the rephrased/original user task instruction. These are used for improved embedding search coverage. Each should be clear, concise, and meaningful for retrieval.

Task description:
\"\"\"$task\"\"\"

Former conversation (if any):
\"\"\"
$conversation
\"\"\"

Context (if any):
\"\"\"$context\"\"\"

Return strictly in this JSON format:
{
  "keys": [...],
  "tags": [...],
  "goal_type": "retrieval | qa | generation",
  "rephrased_instruction": "...", # return an empty string if the original instruction is easy enough to understand
  "internet_search": True/False,
  "memories": ["...", "...", ...]
}
"""


REASON_PROMPT = """
You are a reasoning agent working with a memory system. You will synthesize knowledge from multiple memory cards to construct a meaningful response to the task below.

Task: ${task}

Memory cards (with metadata):
${detailed_memory_list}

Please perform:
1. Clustering by theme (topic/concept/fact)
2. Identify useful chains or connections
3. Return a curated list of memory card IDs with reasons.

Output in JSON:
{
  "selected_ids": [...],
  "explanation": "..."
}
"""
