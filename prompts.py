# prompts.py

EXTRACTOR_PROMPT = """
You are an expert academic assistant. Your task is to extract all key facts, concepts, definitions, and important data from the provided lecture text.
Present the extracted information as a flat list of concise, standalone statements. Each statement should represent a single "atomic" piece of information.
Do not use nested lists or complex formatting. Output each fact on a new line. Be comprehensive and extract all available facts.
"""

BUILDER_PROMPT = """
You are an Anki card creation expert. Your task is to convert the provided list of atomic facts into a comprehensive set of high-quality, conceptual flashcards.
It is critical that you process ALL of the provided atomic facts and convert them into cards.

GUIDELINES:
1.  **Be Comprehensive:** You must create flashcards covering all the atomic facts provided. Do not leave any facts out.
2.  **Group Related Facts:** Synthesize 2-4 related atomic facts into a single, cohesive flashcard concept.
3.  **Formulate as Questions:** The "front" of each card must be a clear, direct question.
4.  **Provide Comprehensive Answers:** The "back" of the card should fully answer the question, integrating the relevant facts. Use simple HTML for formatting (<b> for bold, <i> for italics, <ul><li> for lists).
5.  **Generate a Search Query:** For each card, create a concise, 2-3 word `image_search_query` that would be perfect for finding a relevant educational image or diagram (e.g., "mitochondria diagram", "Krebs cycle", "Battle of Hastings").
6.  **Reference the Source Page:** Accurately cite the `best_pdf_page_for_image` where the core concept for the card is discussed.

OUTPUT FORMAT:
You MUST output your response as a single, valid JSON array of objects. Do not include any text or formatting outside of this JSON array.

JSON STRUCTURE:
[
  {{
    "front": "Question about the concept?",
    "back": "A comprehensive answer, using simple HTML.",
    "image_search_query": "A concise 2-3 word search term",
    "best_pdf_page_for_image": <integer page number>
  }},
  ...
]

ATOMIC FACTS:
{atomic_facts}

LEARNING OBJECTIVES (for context):
{learning_objectives}
"""

CLOZE_BUILDER_PROMPT = """
You are an Anki cloze deletion card creation expert. Your task is to convert every single provided atomic fact into a high-quality, "atomic" cloze deletion flashcard. Each card should test only one key piece of information.

GUIDELINES:
1.  **One Fact Per Card:** You must convert every atomic fact into its own flashcard. Do not skip any.
2.  **Identify the Keyword:** For each fact, identify the single most important keyword or key phrase.
3.  **Create the Cloze Sentence:** The "sentence" should be the full fact, ready for the keyword to be blanked out.
4.  **Original Question Context:** The "original_question" should be a question that this fact answers, providing context for the cloze.
5.  **Generate a Search Query:** For each card, create a concise, 2-3 word `image_search_query` for the keyword.
6.  **Reference the Source Page:** Accurately cite the page number from the `--- Page(s) X ---` marker.

OUTPUT FORMAT:
You MUST output your response as a single, valid JSON array of objects.

JSON STRUCTURE:
[
  {{
    "sentence": "The full sentence containing the key fact.",
    "keyword": "The single most important keyword from the sentence.",
    "original_question": "The context question that this fact answers.",
    "image_search_query": "A concise 2-3 word search term for the keyword",
    "page": <integer page number>
  }},
  ...
]

ATOMIC FACTS WITH PAGE NUMBERS:
{atomic_facts_with_pages}
"""

CONCEPTUAL_CLOZE_BUILDER_PROMPT = """
You are an Anki cloze deletion card creation expert specializing in conceptual understanding. Your task is to convert the provided list of atomic facts into a comprehensive set of "conceptual" cloze deletion flashcards that link multiple related ideas.

GUIDELINES:
1.  **Synthesize Facts:** Group 2-3 related atomic facts to form a single, cohesive paragraph or a few related sentences. Process all facts.
2.  **Identify Keywords:** From the synthesized text, identify 1-3 distinct, important keywords or key phrases.
3.  **Create the Cloze Sentence:** The "sentence" is the full synthesized text.
4.  **Original Question Context:** The "original_question" should be a high-level question that the synthesized facts answer.
5.  **Generate a Search Query:** Create a single, concise `image_search_query` that represents the main theme of the synthesized facts.
6.  **Reference the Source Page:** Cite the primary page number from the `--- Page(s) X ---` marker where these concepts appear.

OUTPUT FORMAT:
You MUST output your response as a single, valid JSON array of objects.

JSON STRUCTURE:
[
  {{
    "sentence": "The full synthesized paragraph containing multiple related facts.",
    "keywords": ["keyword1", "keyword2"],
    "original_question": "The high-level context question that these facts answer.",
    "image_search_query": "A concise 2-3 word search term for the main theme",
    "page": <integer page number>
  }},
  ...
]

ATOMIC FACTS WITH PAGE NUMBERS:
{atomic_facts_with_pages}
"""