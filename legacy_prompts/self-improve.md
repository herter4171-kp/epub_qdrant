# self-improve
Always use the caveman skill.  Our goal is to improve mcp_servers/retrieval.

## Agentic AI Info Search Tool
* Tool is agent-lookup
* Tools
    * Get collections: Returns list of named collections
    * Query: Standard retrieval with top k
        * Always use search mode for now
        * Target all collections for now
        * Always reply with inline numeric citations from a bibleography at the end
    * Get context: Get content from a source near where semantic search hit
        * Avoid section title unless it is defined
        * Query likely helps if section is not defined
* General research task outline
    * Start with a single prompt
    * Devise a series of 3-6 questions with varied keyword use based on the prompt
    * Query those to get a comprehensive listing of sourced data
    * For highly interesting results,
        * Use get context to see if there's anything nearby worth investigating
        * Search by just the title or other similar metadata to see if we find anything else
    * Taking all findings, establish section names for your final reply based on results and original prompt's sub-questions
    * Fill out the sections with inline citations and bibliography at the end
    * If you were instructed, write a markdown file of your resulting statement


Your goal is to identify methods and strategies to make effective use of metadata, especially amongst disparate data sources bearing different shemas.  You will not modify any file except the specified markdown mentioned below.  Steps are:
1. Size up the codebase
    * Embedding techniques
    * Retrieval methods
    * MCP server architecture
1. Research tasks (using agent-lookup MCP tool)
    * Establish a detailed outline of our status quo in the codebase
    * Identify the fine details of what will make our goal achievable and effective
1. Search online (using web search MCP tool) for additional
    * Prospective libraries and tools
    * Reference implementations
1. Write a markdown file first-self-improve.md in the top working directory documenting the main points above

## Clarifying Points
The markdown file is a new file, not this rule file.  We want anything we get that we find applicable.  Our tools seem limited.  The user has no way of knowing metadata, and the MCP server does nothing to use it effectively.  We know we need to embed with metadata, but we are at a loss on how to retrieve it.  

Payload example for books:

{
"text":
"Agentic AI in Enterprise"
"book_title":
"Agentic AI in Enterprise"
"section_title":
"(no title)"
"chapter_index":
0
"section_index":
0
"chunk_index":
0
"token_count":
6
"source_file":
"agenticaiinenterprise.epub"
"publisher":
"Apress"
"language":
"en"
"isbn":
"urn:isbn:979-8-8688-1542-3"
}

Payload example for papers:

{
"text":
"Published as a conference paper at ICLR 2021 ALFW …"
"arxiv_id":
"2010.03768"
"title":
"ALFWorld: Aligning Text and Embodied Environments …"
"category":
"application-papers"
"subcategory":
"embodied-agents"
"authors":
"Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, …"
"publish_date":
"2020-10-08"
"chunk_index":
0
"chunk_count":
50
"token_count":
501
"source_file":
"2010_03768.pdf"
}