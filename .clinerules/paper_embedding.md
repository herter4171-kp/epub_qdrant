# Paper Retrieval
## Introduction
We need to gather papers from the submodule ai-agent-papers.  Ignore any Chinese that you see.  There are markdown files in the following folders
* agent-frameworks
* application-papers
* capability-papers
* lectures
* newsletters

You can see the markdown files via:
```
tree ai-agent-papers/{agent-frameworks,application-papers,capability-papers,lectures,newsletters}
```

The directory tree should be of use in metadata.  Each directory in the bulleted list above should be a broad categorial label.  The markdown names should correspond to a sub-category, but ensure similar are lumped together, like deep research under newsletters and deep research agents under application-papers.  Ask if unsure.  Metadata will be more important to worry about once we extract the pdfs.

Now, each markdown file will have a list of papers.  Your job is to extract the link from arxiv.org.  For example, you would want to extract metadata from https://arxiv.org/abs/2602.04634 and download the PDF from https://arxiv.org/pdf/2602.04634.  Alternatively, you may need to consider getting metadata via [API](https://info.arxiv.org/help/api/basics.html#using).

Be sure to put an adjustable 1 second delay or otherwise implement respect for the likely rate limiting we will encounter.

Your task is to
1. Ensure a new directory ./downloads
1. Emulate the directory structure of the submodule repo
1. Download all of the PDFs
1. In each, create a CSV with your decided upon metadata entries for that sub-directory as columns with relative path to file at end

The exact way you go about it isn't important.  What we NEED to be sure of is that we get all of the PDFs in a similar directory structure with metadata consistent for the given directory.  If it's easiest to just download one paper from each location, so be it.
