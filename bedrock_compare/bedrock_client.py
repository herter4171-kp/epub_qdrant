"""Query Bedrock Knowledge Base using boto3 directly.

Usage:
    from bedrock_client import BedrockKBClient

    client = BedrockKBClient(kb_id="RQMBIXUSXH")
    results = client.query("your query text", number_of_results=10)
    # Returns: [{"content": {"text": "...", "type": "..."}, "location": {...}, "score": 0.42}, ...]
"""

import boto3
from typing import Any, Dict, List


class BedrockKBClient:
    """Client for querying Amazon Bedrock Knowledge Bases via boto3."""

    def __init__(
        self,
        kb_id: str,
        region: str = "us-gov-west-1",
        profile: str = "default",
    ):
        self.kb_id = kb_id
        self.session = boto3.Session(profile_name=profile, region_name=region)
        self.client = self.session.client("bedrock-agent-runtime")

    def query(
        self,
        text: str,
        number_of_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge base and return raw retrieval results.

        Args:
            text: The query text.
            number_of_results: Number of results to return.

        Returns:
            List of dicts with keys: content, location, score
        """
        response = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": number_of_results,
                }
            },
            retrievalQuery={"text": text},
        )
        return response.get("retrievalResults", [])