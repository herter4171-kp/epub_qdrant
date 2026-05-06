# Server Testing: The Only Source of Truth

A server is up and operational **only when you have tested it yourself**.

Ping the IP? Check the health endpoint? Read the config? None of that is real. Until you send a real query and get back valid results, you are operating on faith, not evidence.

The agent-lookup MCP server's case-sensitivity bug on `papers-2048ctx-SAE` is proof. It reported connected. It accepted the query. It returned a 404 with a JSON error — not an error, a graceful failure disguised as a response. The server was "online" but the collection name it received was lowercase, and it silently refused. You wouldn't know unless you sent a real query and checked the actual response.

**Testing is the only source of truth.** Everything else is speculation.