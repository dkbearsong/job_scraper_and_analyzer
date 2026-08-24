#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for Job Scraper & Analyzer RAG System.

Exposes RAG retrieval, application tailoring, and job insights tools via MCP JSON-RPC 2.0.
"""

import sys
import json
import asyncio
import logging
from typing import Dict, Any

from app.postgres_mgr import PostgresManager
from app.pull_data import DataPuller
from app.ai_engine import AIEngine
from app.rag_engine import RAGEngine
from app.config_utils import _load_user_config
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

load_dotenv()
user_config = _load_user_config()
host = user_config.get("db_host") or os.getenv("DB_HOST") or "localhost"
port = user_config.get("db_port") or os.getenv("DB_PORT") or "5432"
user = user_config.get("db_user") or os.getenv("DB_USER") or "postgres"
password = user_config.get("db_password") or os.getenv("DB_PASSWORD") or ""
db_name = user_config.get("db_name") or os.getenv("DB_NAME") or "web_scraper_db"

_dp = None
_ai = None
_rag = None

def get_dp():
    global _dp
    if _dp is None:
        _dp = DataPuller(dbname=db_name, user=user, password=password, host=host, port=str(port))
    return _dp

def get_rag():
    global _rag, _ai
    if _rag is None:
        if _ai is None:
            _ai = AIEngine(default_provider_name="lm_studio")
        _rag = RAGEngine(dp=get_dp(), ai_engine=_ai)
    return _rag

TOOLS = [
    {
        "name": "search_job_corpus",
        "description": "Perform hybrid vector + metadata search across scraped jobs, extracted requirements, and LLM evaluations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query or required tech stack (e.g. 'Kubernetes Python Remote')"},
                "work_type": {"type": "string", "description": "Optional work arrangement filter (Remote, Hybrid, Onsite)"},
                "top_k": {"type": "integer", "default": 5, "description": "Number of results to return"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_job_insights",
        "description": "Retrieve full extractions, cheap/strong LLM fit scores, red flags, and driving points for a specific job ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "Target job ID"}
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "generate_application_strategy",
        "description": "Generate a tailored cover letter hook, resume bullet modifications, and interview talking points for a job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer", "description": "Target job ID"}
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "query_candidate_profile",
        "description": "Search the candidate's indexed resume and profile background chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Skills or project query"}
            },
            "required": ["query"]
        }
    }
]

async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_job_corpus":
        query = arguments.get("query", "")
        work_type = arguments.get("work_type")
        top_k = arguments.get("top_k", 5)
        results = get_rag().hybrid_search(query_text=query, work_type=work_type, top_k=top_k)
        return {"content": [{"type": "text", "text": json.dumps(results, indent=2)}]}

    elif name == "get_job_insights":
        job_id = arguments.get("job_id")
        rows = get_dp().conn.execute_sql(
            """
            SELECT j.id, j.job_name, c.company_name, j.link, j.pay_range, j.flexibility,
                   j.requirements, j.responsibilities,
                   clr.fit_score, clr.decision, clr.strengths, clr.concerns,
                   slr.final_score, slr.priority, slr.red_flags, slr.tailoring_notes, slr.driving_points
            FROM job j
            LEFT JOIN company c ON j.company_id = c.id
            LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id
            LEFT JOIN strong_llm_results slr ON j.id = slr.job_id
            WHERE j.id = %s;
            """,
            (job_id,),
            fetch=True
        )
        if not rows:
            return {"content": [{"type": "text", "text": f"Job ID {job_id} not found."}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(dict(rows[0]), indent=2, default=str)}]}

    elif name == "generate_application_strategy":
        job_id = arguments.get("job_id")
        if job_id is None:
            return {"content": [{"type": "text", "text": "Missing required argument: job_id"}], "isError": True}
        try:
            job_id = int(job_id)
        except (TypeError, ValueError):
            return {"content": [{"type": "text", "text": "Invalid job_id: must be an integer."}], "isError": True}
        res = await get_rag().generate_tailored_application(job_id)
        return {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}

    elif name == "query_candidate_profile":
        query = arguments.get("query", "")
        results = get_rag().hybrid_search(query_text=query, top_k=3)
        candidate_chunks = [r for r in results if r.get("job_id") is None]
        return {"content": [{"type": "text", "text": json.dumps(candidate_chunks, indent=2)}]}

    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}

async def run_mcp_server():
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            request = json.loads(line.decode("utf-8"))
        except Exception:
            continue

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "job-rag-mcp-server", "version": "1.0.0"}
                }
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                tool_res = await handle_call_tool(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": tool_res
                }
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"}
            }

        out_bytes = (json.dumps(response) + "\n").encode("utf-8")
        writer.write(out_bytes)
        await writer.drain()

if __name__ == "__main__":
    asyncio.run(run_mcp_server())
