"""
Executive Analysis Report Generator.
Compiles data catalog summaries, analytical discoveries, detected anomalies,
and conversation histories into a clean downloadable HTML or Markdown report.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.agent.state import SessionState


def generate_executive_html_report(state: SessionState) -> str:
    """
    Compiles an executive-level analytical report summarizing session insights into HTML.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    active_name = state.active_dataset_name or "None"
    meta = state.metadata_cache.get(active_name)

    row_count = meta.row_count if meta else 0
    col_count = meta.column_count if meta else 0

    # Build schema summary rows
    schema_rows = ""
    if meta:
        for c in meta.columns:
            schema_rows += f"""
            <tr>
                <td><code>{c.name}</code></td>
                <td>{c.dtype}</td>
                <td>{c.null_percentage}%</td>
                <td>{c.distinct_count}</td>
                <td>{c.min_value if c.min_value is not None else '-'}</td>
                <td>{c.max_value if c.max_value is not None else '-'}</td>
            </tr>
            """

    # Build conversation records
    chat_rows = ""
    for msg in state.conversation_history:
        role_class = "user-msg" if msg.role == "user" else "assistant-msg"
        role_title = "User Query" if msg.role == "user" else "AI Analyst Insight"
        chat_rows += f"""
        <div class="message {role_class}">
            <h4>{role_title}</h4>
            <div class="content">{msg.content.replace(chr(10), '<br>')}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Data Analysis Report - {active_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        h1 {{
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 12px;
            margin-top: 0;
        }}
        .meta-bar {{
            display: flex;
            gap: 20px;
            background: #f1f5f9;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            font-size: 0.95rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }}
        th, td {{
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
        }}
        code {{
            background: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
        }}
        .message {{
            border-left: 4px solid #3b82f6;
            background: #f8fafc;
            padding: 14px 18px;
            margin-bottom: 16px;
            border-radius: 0 8px 8px 0;
        }}
        .user-msg {{
            border-left-color: #6366f1;
            background: #eef2ff;
        }}
        .user-msg h4 {{
            color: #4338ca;
            margin-top: 0;
            margin-bottom: 6px;
        }}
        .assistant-msg h4 {{
            color: #1e40af;
            margin-top: 0;
            margin-bottom: 6px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AI Data Analyst - Executive Summary Report</h1>
        <div class="meta-bar">
            <div><strong>Active Dataset:</strong> {active_name}</div>
            <div><strong>Total Records:</strong> {row_count:,}</div>
            <div><strong>Columns:</strong> {col_count}</div>
            <div><strong>Generated At:</strong> {now_str}</div>
        </div>

        <h2>1. Dataset Profile & Schema</h2>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Data Type</th>
                    <th>Null %</th>
                    <th>Distinct Values</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                {schema_rows}
            </tbody>
        </table>

        <h2>2. Analytical Dialogue & Findings</h2>
        {chat_rows if chat_rows else '<p>No analysis queries executed in this session.</p>'}

        <hr style="margin-top: 40px; border: 0; border-top: 1px solid #e2e8f0;">
        <p style="font-size: 0.8rem; color: #94a3b8; text-align: center;">
            Generated by Prism • Deterministic Architecture.
        </p>
    </div>
</body>
</html>
"""
    return html
