"""Healthcare Contact Center Assistant - Streamlit App.

Native Snowflake Streamlit app running in Container Runtime that uses
direct internal DNS to call the Healthcare ReAct Agent running in SPCS.

Features LangGraph-level streaming for real-time progress updates.
"""

import json
import os
import time
from collections.abc import Generator

import requests
from snowflake.snowpark.context import get_active_session

import streamlit as st

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get Snowflake session
session = get_active_session()

# =============================================================================
# Constants - Internal SPCS DNS for Container Runtime
# =============================================================================
# When running in Container Runtime, we can access SPCS services directly via internal DNS
# Format: http://<service-name>.<hash>.svc.spcs.internal:<port>
SPCS_INTERNAL_DNS = "healthcare-agents-service.juiu.svc.spcs.internal"
SPCS_INTERNAL_PORT = 8000
SPCS_INTERNAL_URL = f"http://{SPCS_INTERNAL_DNS}:{SPCS_INTERNAL_PORT}"

# Detect if running in Container Runtime (has access to internal DNS)
IS_CONTAINER_RUNTIME = os.environ.get("SNOWFLAKE_CONTAINER_RUNTIME") is not None


# =============================================================================
# Custom CSS for better UI
# =============================================================================
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .badge-analyst { background-color: #4CAF50; color: white; }
    .badge-search { background-color: #2196F3; color: white; }
    .badge-both { background-color: #9C27B0; color: white; }
    .badge-default { background-color: #607D8B; color: white; }
    .service-status {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-ready { background-color: #4CAF50; color: white; }
    .status-error { background-color: #f44336; color: white; }
    .status-unknown { background-color: #9E9E9E; color: white; }
    .progress-step {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-left: 3px solid #2196F3;
        background-color: #f8f9fa;
        border-radius: 0 0.25rem 0.25rem 0;
    }
    .progress-step.thinking { border-left-color: #FF9800; }
    .progress-step.tool { border-left-color: #4CAF50; }
    .progress-step.complete { border-left-color: #9C27B0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Session State Initialization
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "member_id" not in st.session_state:
    st.session_state.member_id = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "execution_id" not in st.session_state:
    st.session_state.execution_id = None
if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True


# =============================================================================
# Service Status & URL Discovery
# =============================================================================
def check_service_status() -> dict:
    """Check if the SPCS service is running and get endpoint URL."""
    try:
        result = session.sql("CALL SYSTEM$GET_SERVICE_STATUS('HEALTHCARE_DB.STAGING.HEALTHCARE_AGENTS_SERVICE')").collect()
        if result:
            status_json = json.loads(result[0][0])
            if status_json and len(status_json) > 0:
                return {
                    "status": status_json[0].get("status", "UNKNOWN"),
                    "message": status_json[0].get("message", ""),
                    "image": status_json[0].get("image", "").split(":")[-1],
                }
        return {"status": "UNKNOWN", "message": "No status available", "image": ""}
    except Exception as e:
        return {"status": "ERROR", "message": str(e), "image": ""}


def get_service_endpoint_url() -> str | None:
    """Get the internal SPCS service endpoint URL.

    In Container Runtime: Use internal DNS directly (no SQL query needed).
    In Warehouse Runtime: Try SQL function (may fail due to network restrictions).
    """
    # Container Runtime: Direct internal DNS access
    if IS_CONTAINER_RUNTIME:
        return SPCS_INTERNAL_URL

    # Try internal DNS first (works if we have network access)
    try:
        # Quick connectivity check
        response = requests.head(f"{SPCS_INTERNAL_URL}/health", timeout=2)
        if response.status_code < 500:
            return SPCS_INTERNAL_URL
    except requests.exceptions.RequestException:
        pass  # Fall through to SQL method

    # Warehouse Runtime: Try SQL function approach
    try:
        result = session.sql("""
            SELECT SYSTEM$GET_SERVICE_ENDPOINT('HEALTHCARE_DB.STAGING.HEALTHCARE_AGENTS_SERVICE', 'query-api')
        """).collect()
        if result and result[0][0]:
            endpoint = result[0][0]
            return f"http://{endpoint}"
    except Exception as e:
        st.warning(f"Could not get service endpoint: {e}")

    return None


# =============================================================================
# Streaming SSE Client
# =============================================================================
def stream_agent_response(
    endpoint_url: str,
    query: str,
    member_id: str | None,
    execution_id: str | None,
) -> Generator[dict, None, None]:
    """Stream agent events via SSE from the /agents/stream endpoint.

    Args:
        endpoint_url: Base URL of the SPCS service
        query: User query
        member_id: Optional member ID
        execution_id: Thread ID for conversation continuation

    Yields:
        Parsed event dictionaries
    """
    url = f"{endpoint_url}/agents/stream"

    payload = {
        "query": query,
        "member_id": member_id,
        "tenant_id": "streamlit_app",
        "user_id": "streamlit_user",
        "thread_id": execution_id,
        "include_intermediate": True,
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()

            buffer = ""
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    buffer += chunk

                    # Parse SSE events (format: "data: {...}\n\n")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        if event_str.startswith("data: "):
                            try:
                                event_data = json.loads(event_str[6:])  # Remove "data: " prefix
                                yield event_data
                            except json.JSONDecodeError:
                                continue

    except requests.exceptions.RequestException as e:
        yield {"event_type": "error", "data": {"message": f"Connection error: {e}"}}


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.image(
        "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/brand-guidelines/logo-sno-blue-example.svg",
        width=150,
    )
    st.title("🏥 Healthcare Assistant")
    st.markdown("---")

    # Service Status
    st.header("🔗 Service Status")
    service_info = check_service_status()
    if service_info["status"] == "READY":
        st.markdown(
            '<div class="service-status status-ready">✅ Service Ready</div>',
            unsafe_allow_html=True,
        )
        if service_info["image"]:
            st.caption(f"Version: {service_info['image']}")
    elif service_info["status"] == "ERROR":
        st.markdown(
            '<div class="service-status status-error">❌ Service Error</div>',
            unsafe_allow_html=True,
        )
        st.caption(service_info["message"][:100])
    else:
        st.markdown(
            f'<div class="service-status status-unknown">⚠️ {service_info["status"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Member Information
    st.header("👤 Member Information")
    member_id = st.text_input(
        "Member ID",
        value=st.session_state.member_id or "",
        placeholder="e.g., 786924904",
        help="Enter your 9-digit member ID to get personalized information",
    )
    if member_id:
        st.session_state.member_id = member_id
        st.success(f"✅ Member: {member_id}")
    else:
        st.info("💡 Enter Member ID for personalized queries")

    st.markdown("---")

    # Settings
    st.header("⚙️ Settings")
    st.session_state.debug_mode = st.checkbox(
        "Show routing details",
        value=st.session_state.debug_mode,
        help="Display agent routing and execution details",
    )
    st.session_state.use_streaming = st.checkbox(
        "Enable streaming progress",
        value=st.session_state.use_streaming,
        help="Show real-time agent progress (tool calls, thinking, etc.)",
    )

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.execution_id = None
        st.rerun()

    st.markdown("---")

    # Help section
    with st.expander("❓ How to use"):
        st.markdown(
            """
            **Example queries:**
            - "Tell me about member 786924904"
            - "What claims does this member have?"
            - "What is the policy for prescription drugs?"
            - "How do I file an appeal?"
            - "Show top 3 members with highest premiums"
            - "Has this member called us before?"

            **Tips:**
            - Enter a 9-digit Member ID for personalized info
            - Enable debug mode to see tools used
            - Enable streaming to see real-time progress
            - Conversation continues automatically (context preserved)
            """
        )


# =============================================================================
# Main Chat Interface
# =============================================================================
st.title("Healthcare Contact Center Assistant")
st.markdown("Ask questions about your healthcare coverage, claims, and policies.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show metadata in debug mode
        if st.session_state.debug_mode and "metadata" in message and message["metadata"]:
            metadata = message["metadata"]
            tools_used = metadata.get("routing", "none") or "none"

            # Tool badges
            if "query_member_data" in tools_used.lower() or "analyst" in tools_used.lower():
                st.markdown(
                    '<span class="tool-badge badge-analyst">📊 ANALYST</span>',
                    unsafe_allow_html=True,
                )
            if "search_knowledge" in tools_used.lower() or "search" in tools_used.lower():
                st.markdown(
                    '<span class="tool-badge badge-search">🔎 SEARCH</span>',
                    unsafe_allow_html=True,
                )
            if tools_used == "none" or tools_used == "react":
                st.markdown(
                    '<span class="tool-badge badge-default">💬 DIRECT</span>',
                    unsafe_allow_html=True,
                )

            with st.expander("🔍 Execution Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tools Used", tools_used or "None")
                with col2:
                    st.caption(f"Execution ID: {metadata.get('execution_id', 'N/A')}")

                if metadata.get("analyst_results"):
                    st.subheader("📊 Analyst Results")
                    st.json(metadata["analyst_results"])

                if metadata.get("search_results"):
                    st.subheader("🔎 Search Results")
                    st.json(metadata["search_results"])


# =============================================================================
# Fallback: Service Function Call (Non-Streaming)
# =============================================================================
def call_agent_service_sync(query: str, member_id: str | None, execution_id: str | None) -> dict:
    """Call the SPCS agent service via service function (fallback)."""
    try:
        if not execution_id:
            execution_id = f"streamlit_{int(time.time())}"

        safe_query = query.replace("'", "''")
        safe_member_id = (member_id or "").replace("'", "''")

        sql = f"""
            SELECT HEALTHCARE_DB.STAGING.HEALTHCARE_AGENT_QUERY(
                '{safe_query}',
                '{safe_member_id}',
                '{execution_id}'
            ) AS result
        """

        result = session.sql(sql).collect()

        if result and result[0][0]:
            response = result[0][0]
            if isinstance(response, str):
                response = json.loads(response)

            return {
                "output": response.get("output", "No response"),
                "routing": response.get("routing", ""),
                "executionId": execution_id,
                "analystResults": response.get("analystResults"),
                "searchResults": response.get("searchResults"),
            }

        return {"output": "No response from service", "error": True}

    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg:
            return {
                "output": "❌ Service function not found. Please check SPCS deployment.",
                "error": True,
            }
        return {"output": f"❌ Error: {error_msg}", "error": True}


# =============================================================================
# Streaming Chat Handler
# =============================================================================
def handle_streaming_response(
    query: str,
    member_id: str | None,
    execution_id: str | None,
    status_container,
    response_container,
) -> dict:
    """Handle streaming response with real-time progress updates.
    
    Shows the answer IMMEDIATELY when received, without waiting for complete event.
    """
    endpoint_url = get_service_endpoint_url()

    if not endpoint_url:
        # Fallback to sync call
        status_container.warning("⚠️ Could not connect to streaming endpoint. Using fallback...")
        return call_agent_service_sync(query, member_id, execution_id)

    progress_steps = []
    answer_displayed = False
    final_result = {
        "output": "",
        "routing": "",
        "executionId": execution_id or f"streamlit_{int(time.time())}",
        "analystResults": None,
        "searchResults": None,
    }

    try:
        for event in stream_agent_response(endpoint_url, query, member_id, execution_id):
            event_type = event.get("event_type", "")
            data = event.get("data", {})

            if event_type == "node_start":
                final_result["executionId"] = data.get("thread_id", final_result["executionId"])
                status_container.info("🚀 Starting agent workflow...")
                progress_steps.append(("start", "🚀 Agent started"))

            elif event_type == "react_thought":
                thought = data.get("thought", "")[:100]
                action = data.get("action", "")
                iteration = data.get("iteration", 0)

                if action:
                    tool_emoji = "📊" if "member" in action.lower() or "analyst" in action.lower() else "🔎"
                    status_container.info(f"🤔 Iteration {iteration}: Calling {tool_emoji} **{action}**...")
                    progress_steps.append(("tool", f"🔧 Calling tool: {action}"))
                else:
                    status_container.info(f"💭 Thinking: {thought}...")
                    progress_steps.append(("thinking", f"💭 {thought[:50]}..."))

            elif event_type == "react_action":
                result_preview = data.get("result_preview", "")[:100]
                status_container.success("📥 Tool returned results")
                progress_steps.append(("result", f"📥 Got results: {result_preview[:30]}..."))

            elif event_type == "react_answer":
                # Only display FULL answer from final_answer node, not truncated preview
                full_answer = data.get("answer", "")  # Only from final_answer node
                preview = data.get("answer_preview", "")  # Truncated from model node
                
                if full_answer and not answer_displayed:
                    # This is the FULL answer from final_answer node - display it!
                    final_result["output"] = full_answer
                    status_container.empty()
                    response_container.markdown(full_answer)
                    answer_displayed = True
                    progress_steps.append(("complete", "✅ Answer ready"))
                elif preview and not answer_displayed:
                    # This is just a preview - show status but don't display yet
                    status_container.info("✍️ Generating final answer...")
                    final_result["output"] = preview  # Store preview as fallback

            elif event_type == "complete":
                status_container.empty()  # Clear status (if not already cleared)
                # Extract routing and results from complete event
                if data.get("routing"):
                    final_result["routing"] = data["routing"]
                if data.get("analyst_results"):
                    final_result["analystResults"] = data["analyst_results"]
                if data.get("search_results"):
                    final_result["searchResults"] = data["search_results"]

            elif event_type == "error":
                error_msg = data.get("message", "Unknown error")
                status_container.error(f"❌ Error: {error_msg}")
                final_result["output"] = f"Error: {error_msg}"
                final_result["error"] = True
                break

        # Fallback routing from progress steps if not provided by backend
        if not final_result.get("routing"):
            tools_called = [s[1] for s in progress_steps if s[0] == "tool"]
            if any("analyst" in t.lower() or "member" in t.lower() for t in tools_called):
                if any("search" in t.lower() or "knowledge" in t.lower() for t in tools_called):
                    final_result["routing"] = "both"
                else:
                    final_result["routing"] = "analyst"
            elif any("search" in t.lower() or "knowledge" in t.lower() for t in tools_called):
                final_result["routing"] = "search"
            else:
                final_result["routing"] = "react"

    except Exception as e:
        status_container.error(f"❌ Streaming error: {e}")
        # Fallback to sync
        return call_agent_service_sync(query, member_id, execution_id)

    # Mark that answer was already displayed so caller doesn't duplicate
    final_result["_answer_displayed"] = answer_displayed
    return final_result


# =============================================================================
# Chat Input Handler
# =============================================================================
if prompt := st.chat_input("Ask about your healthcare coverage..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent service
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        if st.session_state.use_streaming:
            # Use streaming endpoint
            result = handle_streaming_response(
                prompt,
                st.session_state.member_id,
                st.session_state.execution_id,
                status_placeholder,
                response_placeholder,
            )
        else:
            # Use synchronous service function
            status_placeholder.info("🤔 Thinking...")
            result = call_agent_service_sync(prompt, st.session_state.member_id, st.session_state.execution_id)
            status_placeholder.empty()

        # Store execution_id for conversation continuity
        if result.get("executionId"):
            st.session_state.execution_id = result.get("executionId")

        response_text = result.get("output", "No response received.")

        # Handle errors or display answer (if not already displayed by streaming)
        if result.get("error"):
            st.error(response_text)
        elif not result.get("_answer_displayed"):
            # Only display if streaming didn't already show it
            st.markdown(response_text)

        # Store message with metadata
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "metadata": {
                    "routing": result.get("routing"),
                    "execution_id": result.get("executionId"),
                    "analyst_results": result.get("analystResults"),
                    "search_results": result.get("searchResults"),
                }
                if not result.get("error")
                else None,
            }
        )

        # Show debug info
        if st.session_state.debug_mode and not result.get("error"):
            with st.expander("🔍 Full Response"):
                st.json(result)

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption("Powered by Snowflake Cortex AI • LangGraph ReAct Agent • Built with ❄️ Streamlit in Snowflake")
