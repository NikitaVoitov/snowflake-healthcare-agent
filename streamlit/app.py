"""Healthcare Contact Center Assistant - Streamlit App.

Native Snowflake Streamlit app that uses Service Function to call
the Healthcare ReAct Agent running in SPCS.
"""

import json

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


# =============================================================================
# Service Status Check
# =============================================================================
def check_service_status() -> dict:
    """Check if the SPCS service is running."""
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

    # Debug Settings
    st.header("⚙️ Settings")
    st.session_state.debug_mode = st.checkbox(
        "Show routing details",
        value=st.session_state.debug_mode,
        help="Display agent routing and execution details",
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
            if "query_member_data" in tools_used.lower():
                st.markdown(
                    '<span class="tool-badge badge-analyst">📊 ANALYST</span>',
                    unsafe_allow_html=True,
                )
            if "search_knowledge" in tools_used.lower():
                st.markdown(
                    '<span class="tool-badge badge-search">🔎 SEARCH</span>',
                    unsafe_allow_html=True,
                )
            if tools_used == "none" or not tools_used:
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
# Chat Input Handler
# =============================================================================
def call_agent_service(query: str, member_id: str | None, execution_id: str | None) -> dict:
    """Call the SPCS agent service via service function."""
    try:
        # Build execution_id for conversation continuity
        if not execution_id:
            import time

            execution_id = f"streamlit_{int(time.time())}"

        # Escape single quotes in query
        safe_query = query.replace("'", "''")
        safe_member_id = (member_id or "").replace("'", "''")

        # Call the service function
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
            # Handle both dict and JSON string
            if isinstance(response, str):
                response = json.loads(response)

            return {
                "output": response.get("output", "No response"),
                "routing": response.get("routing", ""),
                "executionId": execution_id,
                "errorCount": response.get("errorCount", 0),
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


# Chat input
if prompt := st.chat_input("Ask about your healthcare coverage..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent service
    with st.chat_message("assistant"), st.spinner("🤔 Thinking..."):
        result = call_agent_service(prompt, st.session_state.member_id, st.session_state.execution_id)

        # Store execution_id for conversation continuity
        if result.get("executionId"):
            st.session_state.execution_id = result.get("executionId")

        response_text = result.get("output", "No response received.")

        # Handle errors
        if result.get("error"):
            st.error(response_text)
        else:
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
