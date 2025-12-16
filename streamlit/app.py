"""Healthcare Contact Center Assistant - Streamlit App.

Native Snowflake Streamlit app that communicates with the Healthcare
Multi-Agent SPCS service via internal DNS.
"""

import httpx

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

# SPCS Service URL (internal DNS from service deployment)
# Format: <service-name>.<schema>.<database>.snowflakecomputing.internal
AGENT_SERVICE_URL = "http://healthcare-agents-service.juiu.svc.spcs.internal:8000"


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
    .routing-badge {
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

    # Member Information
    st.header("👤 Member Information")
    member_id = st.text_input(
        "Member ID",
        value=st.session_state.member_id or "",
        placeholder="e.g., ABC1001",
        help="Enter your member ID to get personalized information",
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
        st.rerun()

    st.markdown("---")

    # Help section
    with st.expander("❓ How to use"):
        st.markdown(
            """
            **Example queries:**
            - "What is my deductible?"
            - "How do I file an appeal?"
            - "What are my recent claims?"
            - "Explain my coverage benefits"
            - "What is the copay for ER visits?"

            **Tips:**
            - Enter your Member ID for personalized info
            - Enable debug mode to see routing details
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
            routing = metadata.get("routing", "unknown")

            # Routing badge
            badge_class = f"badge-{routing}" if routing in ["analyst", "search", "both"] else ""
            st.markdown(
                f'<span class="routing-badge {badge_class}">{routing.upper()}</span>',
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Execution Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Routing", routing)
                    st.metric("Errors", metadata.get("error_count", 0))
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
def call_agent_service(query: str, member_id: str | None) -> dict:
    """Call the SPCS agent service synchronously."""
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{AGENT_SERVICE_URL}/agents/query",
                json={
                    "query": query,
                    "memberId": member_id,
                    "tenantId": "default",
                    "userId": "streamlit_user",
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"output": "⏱️ Request timed out. Please try again.", "error": True}
    except httpx.HTTPStatusError as e:
        return {"output": f"❌ Service error: {e.response.status_code}", "error": True}
    except httpx.ConnectError:
        return {
            "output": (
                "🔌 Unable to connect to agent service. Please check if the service is running."
            ),
            "error": True,
        }
    except Exception as e:
        return {"output": f"❌ Unexpected error: {e!s}", "error": True}


# Chat input
if prompt := st.chat_input("Ask about your healthcare coverage..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent service
    with st.chat_message("assistant"), st.spinner("🤔 Thinking..."):
        result = call_agent_service(prompt, st.session_state.member_id)

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
                    "error_count": result.get("errorCount", 0),
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
st.caption(
    "Powered by Snowflake Cortex AI • LangGraph Multi-Agent System • "
    "Built with ❄️ Streamlit in Snowflake"
)
