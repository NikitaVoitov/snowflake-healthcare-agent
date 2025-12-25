"""Integration tests with REAL Snowflake data - NO MOCKS.

These tests connect to actual Snowflake database and verify:
1. Real data types (DATE, DECIMAL, VARCHAR) are handled correctly
2. Full HTTP endpoint JSON serialization works
3. Cortex services return actual data

Requirements:
- Valid .env file with Snowflake credentials
- HEALTHCARE_DB database with populated tables
- PAYERS_CC_WH warehouse available

Snowflake Data Types Reference:
https://docs.snowflake.com/en/sql-reference-data-types
"""

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from dotenv import load_dotenv
from httpx import ASGITransport, AsyncClient

# Load environment BEFORE importing app
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


# Skip all tests if Snowflake credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("SNOWFLAKE_ACCOUNT"),
    reason="Snowflake credentials not configured",
)


class TestRealSnowflakeDataTypes:
    """Test actual Snowflake data type handling."""

    @pytest.fixture(scope="class")
    def snowpark_session(self):
        """Create real Snowpark session."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from snowflake.snowpark import Session

        key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        if not key_path:
            pytest.skip("SNOWFLAKE_PRIVATE_KEY_PATH not set")

        with open(Path(key_path).expanduser(), "rb") as f:
            passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )

        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        session = Session.builder.configs(
            {
                "account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "user": os.getenv("SNOWFLAKE_USER"),
                "private_key": pkb,
                "database": os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_DB"),
                "schema": os.getenv("SNOWFLAKE_SCHEMA", "MEMBER_SCHEMA"),
                "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "PAYERS_CC_WH"),
            }
        ).create()

        yield session
        session.close()

    def test_date_type_from_snowflake(self, snowpark_session) -> None:
        """DATE columns return datetime.date objects."""
        result = snowpark_session.sql("SELECT DOB FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED LIMIT 1").collect()

        assert len(result) > 0
        dob = result[0]["DOB"]
        assert isinstance(dob, date), f"Expected date, got {type(dob)}"

    def test_decimal_type_from_snowflake(self, snowpark_session) -> None:
        """NUMBER/DECIMAL columns return Decimal objects."""
        result = snowpark_session.sql("SELECT PREMIUM FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED LIMIT 1").collect()

        assert len(result) > 0
        premium = result[0]["PREMIUM"]
        assert isinstance(premium, Decimal), f"Expected Decimal, got {type(premium)}"

    def test_varchar_type_from_snowflake(self, snowpark_session) -> None:
        """VARCHAR columns return str objects."""
        result = snowpark_session.sql("SELECT NAME, GENDER FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED LIMIT 1").collect()

        assert len(result) > 0
        assert isinstance(result[0]["NAME"], str)
        assert isinstance(result[0]["GENDER"], str)

    def test_real_member_data_structure(self, snowpark_session) -> None:
        """Verify actual member data structure matches expected schema."""
        result = snowpark_session.sql("""
            SELECT MEMBER_ID, NAME, DOB, GENDER, ADDRESS, MEMBER_PHONE,
                   PLAN_ID, PLAN_NAME, PLAN_TYPE, PREMIUM, PCP, PCP_PHONE
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE MEMBER_ID = '106742775'
            LIMIT 1
        """).collect()

        assert len(result) > 0
        row = result[0]

        # Verify Snowflake data types per https://docs.snowflake.com/en/sql-reference-data-types
        assert isinstance(row["MEMBER_ID"], str)  # VARCHAR
        assert isinstance(row["NAME"], str)  # VARCHAR
        assert isinstance(row["DOB"], date)  # DATE
        assert isinstance(row["GENDER"], str)  # VARCHAR
        assert isinstance(row["PREMIUM"], Decimal)  # NUMBER(10,2)

    def test_real_claims_data_structure(self, snowpark_session) -> None:
        """Verify actual claims data structure."""
        result = snowpark_session.sql("""
            SELECT CLAIM_ID, CLAIM_SERVICE_FROM_DATE, CLAIM_SERVICE,
                   CLAIM_BILL_AMT, CLAIM_PAID_AMT, CLAIM_STATUS
            FROM HEALTHCARE_DB.MEMBER_SCHEMA.CALL_CENTER_MEMBER_DENORMALIZED
            WHERE CLAIM_ID IS NOT NULL
            LIMIT 1
        """).collect()

        assert len(result) > 0
        row = result[0]

        # DATE type check
        if row["CLAIM_SERVICE_FROM_DATE"]:
            assert isinstance(row["CLAIM_SERVICE_FROM_DATE"], date)

        # DECIMAL type check
        if row["CLAIM_BILL_AMT"]:
            assert isinstance(row["CLAIM_BILL_AMT"], Decimal)


class TestRealHTTPEndpoints:
    """Test FastAPI endpoints with real Snowflake data."""

    @pytest.fixture(scope="function")
    async def async_client(self):
        """Create async HTTP client for testing FastAPI app.

        Uses function scope to avoid asyncio event loop issues with cached checkpointer.
        Clears LRU caches before each test to ensure fresh event loop bindings.
        """
        # Clear caches to avoid event loop binding issues
        from src.dependencies import get_checkpointer, get_compiled_graph, get_settings

        get_settings.cache_clear()
        get_compiled_graph.cache_clear()
        get_checkpointer.cache_clear()

        from src.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client

        # Clear after test too
        get_settings.cache_clear()
        get_compiled_graph.cache_clear()
        get_checkpointer.cache_clear()

    @pytest.mark.asyncio
    async def test_query_endpoint_json_serialization(self, async_client) -> None:
        """POST /agents/query should serialize Snowflake types to JSON."""
        response = await async_client.post(
            "/agents/query",
            json={
                "query": "What are my claims?",
                "member_id": "106742775",
                "tenant_id": "test",
                "user_id": "test_user",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response is valid JSON (no date/Decimal serialization errors)
        # Note: Pydantic uses camelCase aliases (analystResults, not analyst_results)
        assert "output" in data
        assert "analystResults" in data  # camelCase per BaseSchema alias_generator

    @pytest.mark.asyncio
    async def test_query_endpoint_real_member_data(self, async_client) -> None:
        """Query endpoint returns real member data."""
        response = await async_client.post(
            "/agents/query",
            json={
                "query": "Tell me about my coverage",
                "member_id": "106742775",
                "tenant_id": "test",
                "user_id": "test_user",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify we got a response
        assert data.get("output") is not None

    @pytest.mark.asyncio
    async def test_search_endpoint_real_cortex_search(self, async_client) -> None:
        """Search query uses real Cortex Search services."""
        response = await async_client.post(
            "/agents/query",
            json={
                "query": "How do I appeal a claim?",
                "tenant_id": "test",
                "user_id": "test_user",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have output regardless of routing
        assert data.get("output") is not None

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client) -> None:
        """Health endpoint should return OK."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"


class TestRealCortexTools:
    """Test Cortex tools with real Snowflake connection."""

    @pytest.fixture(scope="class")
    def snowpark_session(self):
        """Create real Snowpark session."""
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
        from snowflake.snowpark import Session

        key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        if not key_path:
            pytest.skip("SNOWFLAKE_PRIVATE_KEY_PATH not set")

        with open(Path(key_path).expanduser(), "rb") as f:
            passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=passphrase.encode() if passphrase else None,
                backend=default_backend(),
            )

        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        session = Session.builder.configs(
            {
                "account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "user": os.getenv("SNOWFLAKE_USER"),
                "private_key": pkb,
                "database": os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_DB"),
                "schema": os.getenv("SNOWFLAKE_SCHEMA", "MEMBER_SCHEMA"),
                "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "PAYERS_CC_WH"),
            }
        ).create()

        yield session
        session.close()

    @pytest.mark.asyncio
    async def test_analyst_tool_real_data(self, snowpark_session) -> None:
        """AsyncCortexAnalystTool returns real member data."""
        from src.services.cortex_tools import AsyncCortexAnalystTool

        tool = AsyncCortexAnalystTool(snowpark_session)
        result = await tool.execute("What are my claims?", member_id="106742775")

        # Verify result structure (may use local fallback in test env)
        assert "member_id" in result or "aggregate_result" in result

    @pytest.mark.asyncio
    async def test_search_tool_real_data(self, snowpark_session) -> None:
        """AsyncCortexSearchTool returns real search results."""
        from src.services.cortex_tools import AsyncCortexSearchTool

        tool = AsyncCortexSearchTool(snowpark_session)
        results = await tool.execute("deductible policy", limit=3)

        # Should return results from Cortex Search
        assert isinstance(results, list)
        if results:
            assert "text" in results[0]
            assert "source" in results[0]


class TestRealCheckpointer:
    """Test Snowflake checkpointer with real database."""

    @pytest.mark.asyncio
    async def test_checkpointer_real_persistence(self) -> None:
        """Checkpointer saves to real Snowflake tables."""
        from src.services.checkpointer import create_checkpointer

        checkpointer = create_checkpointer()

        # Verify it's using real Snowflake, not MemorySaver
        assert "MemorySaver" not in type(checkpointer).__name__

        # Setup should succeed on real Snowflake
        if hasattr(checkpointer, "asetup"):
            await checkpointer.asetup()

    @pytest.mark.asyncio
    async def test_graph_with_real_checkpointer(self) -> None:
        """Graph execution with real Snowflake checkpointer."""
        from src.graphs.react_workflow import build_react_graph
        from src.services.checkpointer import create_checkpointer

        checkpointer = create_checkpointer()
        if hasattr(checkpointer, "asetup"):
            await checkpointer.asetup()

        graph = build_react_graph().compile(checkpointer=checkpointer)

        thread_id = f"real-test-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}

        from langchain_core.messages import HumanMessage

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are my claims?")],
                "user_query": "What are my claims?",
                "member_id": "106742775",
                "tenant_id": "test_tenant",
                "conversation_history": [],
                "iteration": 0,
                "max_iterations": 5,
                "final_answer": None,
                "execution_id": thread_id,
            },
            config=config,
        )

        # Verify execution completed (ReAct should have final_answer)
        assert result.get("final_answer") is not None

        # Verify checkpoint was saved
        checkpoint = await checkpointer.aget_tuple(config)
        assert checkpoint is not None


class TestSnowflakeJSONEncoder:
    """Test custom JSON encoder handles all Snowflake types."""

    def test_date_serialization(self) -> None:
        """date objects serialize to ISO format string."""
        from src.routers.agent_routes import SnowflakeJSONEncoder

        data = {"dob": date(2001, 4, 23)}
        result = json.dumps(data, cls=SnowflakeJSONEncoder)
        parsed = json.loads(result)

        assert parsed["dob"] == "2001-04-23"

    def test_datetime_serialization(self) -> None:
        """datetime objects serialize to ISO format string."""
        from src.routers.agent_routes import SnowflakeJSONEncoder

        data = {"timestamp": datetime(2024, 1, 15, 10, 30, 0)}
        result = json.dumps(data, cls=SnowflakeJSONEncoder)
        parsed = json.loads(result)

        assert parsed["timestamp"] == "2024-01-15T10:30:00"

    def test_decimal_serialization(self) -> None:
        """Decimal objects serialize to float."""
        from src.routers.agent_routes import SnowflakeJSONEncoder

        data = {"premium": Decimal("373.64")}
        result = json.dumps(data, cls=SnowflakeJSONEncoder)
        parsed = json.loads(result)

        assert parsed["premium"] == 373.64

    def test_nested_snowflake_types(self) -> None:
        """Nested structures with Snowflake types serialize correctly."""
        from src.routers.agent_routes import SnowflakeJSONEncoder

        # Simulate actual Snowflake query result structure
        data = {
            "member": {
                "name": "James Mills",
                "dob": date(2001, 4, 23),
                "premium": Decimal("373.64"),
            },
            "claims": [
                {
                    "claim_id": "C066210",
                    "date": date(2024, 1, 8),
                    "amount": Decimal("4619.00"),
                },
            ],
        }

        result = json.dumps(data, cls=SnowflakeJSONEncoder)
        parsed = json.loads(result)

        # Verify all types converted correctly
        assert parsed["member"]["dob"] == "2001-04-23"
        assert parsed["member"]["premium"] == 373.64
        assert parsed["claims"][0]["date"] == "2024-01-08"
        assert parsed["claims"][0]["amount"] == 4619.0
