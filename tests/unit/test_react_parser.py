"""Unit tests for ReAct parser with structured JSON output from Cortex COMPLETE."""

import json

from src.graphs.react_parser import (
    VALID_ACTIONS,
    format_observation,
    parse_react_response,
)


class TestParseReactResponse:
    """Tests for structured JSON parsing from Cortex COMPLETE response_format."""

    def test_valid_query_member_data(self) -> None:
        """Test parsing valid JSON for query_member_data action."""
        response = json.dumps(
            {
                "thought": "I need to look up member information",
                "action": "query_member_data",
                "action_input": {
                    "query": "What are the member's claims?",
                    "member_id": "123456789",
                },
            }
        )

        result = parse_react_response(response)

        assert result["thought"] == "I need to look up member information"
        assert result["action"] == "query_member_data"
        assert result["action_input"]["query"] == "What are the member's claims?"
        assert result["action_input"]["member_id"] == "123456789"

    def test_valid_search_knowledge(self) -> None:
        """Test parsing valid JSON for search_knowledge action."""
        response = json.dumps(
            {
                "thought": "I should search for policy information",
                "action": "search_knowledge",
                "action_input": {
                    "query": "What is the deductible policy?",
                },
            }
        )

        result = parse_react_response(response)

        assert result["action"] == "search_knowledge"
        assert result["action_input"]["query"] == "What is the deductible policy?"

    def test_valid_final_answer(self) -> None:
        """Test parsing valid JSON for FINAL_ANSWER action."""
        response = json.dumps(
            {
                "thought": "I have all the information needed",
                "action": "FINAL_ANSWER",
                "action_input": {
                    "answer": "Your premium is $500 per month.",
                },
            }
        )

        result = parse_react_response(response)

        assert result["action"] == "FINAL_ANSWER"
        assert result["action_input"]["answer"] == "Your premium is $500 per month."

    def test_invalid_action_falls_back_to_final_answer(self) -> None:
        """Test that invalid action falls back to FINAL_ANSWER."""
        response = json.dumps(
            {
                "thought": "Some reasoning",
                "action": "invalid_tool",
                "action_input": {"query": "test"},
            }
        )

        result = parse_react_response(response)

        assert result["action"] == "FINAL_ANSWER"
        # Should use thought as fallback answer
        assert result["action_input"]["answer"] == "Some reasoning"

    def test_malformed_json_returns_defaults(self) -> None:
        """Test that malformed JSON returns default values."""
        malformed_response = "{ invalid json here"

        result = parse_react_response(malformed_response)

        assert result["action"] == "FINAL_ANSWER"
        assert "issue" in result["action_input"]["answer"].lower()

    def test_empty_response_returns_defaults(self) -> None:
        """Test handling of empty response."""
        result = parse_react_response("")

        assert result["action"] == "FINAL_ANSWER"
        assert "issue" in result["action_input"]["answer"].lower()

    def test_whitespace_only_returns_defaults(self) -> None:
        """Test handling of whitespace-only response."""
        result = parse_react_response("   \n\t  ")

        assert result["action"] == "FINAL_ANSWER"
        assert "issue" in result["action_input"]["answer"].lower()

    def test_null_member_id_preserved(self) -> None:
        """Test that null member_id is handled correctly."""
        response = json.dumps(
            {
                "thought": "Aggregate query, no member ID needed",
                "action": "query_member_data",
                "action_input": {
                    "query": "How many members do we have?",
                    "member_id": None,
                },
            }
        )

        result = parse_react_response(response)

        assert result["action_input"]["member_id"] is None

    def test_missing_thought_field(self) -> None:
        """Test handling missing thought field."""
        response = json.dumps(
            {
                "action": "search_knowledge",
                "action_input": {"query": "test query"},
            }
        )

        result = parse_react_response(response)

        assert result["thought"] == ""
        assert result["action"] == "search_knowledge"

    def test_missing_action_defaults_to_final_answer(self) -> None:
        """Test handling missing action field."""
        response = json.dumps(
            {
                "thought": "Some thought",
                "action_input": {"answer": "Some answer"},
            }
        )

        result = parse_react_response(response)

        assert result["action"] == "FINAL_ANSWER"

    def test_missing_action_input_defaults_to_empty(self) -> None:
        """Test handling missing action_input field."""
        response = json.dumps(
            {
                "thought": "Some thought",
                "action": "search_knowledge",
            }
        )

        result = parse_react_response(response)

        assert result["action_input"] == {}

    def test_all_valid_actions_accepted(self) -> None:
        """Test that all valid actions are accepted."""
        for action in VALID_ACTIONS:
            action_input = {"answer": "test"} if action == "FINAL_ANSWER" else {"query": "test"}
            response = json.dumps(
                {
                    "thought": f"Testing {action}",
                    "action": action,
                    "action_input": action_input,
                }
            )

            result = parse_react_response(response)

            assert result["action"] == action


class TestFormatObservation:
    """Tests for observation formatting."""

    def test_format_analyst_observation_with_member_info(self) -> None:
        """Test formatting analyst result with member information."""
        result = {
            "member_id": "123456789",
            "member_info": {
                "NAME": "John Doe",
                "DOB": "1990-01-15",
                "PHONE": "555-1234",
            },
            "coverage": {
                "PLAN_NAME": "Gold Plan",
                "PREMIUM": 500,
            },
            "claims": [],
        }

        observation = format_observation("query_member_data", result)

        assert "John Doe" in observation
        assert "Gold Plan" in observation
        assert "500" in observation

    def test_format_analyst_observation_aggregate(self) -> None:
        """Test formatting aggregate query result."""
        result = {
            "aggregate_result": {
                "query_type": "member_count",
                "total_members": 242,
                "total_plans": 5,
            },
        }

        observation = format_observation("query_member_data", result)

        assert "242" in observation
        assert "members" in observation.lower()

    def test_format_search_observation(self) -> None:
        """Test formatting search results."""
        results = [
            {"text": "FAQ about deductibles", "score": 0.9, "source": "faqs"},
            {"text": "Policy document", "score": 0.85, "source": "policies"},
        ]

        observation = format_observation("search_knowledge", results)

        assert "FAQ about deductibles" in observation
        assert "faqs" in observation

    def test_format_search_observation_prioritizes_transcripts(self) -> None:
        """Test that search results prioritize transcripts over FAQs."""
        results = [
            {"text": "FAQ content", "score": 0.95, "source": "faqs"},
            {"text": "Call transcript content", "score": 0.80, "source": "transcripts"},
        ]

        observation = format_observation("search_knowledge", results)

        # Transcript should appear first despite lower score
        transcript_pos = observation.find("Call transcript")
        faq_pos = observation.find("FAQ content")
        assert transcript_pos < faq_pos

    def test_format_search_observation_empty(self) -> None:
        """Test formatting empty search results."""
        observation = format_observation("search_knowledge", [])

        assert "No relevant documents found" in observation

    def test_format_observation_none_result(self) -> None:
        """Test formatting None result."""
        observation = format_observation("query_member_data", None)

        assert "No results found" in observation

    def test_format_observation_string_result(self) -> None:
        """Test formatting string result."""
        observation = format_observation("some_tool", "Simple string result")

        assert "Simple string result" in observation

    def test_format_observation_unknown_tool(self) -> None:
        """Test formatting result from unknown tool."""
        observation = format_observation("unknown_tool", {"data": "value"})

        assert "data" in observation


class TestValidActions:
    """Tests for VALID_ACTIONS constant."""

    def test_contains_expected_actions(self) -> None:
        """Test that VALID_ACTIONS contains all expected actions."""
        assert "query_member_data" in VALID_ACTIONS
        assert "search_knowledge" in VALID_ACTIONS
        assert "FINAL_ANSWER" in VALID_ACTIONS

    def test_no_extra_actions(self) -> None:
        """Test that VALID_ACTIONS only contains expected actions."""
        assert len(VALID_ACTIONS) == 3
        assert VALID_ACTIONS.issubset({"query_member_data", "search_knowledge", "FINAL_ANSWER"})
        assert {"query_member_data", "search_knowledge", "FINAL_ANSWER"}.issubset(VALID_ACTIONS)
