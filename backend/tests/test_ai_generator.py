"""
Unit tests for AIGenerator functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test_api_key", "test_model")
        
        assert generator.model == "test_model"
        assert generator.base_params["model"] == "test_model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tools"""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "This is a test response"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        result = generator.generate_response("What is Python?")
        
        assert result == "This is a test response"
        mock_client.messages.create.assert_called_once()
        
        # Check that tools were not included in the call
        call_args = mock_client.messages.create.call_args
        assert "tools" not in call_args[1]
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Response with history"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        result = generator.generate_response(
            "Follow up question", 
            conversation_history="Previous conversation..."
        )
        
        assert result == "Response with history"
        
        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation..." in system_content
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        # Setup mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Direct response without tools"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Mock tools
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = MagicMock()
        
        generator = AIGenerator("test_api_key", "test_model")
        result = generator.generate_response(
            "General question", 
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct response without tools"
        
        # Check that tools were included in the call
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == mock_tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation with tool use"""
        # Setup mock client
        mock_client = MagicMock()
        
        # First response with tool use
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "Python basics"}
        mock_tool_response.content = [mock_tool_block]
        
        # Final response after tool execution
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Based on the search results: Python is a programming language."
        
        # Configure mock to return different responses on subsequent calls
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tools and tool manager
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Python is a high-level programming language."
        
        generator = AIGenerator("test_api_key", "test_model")
        result = generator.generate_response(
            "What is Python?",
            tools=mock_tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on the search results: Python is a programming language."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_single_tool(self, mock_anthropic):
        """Test _handle_tool_execution with single tool call"""
        # Setup mock client for final response
        mock_client = MagicMock()
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final response after tool use"
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        
        # Create mock initial response with tool use
        mock_initial_response = MagicMock()
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Test question"}],
            "system": "Test system prompt"
        }
        
        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        result = generator._handle_tool_execution(
            mock_initial_response, 
            base_params, 
            mock_tool_manager
        )
        
        assert result == "Final response after tool use"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="test query"
        )
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic):
        """Test _handle_tool_execution with multiple tool calls"""
        # Setup mock client for final response
        mock_client = MagicMock()
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Final response after multiple tools"
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        
        # Create mock initial response with multiple tool uses
        mock_initial_response = MagicMock()
        mock_tool_block1 = MagicMock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "first query"}
        
        mock_tool_block2 = MagicMock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"course_name": "Python Course"}
        
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]
        
        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Test question"}],
            "system": "Test system prompt"
        }
        
        # Mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result"
        ]
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "Final response after multiple tools"
        assert mock_tool_manager.execute_tool.call_count == 2
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_error(self, mock_anthropic):
        """Test _handle_tool_execution with tool execution error"""
        # Setup mock client for final response
        mock_client = MagicMock()
        mock_final_response = MagicMock()
        mock_final_response.content = [MagicMock()]
        mock_final_response.content[0].text = "Error response"
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        
        # Create mock initial response with tool use
        mock_initial_response = MagicMock()
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        mock_initial_response.content = [mock_tool_block]
        
        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "Test question"}],
            "system": "Test system prompt"
        }
        
        # Mock tool manager that returns error
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Search error: Database connection failed"
        
        result = generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )
        
        assert result == "Error response"
        # Tool should still be executed even if it returns an error
        mock_tool_manager.execute_tool.assert_called_once()
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator("test_api_key", "test_model")
        
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "tool" in generator.SYSTEM_PROMPT.lower()
        assert "course" in generator.SYSTEM_PROMPT.lower()
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test handling of Anthropic API errors"""
        # Setup mock client that raises an exception
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error: Rate limit exceeded")
        mock_anthropic.return_value = mock_client
        
        generator = AIGenerator("test_api_key", "test_model")
        
        # Should raise the exception (not catch it)
        with pytest.raises(Exception) as excinfo:
            generator.generate_response("Test question")
        
        assert "API Error: Rate limit exceeded" in str(excinfo.value)