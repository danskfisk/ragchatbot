"""
Unit tests for CourseSearchTool functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""
    
    def test_get_tool_definition(self, populated_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseSearchTool(populated_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
    
    def test_execute_basic_query(self, populated_vector_store):
        """Test basic query execution without filters"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("Python programming")
        
        # Should not return an error
        assert not result.startswith("Search error:")
        assert not result.startswith("No relevant content found")
        # Should contain course context
        assert "Introduction to Python" in result or "Advanced Machine Learning" in result
    
    def test_execute_with_course_filter(self, populated_vector_store):
        """Test query execution with course name filter"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("programming", course_name="Introduction to Python")
        
        # Should not return an error
        assert not result.startswith("Search error:")
        # Should contain Python course content
        assert "Introduction to Python" in result
        # Should not contain ML course content
        assert "Advanced Machine Learning" not in result
    
    def test_execute_with_lesson_filter(self, populated_vector_store):
        """Test query execution with lesson number filter"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("programming", lesson_number=1)
        
        # Should not return an error
        assert not result.startswith("Search error:")
        # Should contain lesson 1 content
        assert "Lesson 1" in result
        # Should not contain lesson 2 content  
        assert "Lesson 2" not in result
    
    def test_execute_with_course_and_lesson_filter(self, populated_vector_store):
        """Test query execution with both course and lesson filters"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("programming", course_name="Introduction to Python", lesson_number=1)
        
        # Should not return an error
        assert not result.startswith("Search error:")
        # Should contain specific course and lesson
        assert "Introduction to Python" in result
        assert "Lesson 1" in result
    
    def test_execute_nonexistent_course(self, populated_vector_store):
        """Test query with non-existent course name"""
        tool = CourseSearchTool(populated_vector_store)
        result = tool.execute("test query", course_name="Nonexistent Course")
        
        # Should return appropriate error message
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_execute_no_results(self, populated_vector_store):
        """Test query that returns no results"""
        tool = CourseSearchTool(populated_vector_store)
        # Query for something that shouldn't exist in our test data
        result = tool.execute("quantum computing blockchain cryptocurrency")
        
        # Should handle empty results gracefully
        assert "No relevant content found" in result
    
    def test_execute_vector_store_error(self):
        """Test handling of vector store errors"""
        # Create a mock vector store that raises an exception
        mock_vector_store = MagicMock()
        mock_vector_store.search.side_effect = Exception("Database connection failed")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should handle error gracefully
        assert "Database connection failed" in result
    
    def test_format_results_single_result(self, populated_vector_store):
        """Test result formatting with single search result"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Create mock search results
        results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Introduction to Python", "lesson_number": 1}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        assert "[Introduction to Python - Lesson 1]" in formatted
        assert "Python is a programming language" in formatted
        assert len(tool.last_sources) == 1
        assert "Introduction to Python - Lesson 1" in tool.last_sources[0]
    
    def test_format_results_multiple_results(self, populated_vector_store):
        """Test result formatting with multiple search results"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Create mock search results
        results = SearchResults(
            documents=[
                "Python is a programming language", 
                "Control structures help program flow"
            ],
            metadata=[
                {"course_title": "Introduction to Python", "lesson_number": 1},
                {"course_title": "Introduction to Python", "lesson_number": 2}
            ],
            distances=[0.1, 0.3]
        )
        
        formatted = tool._format_results(results)
        
        assert "[Introduction to Python - Lesson 1]" in formatted
        assert "[Introduction to Python - Lesson 2]" in formatted
        assert "Python is a programming language" in formatted
        assert "Control structures help program flow" in formatted
        assert len(tool.last_sources) == 2
    
    def test_format_results_no_lesson_number(self, populated_vector_store):
        """Test result formatting when lesson number is missing"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Create mock search results without lesson number
        results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Introduction to Python"}],
            distances=[0.1]
        )
        
        formatted = tool._format_results(results)
        
        assert "[Introduction to Python]" in formatted
        assert "Python is a programming language" in formatted
        assert len(tool.last_sources) == 1
        assert "Introduction to Python" == tool.last_sources[0]
    
    def test_source_tracking(self, populated_vector_store):
        """Test that sources are properly tracked"""
        tool = CourseSearchTool(populated_vector_store)
        
        # Execute a search that should return results
        result = tool.execute("Python programming")
        
        # Sources should be populated
        assert len(tool.last_sources) > 0
        # Sources should contain course information
        for source in tool.last_sources:
            assert any(course in source for course in ["Introduction to Python", "Advanced Machine Learning"])


class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_register_tool(self, populated_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(populated_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_get_tool_definitions(self, populated_vector_store):
        """Test getting tool definitions"""
        manager = ToolManager()
        tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool(self, populated_vector_store):
        """Test tool execution through manager"""
        manager = ToolManager()
        tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="Python programming")
        
        assert not result.startswith("Tool 'search_course_content' not found")
        assert "Introduction to Python" in result or "Advanced Machine Learning" in result
    
    def test_execute_nonexistent_tool(self, populated_vector_store):
        """Test executing non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, populated_vector_store):
        """Test getting sources from last search"""
        manager = ToolManager()
        tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(tool)
        
        # Execute a search
        manager.execute_tool("search_course_content", query="Python programming")
        
        # Get sources
        sources = manager.get_last_sources()
        
        assert len(sources) > 0
    
    def test_reset_sources(self, populated_vector_store):
        """Test resetting sources"""
        manager = ToolManager()
        tool = CourseSearchTool(populated_vector_store)
        manager.register_tool(tool)
        
        # Execute a search to populate sources
        manager.execute_tool("search_course_content", query="Python programming")
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0