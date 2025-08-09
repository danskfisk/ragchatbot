"""
Integration tests for RAG System functionality.
"""
import pytest
import tempfile
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from search_tools import CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAG System integration functionality"""
    
    def test_initialization(self, test_config):
        """Test RAG system initialization"""
        rag_system = RAGSystem(test_config)
        
        # Check all components are initialized
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None
        
        # Check tools are registered
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_success(self, mock_doc_processor, test_config):
        """Test successfully adding a course document"""
        # Mock document processor
        mock_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            lessons=[Lesson(lesson_number=1, title="Test Lesson")]
        )
        mock_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            )
        ]
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_course_document.return_value = (mock_course, mock_chunks)
        mock_doc_processor.return_value = mock_processor_instance
        
        rag_system = RAGSystem(test_config)
        
        # Test adding document
        course, chunk_count = rag_system.add_course_document("test_file.pdf")
        
        assert course.title == "Test Course"
        assert chunk_count == 1
        
        # Verify course was added to vector store
        existing_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Test Course" in existing_titles
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('rag_system.DocumentProcessor')
    def test_add_course_document_error(self, mock_doc_processor, test_config):
        """Test error handling when adding course document"""
        # Mock document processor to raise exception
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_course_document.side_effect = Exception("Processing error")
        mock_doc_processor.return_value = mock_processor_instance
        
        rag_system = RAGSystem(test_config)
        
        # Test adding document with error
        course, chunk_count = rag_system.add_course_document("test_file.pdf")
        
        assert course is None
        assert chunk_count == 0
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('rag_system.os.listdir')
    @patch('rag_system.os.path.exists')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_success(self, mock_doc_processor, mock_exists, mock_listdir, test_config):
        """Test successfully adding courses from folder"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "not_a_course.jpg"]
        
        # Mock document processor
        mock_courses = [
            Course(title="Course 1", lessons=[]),
            Course(title="Course 2", lessons=[])
        ]
        mock_chunks_list = [
            [CourseChunk(content="Content 1", course_title="Course 1", chunk_index=0)],
            [CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)]
        ]
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_course_document.side_effect = [
            (mock_courses[0], mock_chunks_list[0]),
            (mock_courses[1], mock_chunks_list[1])
        ]
        mock_doc_processor.return_value = mock_processor_instance
        
        rag_system = RAGSystem(test_config)
        
        # Test adding folder
        total_courses, total_chunks = rag_system.add_course_folder("test_folder")
        
        assert total_courses == 2
        assert total_chunks == 2
        
        # Verify courses were added
        existing_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Course 1" in existing_titles
        assert "Course 2" in existing_titles
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('rag_system.os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, test_config):
        """Test adding courses from non-existent folder"""
        mock_exists.return_value = False
        
        rag_system = RAGSystem(test_config)
        
        # Test adding non-existent folder
        total_courses, total_chunks = rag_system.add_course_folder("nonexistent_folder")
        
        assert total_courses == 0
        assert total_chunks == 0
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('rag_system.os.listdir')
    @patch('rag_system.os.path.exists')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_skip_existing(self, mock_doc_processor, mock_exists, mock_listdir, test_config):
        """Test skipping existing courses when adding folder"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.pdf"]
        
        # Mock document processor
        mock_courses = [
            Course(title="Existing Course", lessons=[]),
            Course(title="New Course", lessons=[])
        ]
        mock_chunks_list = [
            [CourseChunk(content="Content 1", course_title="Existing Course", chunk_index=0)],
            [CourseChunk(content="Content 2", course_title="New Course", chunk_index=0)]
        ]
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_course_document.side_effect = [
            (mock_courses[0], mock_chunks_list[0]),
            (mock_courses[1], mock_chunks_list[1])
        ]
        mock_doc_processor.return_value = mock_processor_instance
        
        rag_system = RAGSystem(test_config)
        
        # Pre-populate with one course
        rag_system.vector_store.add_course_metadata(mock_courses[0])
        
        # Test adding folder (should skip existing course)
        total_courses, total_chunks = rag_system.add_course_folder("test_folder")
        
        assert total_courses == 1  # Only new course added
        assert total_chunks == 1
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_without_session(self, mock_anthropic, test_config):
        """Test querying without session ID"""
        # Setup mock Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Test query without session
        response, sources = rag_system.query("What is Python?")
        
        assert response == "Test response"
        assert isinstance(sources, list)
        
        # Verify AI was called with proper parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert "Answer this question about course materials:" in call_args[1]["messages"][0]["content"]
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_session(self, mock_anthropic, test_config):
        """Test querying with session ID"""
        # Setup mock Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Test response with history"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Create session with history
        session_id = "test_session"
        rag_system.session_manager.add_exchange(session_id, "Previous question", "Previous answer")
        
        # Test query with session
        response, sources = rag_system.query("Follow up question", session_id=session_id)
        
        assert response == "Test response with history"
        
        # Verify history was included
        call_args = mock_client.messages.create.call_args
        assert "Previous question" in call_args[1]["system"]
        assert "Previous answer" in call_args[1]["system"]
        
        # Verify new exchange was added
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert "Follow up question" in history
        assert "Test response with history" in history
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_with_tool_use(self, mock_anthropic, test_config, mock_course_data, mock_course_chunks):
        """Test querying that triggers tool use"""
        # Setup mock Anthropic client for tool use
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
        mock_final_response.content[0].text = "Python is a high-level programming language."
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Populate with test data
        for course in mock_course_data:
            rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(mock_course_chunks)
        
        # Test query that should trigger tool use
        response, sources = rag_system.query("What is Python?")
        
        assert response == "Python is a high-level programming language."
        assert len(sources) > 0  # Should have sources from search
        
        # Verify tool was executed (2 API calls made)
        assert mock_client.messages.create.call_count == 2
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    def test_get_course_analytics(self, test_config, mock_course_data):
        """Test getting course analytics"""
        rag_system = RAGSystem(test_config)
        
        # Add test courses
        for course in mock_course_data:
            rag_system.vector_store.add_course_metadata(course)
        
        # Test analytics
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 2
        assert "Introduction to Python" in analytics["course_titles"]
        assert "Advanced Machine Learning" in analytics["course_titles"]
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_query_ai_error_propagation(self, mock_anthropic, test_config):
        """Test that AI errors are propagated properly"""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(test_config)
        
        # Test that exception is raised
        with pytest.raises(Exception) as excinfo:
            rag_system.query("Test question")
        
        assert "API Error" in str(excinfo.value)
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    def test_source_management(self, test_config, mock_course_data, mock_course_chunks):
        """Test that sources are properly managed and reset"""
        rag_system = RAGSystem(test_config)
        
        # Populate with test data
        for course in mock_course_data:
            rag_system.vector_store.add_course_metadata(course)
        rag_system.vector_store.add_course_content(mock_course_chunks)
        
        # Execute search directly to populate sources
        search_result = rag_system.search_tool.execute("Python programming")
        
        # Verify sources were populated
        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0
        
        # Reset sources
        rag_system.tool_manager.reset_sources()
        
        # Verify sources were cleared
        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) == 0
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)