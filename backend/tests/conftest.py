"""
Test configuration and fixtures for RAG system tests.
"""
import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Add backend to Python path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from config import Config


@pytest.fixture
def test_config():
    """Create a test configuration with temporary paths"""
    config = Config()
    # Use temporary directory for test ChromaDB
    config.CHROMA_PATH = tempfile.mkdtemp(prefix="test_chroma_")
    config.ANTHROPIC_API_KEY = "test_key"
    config.MAX_RESULTS = 3  # Smaller for testing
    return config


@pytest.fixture
def mock_course_data():
    """Create mock course data for testing"""
    course1 = Course(
        title="Introduction to Python",
        instructor="John Doe",
        course_link="https://example.com/python-intro",
        lessons=[
            Lesson(lesson_number=1, title="Variables and Data Types", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Control Structures", lesson_link="https://example.com/lesson2"),
        ]
    )
    
    course2 = Course(
        title="Advanced Machine Learning",
        instructor="Jane Smith", 
        course_link="https://example.com/ml-advanced",
        lessons=[
            Lesson(lesson_number=1, title="Neural Networks", lesson_link="https://example.com/ml-lesson1"),
            Lesson(lesson_number=2, title="Deep Learning", lesson_link="https://example.com/ml-lesson2"),
        ]
    )
    
    return [course1, course2]


@pytest.fixture
def mock_course_chunks():
    """Create mock course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Python is a high-level programming language. Variables store data values.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Control structures like if statements help control program flow.",
            course_title="Introduction to Python", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks.",
            course_title="Advanced Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Deep learning uses neural networks with many layers.",
            course_title="Advanced Machine Learning",
            lesson_number=2, 
            chunk_index=1
        ),
    ]
    return chunks


@pytest.fixture
def test_vector_store(test_config):
    """Create a test vector store with temporary ChromaDB"""
    vector_store = VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
    )
    yield vector_store
    # Cleanup
    if os.path.exists(test_config.CHROMA_PATH):
        shutil.rmtree(test_config.CHROMA_PATH)


@pytest.fixture 
def populated_vector_store(test_vector_store, mock_course_data, mock_course_chunks):
    """Create a vector store populated with test data"""
    # Add course metadata
    for course in mock_course_data:
        test_vector_store.add_course_metadata(course)
    
    # Add course content chunks
    test_vector_store.add_course_content(mock_course_chunks)
    
    return test_vector_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = MagicMock()
    
    # Mock response for non-tool calls
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Test response"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock tool use response from Anthropic"""
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_block]
    
    return mock_response


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=["Python is a programming language", "Control structures help program flow"],
        metadata=[
            {"course_title": "Introduction to Python", "lesson_number": 1},
            {"course_title": "Introduction to Python", "lesson_number": 2}
        ],
        distances=[0.1, 0.3]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


def cleanup_test_db(chroma_path: str):
    """Cleanup test database"""
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)