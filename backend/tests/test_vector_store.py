"""
Unit tests for VectorStore functionality.
"""
import pytest
import tempfile
import shutil
import sys
import os

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults data class"""
    
    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key1': 'val1'}, {'key2': 'val2'}]],
            'distances': [[0.1, 0.3]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key1': 'val1'}, {'key2': 'val2'}]
        assert results.distances == [0.1, 0.3]
        assert results.error is None
    
    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_constructor(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
    
    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(['doc'], [{'key': 'val'}], [0.1])
        
        assert empty_results.is_empty() is True
        assert non_empty_results.is_empty() is False


class TestVectorStore:
    """Test VectorStore functionality"""
    
    def test_initialization(self, test_config):
        """Test VectorStore initialization"""
        vector_store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        
        assert vector_store.max_results == test_config.MAX_RESULTS
        assert vector_store.client is not None
        assert vector_store.course_catalog is not None
        assert vector_store.course_content is not None
        
        # Cleanup
        shutil.rmtree(test_config.CHROMA_PATH)
    
    def test_add_course_metadata(self, test_vector_store, mock_course_data):
        """Test adding course metadata to catalog"""
        course = mock_course_data[0]  # Introduction to Python
        
        test_vector_store.add_course_metadata(course)
        
        # Verify course was added
        result = test_vector_store.course_catalog.get(ids=[course.title])
        assert result is not None
        assert len(result['ids']) == 1
        assert result['ids'][0] == course.title
        
        metadata = result['metadatas'][0]
        assert metadata['title'] == course.title
        assert metadata['instructor'] == course.instructor
        assert metadata['course_link'] == course.course_link
        assert metadata['lesson_count'] == len(course.lessons)
        assert 'lessons_json' in metadata
    
    def test_add_course_content(self, test_vector_store, mock_course_chunks):
        """Test adding course content chunks"""
        chunks = mock_course_chunks[:2]  # First 2 chunks
        
        test_vector_store.add_course_content(chunks)
        
        # Verify chunks were added
        result = test_vector_store.course_content.get()
        assert len(result['ids']) == 2
        
        # Check chunk content and metadata
        for i, chunk in enumerate(chunks):
            assert chunk.content in result['documents']
            
            # Find the metadata for this chunk
            chunk_metadata = None
            for metadata in result['metadatas']:
                if metadata['chunk_index'] == chunk.chunk_index:
                    chunk_metadata = metadata
                    break
            
            assert chunk_metadata is not None
            assert chunk_metadata['course_title'] == chunk.course_title
            assert chunk_metadata['lesson_number'] == chunk.lesson_number
    
    def test_add_empty_course_content(self, test_vector_store):
        """Test adding empty course content list"""
        # Should not raise an exception
        test_vector_store.add_course_content([])
        
        # Verify no content was added
        result = test_vector_store.course_content.get()
        assert len(result['ids']) == 0
    
    def test_resolve_course_name_exact_match(self, populated_vector_store):
        """Test course name resolution with exact match"""
        resolved = populated_vector_store._resolve_course_name("Introduction to Python")
        assert resolved == "Introduction to Python"
    
    def test_resolve_course_name_partial_match(self, populated_vector_store):
        """Test course name resolution with partial match"""
        resolved = populated_vector_store._resolve_course_name("Python")
        assert resolved == "Introduction to Python"
    
    def test_resolve_course_name_fuzzy_match(self, populated_vector_store):
        """Test course name resolution with fuzzy match"""
        resolved = populated_vector_store._resolve_course_name("Machine Learning")
        assert resolved == "Advanced Machine Learning"
    
    def test_resolve_course_name_no_match(self, populated_vector_store):
        """Test course name resolution with no match"""
        resolved = populated_vector_store._resolve_course_name("Nonexistent Course")
        assert resolved is None
    
    def test_build_filter_no_filters(self, test_vector_store):
        """Test building filter with no parameters"""
        filter_dict = test_vector_store._build_filter(None, None)
        assert filter_dict is None
    
    def test_build_filter_course_only(self, test_vector_store):
        """Test building filter with course title only"""
        filter_dict = test_vector_store._build_filter("Introduction to Python", None)
        assert filter_dict == {"course_title": "Introduction to Python"}
    
    def test_build_filter_lesson_only(self, test_vector_store):
        """Test building filter with lesson number only"""
        filter_dict = test_vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}
    
    def test_build_filter_both(self, test_vector_store):
        """Test building filter with both course and lesson"""
        filter_dict = test_vector_store._build_filter("Introduction to Python", 1)
        expected = {"$and": [
            {"course_title": "Introduction to Python"},
            {"lesson_number": 1}
        ]}
        assert filter_dict == expected
    
    def test_search_no_filters(self, populated_vector_store):
        """Test search without any filters"""
        results = populated_vector_store.search("Python programming")
        
        assert results.error is None
        assert not results.is_empty()
        assert len(results.documents) > 0
        # Should find content from Introduction to Python course
        assert any("Python" in doc for doc in results.documents)
    
    def test_search_with_course_filter(self, populated_vector_store):
        """Test search with course name filter"""
        results = populated_vector_store.search("programming", course_name="Introduction to Python")
        
        assert results.error is None
        assert not results.is_empty()
        
        # All results should be from the specified course
        for metadata in results.metadata:
            assert metadata['course_title'] == "Introduction to Python"
    
    def test_search_with_lesson_filter(self, populated_vector_store):
        """Test search with lesson number filter"""
        results = populated_vector_store.search("programming", lesson_number=1)
        
        assert results.error is None
        
        # All results should be from lesson 1
        for metadata in results.metadata:
            assert metadata['lesson_number'] == 1
    
    def test_search_with_both_filters(self, populated_vector_store):
        """Test search with both course and lesson filters"""
        results = populated_vector_store.search(
            "programming",
            course_name="Introduction to Python",
            lesson_number=1
        )
        
        assert results.error is None
        
        # All results should match both filters
        for metadata in results.metadata:
            assert metadata['course_title'] == "Introduction to Python"
            assert metadata['lesson_number'] == 1
    
    def test_search_nonexistent_course(self, populated_vector_store):
        """Test search with non-existent course name"""
        results = populated_vector_store.search("test", course_name="Nonexistent Course")
        
        assert results.error is not None
        assert "No course found matching 'Nonexistent Course'" in results.error
        assert results.is_empty()
    
    def test_search_no_results(self, populated_vector_store):
        """Test search that returns no results"""
        results = populated_vector_store.search("quantum computing blockchain cryptocurrency")
        
        assert results.error is None
        assert results.is_empty()
    
    def test_search_with_custom_limit(self, populated_vector_store):
        """Test search with custom result limit"""
        results = populated_vector_store.search("programming", limit=1)
        
        assert results.error is None
        assert len(results.documents) <= 1
    
    def test_get_existing_course_titles(self, populated_vector_store):
        """Test getting existing course titles"""
        titles = populated_vector_store.get_existing_course_titles()
        
        assert len(titles) == 2
        assert "Introduction to Python" in titles
        assert "Advanced Machine Learning" in titles
    
    def test_get_course_count(self, populated_vector_store):
        """Test getting course count"""
        count = populated_vector_store.get_course_count()
        assert count == 2
    
    def test_get_all_courses_metadata(self, populated_vector_store):
        """Test getting all courses metadata"""
        metadata_list = populated_vector_store.get_all_courses_metadata()
        
        assert len(metadata_list) == 2
        
        # Check that lessons JSON was parsed
        for metadata in metadata_list:
            assert 'lessons' in metadata
            assert 'lessons_json' not in metadata  # Should be removed
            assert isinstance(metadata['lessons'], list)
            assert len(metadata['lessons']) > 0
    
    def test_get_course_link(self, populated_vector_store):
        """Test getting course link"""
        link = populated_vector_store.get_course_link("Introduction to Python")
        assert link == "https://example.com/python-intro"
        
        # Test non-existent course
        link = populated_vector_store.get_course_link("Nonexistent Course")
        assert link is None
    
    def test_get_lesson_link(self, populated_vector_store):
        """Test getting lesson link"""
        link = populated_vector_store.get_lesson_link("Introduction to Python", 1)
        assert link == "https://example.com/lesson1"
        
        # Test non-existent lesson
        link = populated_vector_store.get_lesson_link("Introduction to Python", 99)
        assert link is None
        
        # Test non-existent course
        link = populated_vector_store.get_lesson_link("Nonexistent Course", 1)
        assert link is None
    
    def test_clear_all_data(self, populated_vector_store):
        """Test clearing all data"""
        # Verify data exists
        assert populated_vector_store.get_course_count() > 0
        
        populated_vector_store.clear_all_data()
        
        # Verify data is cleared
        assert populated_vector_store.get_course_count() == 0
        
        # Verify collections still exist and can be used
        result = populated_vector_store.course_catalog.get()
        assert len(result['ids']) == 0
        
        result = populated_vector_store.course_content.get()
        assert len(result['ids']) == 0
    
    def test_empty_vector_store_operations(self, test_vector_store):
        """Test operations on empty vector store"""
        # Should not raise exceptions
        assert test_vector_store.get_course_count() == 0
        assert test_vector_store.get_existing_course_titles() == []
        assert test_vector_store.get_all_courses_metadata() == []
        
        # Search should return empty results
        results = test_vector_store.search("test query")
        assert results.is_empty()
        assert results.error is None