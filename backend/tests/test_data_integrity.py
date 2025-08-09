"""
Data integrity tests for RAG system database and document processing.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

from vector_store import VectorStore
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from config import config


class TestDataIntegrity:
    """Test data integrity and document processing"""
    
    @pytest.mark.slow
    def test_actual_chromadb_data_existence(self):
        """Test if actual ChromaDB has data loaded"""
        # Use actual config to check real database
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            # Check if courses exist
            course_count = vector_store.get_course_count()
            print(f"Found {course_count} courses in database")
            
            if course_count == 0:
                pytest.fail("No courses found in ChromaDB - data may not be loaded")
            
            # Check course titles
            course_titles = vector_store.get_existing_course_titles()
            print(f"Course titles: {course_titles}")
            
            assert len(course_titles) > 0
            
            # Try a basic search
            results = vector_store.search("python programming")
            print(f"Search returned {len(results.documents)} results")
            
            if results.error:
                pytest.fail(f"Search failed with error: {results.error}")
            
        except Exception as e:
            pytest.fail(f"Failed to access actual ChromaDB: {e}")
    
    @pytest.mark.slow
    def test_document_processing_real_files(self):
        """Test document processing with real course files"""
        docs_path = "/Users/llo/Development/claude-code-course/ragchatbot/docs"
        
        if not os.path.exists(docs_path):
            pytest.skip("Docs folder not found")
        
        # List actual files
        files = [f for f in os.listdir(docs_path) if f.endswith(('.txt', '.pdf', '.docx'))]
        
        if not files:
            pytest.skip("No document files found")
        
        print(f"Found document files: {files}")
        
        # Test document processor with real files
        try:
            processor = DocumentProcessor(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            for file in files[:1]:  # Test just first file to avoid long test
                file_path = os.path.join(docs_path, file)
                print(f"Processing file: {file}")
                
                course, chunks = processor.process_course_document(file_path)
                
                assert course is not None, f"Failed to process course from {file}"
                assert len(chunks) > 0, f"No chunks created from {file}"
                assert course.title, f"Course title is empty for {file}"
                
                # Check chunk content
                for chunk in chunks[:3]:  # Check first 3 chunks
                    assert chunk.content, f"Empty chunk content in {file}"
                    assert chunk.course_title == course.title, f"Mismatched course title in chunk"
                    assert chunk.chunk_index >= 0, f"Invalid chunk index in {file}"
                
                print(f"Successfully processed {file}: {course.title} with {len(chunks)} chunks")
                
        except Exception as e:
            pytest.fail(f"Document processing failed: {e}")
    
    @pytest.mark.slow  
    def test_end_to_end_data_flow(self, test_config):
        """Test complete data flow from documents to search results"""
        docs_path = "/Users/llo/Development/claude-code-course/ragchatbot/docs"
        
        if not os.path.exists(docs_path):
            pytest.skip("Docs folder not found")
        
        # Create test RAG system
        rag_system = RAGSystem(test_config)
        
        try:
            # Load documents
            courses_added, chunks_added = rag_system.add_course_folder(docs_path)
            
            if courses_added == 0:
                pytest.skip("No courses were successfully processed")
            
            print(f"Added {courses_added} courses with {chunks_added} chunks")
            
            # Test basic search functionality
            search_tool = rag_system.search_tool
            
            # Try various searches
            test_queries = [
                "python",
                "programming", 
                "course introduction",
                "lesson 1"
            ]
            
            for query in test_queries:
                result = search_tool.execute(query)
                print(f"Query '{query}' result length: {len(result)}")
                
                # Should not be an error message
                assert not result.startswith("Search error:"), f"Search failed for query: {query}"
                
                # If no results, that's ok for some queries
                if result.startswith("No relevant content found"):
                    print(f"No results for query: {query}")
                else:
                    # Should contain course context
                    assert "[" in result and "]" in result, f"Missing course context in result for: {query}"
                    
        except Exception as e:
            pytest.fail(f"End-to-end data flow failed: {e}")
        finally:
            # Cleanup
            import shutil
            if os.path.exists(test_config.CHROMA_PATH):
                shutil.rmtree(test_config.CHROMA_PATH)
    
    def test_chromadb_consistency(self, populated_vector_store):
        """Test ChromaDB data consistency"""
        # Check that course catalog and content are in sync
        course_titles_catalog = populated_vector_store.get_existing_course_titles()
        
        # Get all content
        content_results = populated_vector_store.course_content.get()
        content_course_titles = set()
        
        for metadata in content_results['metadatas']:
            content_course_titles.add(metadata['course_title'])
        
        # Every course in content should exist in catalog
        for title in content_course_titles:
            assert title in course_titles_catalog, f"Course '{title}' in content but not in catalog"
        
        print(f"Verified consistency between {len(course_titles_catalog)} catalog entries and {len(content_course_titles)} content courses")
    
    def test_chunk_metadata_integrity(self, populated_vector_store):
        """Test that chunk metadata is properly structured"""
        content_results = populated_vector_store.course_content.get()
        
        required_fields = ['course_title', 'chunk_index']
        
        for i, metadata in enumerate(content_results['metadatas']):
            # Check required fields
            for field in required_fields:
                assert field in metadata, f"Missing {field} in chunk metadata {i}"
                assert metadata[field] is not None, f"Null {field} in chunk metadata {i}"
            
            # Check data types
            assert isinstance(metadata['course_title'], str), f"Invalid course_title type in chunk {i}"
            assert isinstance(metadata['chunk_index'], int), f"Invalid chunk_index type in chunk {i}"
            
            # Lesson number can be None or int
            if 'lesson_number' in metadata:
                lesson_num = metadata['lesson_number']
                assert lesson_num is None or isinstance(lesson_num, int), f"Invalid lesson_number type in chunk {i}"
    
    def test_course_metadata_integrity(self, populated_vector_store):
        """Test that course metadata is properly structured"""
        catalog_results = populated_vector_store.course_catalog.get()
        
        required_fields = ['title', 'lesson_count', 'lessons_json']
        
        for i, metadata in enumerate(catalog_results['metadatas']):
            # Check required fields
            for field in required_fields:
                assert field in metadata, f"Missing {field} in course metadata {i}"
                assert metadata[field] is not None, f"Null {field} in course metadata {i}"
            
            # Check data types
            assert isinstance(metadata['title'], str), f"Invalid title type in course {i}"
            assert isinstance(metadata['lesson_count'], int), f"Invalid lesson_count type in course {i}"
            assert isinstance(metadata['lessons_json'], str), f"Invalid lessons_json type in course {i}"
            
            # Validate lessons JSON
            import json
            try:
                lessons = json.loads(metadata['lessons_json'])
                assert isinstance(lessons, list), f"lessons_json should be a list in course {i}"
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in lessons_json for course {i}")
    
    def test_search_result_completeness(self, populated_vector_store):
        """Test that search results contain all necessary information"""
        results = populated_vector_store.search("programming")
        
        if results.is_empty():
            pytest.skip("No search results to test")
        
        # Check that we have same number of documents, metadata, and distances
        assert len(results.documents) == len(results.metadata), "Mismatched documents and metadata count"
        assert len(results.documents) == len(results.distances), "Mismatched documents and distances count"
        
        # Check each result has required fields
        for i, (doc, metadata, distance) in enumerate(zip(results.documents, results.metadata, results.distances)):
            assert doc, f"Empty document at index {i}"
            assert isinstance(doc, str), f"Document should be string at index {i}"
            
            assert metadata, f"Empty metadata at index {i}"
            assert isinstance(metadata, dict), f"Metadata should be dict at index {i}"
            assert 'course_title' in metadata, f"Missing course_title in metadata at index {i}"
            
            assert isinstance(distance, (int, float)), f"Distance should be numeric at index {i}"
            assert distance >= 0, f"Distance should be non-negative at index {i}"
    
    @pytest.mark.slow
    def test_real_chromadb_search_quality(self):
        """Test search quality with real ChromaDB data"""
        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=config.MAX_RESULTS
            )
            
            # Test queries that should return relevant results
            quality_tests = [
                {
                    "query": "python variables",
                    "should_contain": ["python", "variable"]
                },
                {
                    "query": "course introduction",
                    "should_contain": ["course", "introduction"]  
                },
                {
                    "query": "lesson 1",
                    "should_contain": ["lesson"]
                }
            ]
            
            for test in quality_tests:
                results = vector_store.search(test["query"])
                
                if results.error:
                    pytest.fail(f"Search failed for '{test['query']}': {results.error}")
                
                if results.is_empty():
                    print(f"No results for quality test query: {test['query']}")
                    continue
                
                # Check that results are somewhat relevant
                combined_text = " ".join(results.documents).lower()
                
                relevant_found = any(term.lower() in combined_text for term in test["should_contain"])
                
                if not relevant_found:
                    print(f"Warning: Query '{test['query']}' may have low relevance results")
                    print(f"Expected terms: {test['should_contain']}")
                    print(f"First result: {results.documents[0][:200]}...")
                
        except Exception as e:
            pytest.fail(f"Real ChromaDB search quality test failed: {e}")
    
    def test_course_resolution_accuracy(self, populated_vector_store):
        """Test accuracy of course name resolution"""
        # Test exact matches
        exact_matches = [
            "Introduction to Python",
            "Advanced Machine Learning"
        ]
        
        for course_name in exact_matches:
            resolved = populated_vector_store._resolve_course_name(course_name)
            assert resolved == course_name, f"Exact match failed for: {course_name}"
        
        # Test partial matches
        partial_matches = [
            ("Python", "Introduction to Python"),
            ("Machine Learning", "Advanced Machine Learning"),
            ("Introduction", "Introduction to Python")
        ]
        
        for partial, expected in partial_matches:
            resolved = populated_vector_store._resolve_course_name(partial)
            assert resolved == expected, f"Partial match failed: '{partial}' should resolve to '{expected}', got '{resolved}'"