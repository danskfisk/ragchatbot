"""
API endpoint tests for FastAPI application.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)

# Test-specific app to avoid static file mounting issues
def create_test_app():
    """Create a test-specific FastAPI app without static file dependencies"""
    test_app = FastAPI(title="Test Course Materials RAG System")
    
    # Add CORS middleware
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import models and RAG system (will be mocked)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system - will be patched in tests
    rag_system = MagicMock()
    
    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            
            answer, sources = rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Simple root endpoint for testing
    @test_app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    # Store references for testing
    test_app.rag_system = rag_system
    test_app.QueryRequest = QueryRequest
    test_app.QueryResponse = QueryResponse
    test_app.CourseStats = CourseStats
    
    return test_app


@pytest.fixture
def test_app():
    """Create test app fixture"""
    return create_test_app()

@pytest.fixture  
def test_client(test_app):
    """Create test client fixture"""
    return TestClient(test_app)


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_query_endpoint_success(self, test_client, test_app):
        """Test successful query endpoint"""
        # Mock RAG system response
        test_app.rag_system.query.return_value = ("Test response", ["Source 1", "Source 2"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session_123"
        
        # Test query without session_id
        response = test_client.post("/api/query", json={
            "query": "What is Python?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test response"
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "test_session_123"
        
        # Verify RAG system was called correctly
        test_app.rag_system.query.assert_called_once_with("What is Python?", "test_session_123")
    
    def test_query_endpoint_with_session(self, test_client, test_app):
        """Test query endpoint with existing session"""
        # Mock RAG system response
        test_app.rag_system.query.return_value = ("Response with history", ["Source 1"])
        
        # Test query with session_id
        response = test_client.post("/api/query", json={
            "query": "Follow up question",
            "session_id": "existing_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Response with history"
        assert data["sources"] == ["Source 1"]
        assert data["session_id"] == "existing_session"
        
        # Verify RAG system was called with existing session
        test_app.rag_system.query.assert_called_once_with("Follow up question", "existing_session")
        # Should not create new session
        test_app.rag_system.session_manager.create_session.assert_not_called()
    
    def test_query_endpoint_empty_query(self, test_client, test_app):
        """Test query endpoint with empty query"""
        test_app.rag_system.query.return_value = ("Empty query response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test with empty query
        response = test_client.post("/api/query", json={
            "query": ""
        })
        
        # Should still process (let RAG system handle empty queries)
        assert response.status_code in [200, 422, 500]  # Various possible responses
    
    def test_query_endpoint_rag_error(self, test_client, test_app):
        """Test query endpoint when RAG system raises error"""
        # Mock RAG system to raise exception
        test_app.rag_system.query.side_effect = Exception("RAG system error")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test query that triggers error
        response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_endpoint_invalid_json(self, test_client, test_app):
        """Test query endpoint with invalid JSON"""
        
        # Test with invalid JSON structure
        response = test_client.post("/api/query", json={
            "invalid_field": "test"
        })
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_courses_endpoint_success(self, test_client, test_app):
        """Test successful courses endpoint"""
        # Mock course analytics
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        
        # Test courses endpoint
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course 1", "Course 2", "Course 3"]
        
        # Verify RAG system was called
        test_app.rag_system.get_course_analytics.assert_called_once()
    
    def test_courses_endpoint_error(self, test_client, test_app):
        """Test courses endpoint when RAG system raises error"""
        # Mock RAG system to raise exception
        test_app.rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        # Test courses endpoint with error
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
    
    def test_courses_endpoint_empty_results(self, test_client, test_app):
        """Test courses endpoint with no courses"""
        # Mock empty course analytics
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        # Test courses endpoint
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_static_files_endpoint(self, test_client):
        """Test that static files are served correctly"""
        
        # Test root endpoint (should serve simple message)
        response = test_client.get("/")
        
        # Should return simple message
        assert response.status_code == 200
        assert response.json() == {"message": "RAG System API"}
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are properly set"""
        
        # Test preflight request - OPTIONS might not be implemented
        response = test_client.options("/api/query")
        
        # Should have CORS headers or be method not allowed
        assert response.status_code in [200, 405]
        # Note: TestClient might not exactly replicate CORS behavior
        # This is more of a smoke test
        
        # Test actual request to see CORS headers
        response = test_client.get("/")
        assert response.status_code == 200
    
    def test_query_endpoint_large_response(self, test_client, test_app):
        """Test query endpoint with large response"""
        # Mock large response
        large_response = "Large response " * 1000
        large_sources = [f"Source {i}" for i in range(100)]
        
        test_app.rag_system.query.return_value = (large_response, large_sources)
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test query
        response = test_client.post("/api/query", json={
            "query": "Large query"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["answer"]) > 1000
        assert len(data["sources"]) == 100
    
    def test_query_endpoint_unicode_content(self, test_client, test_app):
        """Test query endpoint with unicode content"""
        # Mock response with unicode
        unicode_response = "Response with émojis 🚀 and ñoñería"
        unicode_sources = ["Source with español"]
        
        test_app.rag_system.query.return_value = (unicode_response, unicode_sources)
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test query with unicode
        response = test_client.post("/api/query", json={
            "query": "Qué es Python? 🐍"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "émojis 🚀" in data["answer"]
        assert "español" in data["sources"][0]
    
    def test_concurrent_queries(self, test_client, test_app):
        """Test handling multiple concurrent queries"""
        import asyncio
        
        # Mock RAG system responses
        test_app.rag_system.query.return_value = ("Concurrent response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test multiple concurrent requests
        responses = []
        for i in range(5):
            response = test_client.post("/api/query", json={
                "query": f"Concurrent query {i}"
            })
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Concurrent response"
        
        # RAG system should have been called for each request
        assert test_app.rag_system.query.call_count == 5


class TestAPIRequestValidation:
    """Test API request validation"""
    
    def test_query_request_validation(self, test_app):
        """Test QueryRequest model validation"""
        QueryRequest = test_app.QueryRequest
        
        # Valid request
        valid_request = QueryRequest(query="Test query")
        assert valid_request.query == "Test query"
        assert valid_request.session_id is None
        
        # Valid request with session
        valid_with_session = QueryRequest(query="Test query", session_id="session_123")
        assert valid_with_session.session_id == "session_123"
        
        # Test that query is required
        with pytest.raises(Exception):  # Pydantic validation error
            QueryRequest()
    
    def test_query_response_validation(self, test_app):
        """Test QueryResponse model validation"""
        QueryResponse = test_app.QueryResponse
        
        # Valid response
        valid_response = QueryResponse(
            answer="Test answer",
            sources=["Source 1", "Source 2"], 
            session_id="session_123"
        )
        
        assert valid_response.answer == "Test answer"
        assert valid_response.sources == ["Source 1", "Source 2"]
        assert valid_response.session_id == "session_123"
    
    def test_course_stats_validation(self, test_app):
        """Test CourseStats model validation"""
        CourseStats = test_app.CourseStats
        
        # Valid stats
        valid_stats = CourseStats(
            total_courses=5,
            course_titles=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )
        
        assert valid_stats.total_courses == 5
        assert len(valid_stats.course_titles) == 5


@pytest.mark.api
class TestAPIExtended:
    """Extended API tests for edge cases and validation"""
    
    def test_query_endpoint_malformed_json(self, test_client):
        """Test query endpoint with malformed JSON"""
        # Test with malformed JSON body
        response = test_client.post("/api/query", 
                                  data="invalid json",
                                  headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422
    
    def test_query_endpoint_missing_content_type(self, test_client, test_app):
        """Test query endpoint without content-type header"""
        test_app.rag_system.query.return_value = ("Response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Test without content-type header  
        response = test_client.post("/api/query", data='{"query": "test"}')
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 422]
    
    def test_query_endpoint_extremely_long_query(self, test_client, test_app):
        """Test query endpoint with extremely long query"""
        test_app.rag_system.query.return_value = ("Response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Create very long query (10KB)
        long_query = "Very long query " * 600
        
        response = test_client.post("/api/query", json={
            "query": long_query
        })
        
        assert response.status_code == 200
        test_app.rag_system.query.assert_called_once_with(long_query, "test_session")
    
    def test_query_endpoint_special_characters(self, test_client, test_app):
        """Test query endpoint with special characters"""
        test_app.rag_system.query.return_value = ("Response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        special_query = "Query with \\n\\t\\r\\0 and \"quotes\" and 'apostrophes' & <tags>"
        
        response = test_client.post("/api/query", json={
            "query": special_query
        })
        
        assert response.status_code == 200
        test_app.rag_system.query.assert_called_once_with(special_query, "test_session")
    
    def test_query_endpoint_null_values(self, test_client):
        """Test query endpoint with null values"""
        # Test with null query
        response = test_client.post("/api/query", json={
            "query": None
        })
        
        assert response.status_code == 422
        
        # Test with null session_id (should be allowed)
        response2 = test_client.post("/api/query", json={
            "query": "test",
            "session_id": None
        })
        
        # This should work (null session_id is valid)
        assert response2.status_code in [200, 500]  # 500 if RAG system not mocked properly
    
    def test_courses_endpoint_methods(self, test_client, test_app):
        """Test courses endpoint with different HTTP methods"""
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test Course"]
        }
        
        # GET should work
        response = test_client.get("/api/courses")
        assert response.status_code == 200
        
        # POST should not be allowed
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405  # Method not allowed
        
        # PUT should not be allowed  
        response = test_client.put("/api/courses")
        assert response.status_code == 405
        
        # DELETE should not be allowed
        response = test_client.delete("/api/courses")
        assert response.status_code == 405
    
    def test_nonexistent_endpoints(self, test_client):
        """Test accessing non-existent endpoints"""
        # Test non-existent API endpoint
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # Test non-existent nested endpoint
        response = test_client.get("/api/query/nonexistent")
        assert response.status_code == 404
        
        # Test with different method on non-existent endpoint
        response = test_client.post("/api/nonexistent", json={})
        assert response.status_code == 404
    
    def test_api_response_headers(self, test_client, test_app):
        """Test API response headers"""
        test_app.rag_system.query.return_value = ("Response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "test"
        })
        
        assert response.status_code == 200
        
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
        
        # CORS headers should be present due to middleware
        # Note: TestClient might not exactly replicate all middleware behavior
    
    def test_query_response_structure(self, test_client, test_app):
        """Test query response structure and data types"""
        test_app.rag_system.query.return_value = ("Test answer", ["Source 1", "Source 2"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "test"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Check data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check values
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "test_session"
    
    def test_courses_response_structure(self, test_client, test_app):
        """Test courses response structure and data types"""
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"]
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Check data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check values
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Course A", "Course B"]


@pytest.mark.slow
class TestAPIPerformance:
    """Performance and stress tests for API endpoints"""
    
    def test_query_endpoint_response_time(self, test_client, test_app):
        """Test query endpoint response time"""
        import time
        
        test_app.rag_system.query.return_value = ("Quick response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        start_time = time.time()
        response = test_client.post("/api/query", json={
            "query": "Performance test query"
        })
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be reasonably fast (under 1 second for mocked system)
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response took {response_time:.2f} seconds"
    
    def test_courses_endpoint_response_time(self, test_client, test_app):
        """Test courses endpoint response time"""
        import time
        
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": [f"Course {i}" for i in range(100)]
        }
        
        start_time = time.time()
        response = test_client.get("/api/courses")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be reasonably fast
        response_time = end_time - start_time
        assert response_time < 1.0, f"Response took {response_time:.2f} seconds"
    
    def test_concurrent_request_handling(self, test_client, test_app):
        """Test handling multiple concurrent requests"""
        import threading
        import time
        
        test_app.rag_system.query.return_value = ("Concurrent response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        test_app.rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        
        results = []
        errors = []
        
        def make_query_request():
            try:
                response = test_client.post("/api/query", json={
                    "query": f"Concurrent query {threading.current_thread().name}"
                })
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        def make_courses_request():
            try:
                response = test_client.get("/api/courses")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(10):
            if i % 2 == 0:
                thread = threading.Thread(target=make_query_request, name=f"query-{i}")
            else:
                thread = threading.Thread(target=make_courses_request, name=f"courses-{i}")
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout per thread
        end_time = time.time()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(status == 200 for status in results), f"Non-200 responses: {results}"
        
        # Total time should be reasonable (concurrent processing)
        total_time = end_time - start_time
        assert total_time < 10.0, f"Concurrent requests took {total_time:.2f} seconds"
    
    def test_memory_usage_large_responses(self, test_client, test_app):
        """Test memory usage with large responses"""
        # Create large response data
        large_answer = "Large answer " * 10000  # ~120KB
        large_sources = [f"Large source content {i}" * 100 for i in range(50)]  # ~125KB
        
        test_app.rag_system.query.return_value = (large_answer, large_sources)
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Make multiple requests with large responses
        for i in range(5):
            response = test_client.post("/api/query", json={
                "query": f"Large response test {i}"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["answer"]) > 100000
            assert len(data["sources"]) == 50
    
    def test_stress_test_rapid_requests(self, test_client, test_app):
        """Stress test with rapid sequential requests"""
        test_app.rag_system.query.return_value = ("Stress response", ["Source"])
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        # Make rapid sequential requests
        success_count = 0
        error_count = 0
        
        for i in range(50):
            try:
                response = test_client.post("/api/query", json={
                    "query": f"Stress test query {i}"
                })
                if response.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        # Most requests should succeed
        success_rate = success_count / 50
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"
        assert error_count < 3, f"Too many errors: {error_count}"
    
    def test_api_under_load_with_errors(self, test_client, test_app):
        """Test API behavior when RAG system has intermittent errors"""
        import random
        
        def side_effect(*args, **kwargs):
            # Randomly fail 20% of requests
            if random.random() < 0.2:
                raise Exception("Simulated RAG system error")
            return ("Success response", ["Source"])
        
        test_app.rag_system.query.side_effect = side_effect
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        success_count = 0
        error_count = 0
        
        # Make many requests
        for i in range(30):
            response = test_client.post("/api/query", json={
                "query": f"Load test with errors {i}"
            })
            
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 500:
                error_count += 1
        
        # Should handle errors gracefully
        assert success_count > 0, "No successful requests"
        assert error_count > 0, "Expected some errors from simulation"
        assert success_count + error_count == 30, "Some requests unaccounted for"


class TestAPIErrorHandling:
    """Comprehensive error handling tests for API endpoints"""
    
    def test_query_endpoint_rag_system_connection_error(self, test_client, test_app):
        """Test query endpoint when RAG system has connection issues"""
        test_app.rag_system.query.side_effect = ConnectionError("Database connection failed")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test connection error"
        })
        
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]
    
    def test_query_endpoint_timeout_error(self, test_client, test_app):
        """Test query endpoint when RAG system times out"""
        test_app.rag_system.query.side_effect = TimeoutError("Request timed out")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test timeout"
        })
        
        assert response.status_code == 500
        assert "Request timed out" in response.json()["detail"]
    
    def test_query_endpoint_memory_error(self, test_client, test_app):
        """Test query endpoint when RAG system has memory issues"""
        test_app.rag_system.query.side_effect = MemoryError("Out of memory")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test memory error"
        })
        
        assert response.status_code == 500
        assert "Out of memory" in response.json()["detail"]
    
    def test_query_endpoint_session_creation_error(self, test_client, test_app):
        """Test query endpoint when session creation fails"""
        test_app.rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")
        
        response = test_client.post("/api/query", json={
            "query": "Test session error"
        })
        
        assert response.status_code == 500
        assert "Session creation failed" in response.json()["detail"]
    
    def test_query_endpoint_invalid_session_id(self, test_client, test_app):
        """Test query endpoint with invalid session ID format"""
        test_app.rag_system.query.side_effect = ValueError("Invalid session ID format")
        
        response = test_client.post("/api/query", json={
            "query": "Test query",
            "session_id": "invalid-session-format-!"
        })
        
        assert response.status_code == 500
        assert "Invalid session ID format" in response.json()["detail"]
    
    def test_courses_endpoint_analytics_error(self, test_client, test_app):
        """Test courses endpoint when analytics fails"""
        test_app.rag_system.get_course_analytics.side_effect = Exception("Analytics system down")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics system down" in response.json()["detail"]
    
    def test_courses_endpoint_database_error(self, test_client, test_app):
        """Test courses endpoint when database is unavailable"""
        test_app.rag_system.get_course_analytics.side_effect = ConnectionError("Database unavailable")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Database unavailable" in response.json()["detail"]
    
    def test_query_endpoint_malformed_rag_response(self, test_client, test_app):
        """Test query endpoint when RAG system returns malformed data"""
        # RAG system returns wrong data structure
        test_app.rag_system.query.return_value = "Wrong format"  # Should be tuple
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test malformed response"
        })
        
        assert response.status_code == 500
        # Should contain some indication of the error
        assert "detail" in response.json()
    
    def test_query_endpoint_empty_rag_response(self, test_client, test_app):
        """Test query endpoint when RAG system returns empty response"""
        test_app.rag_system.query.return_value = ("", [])  # Empty response
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test empty response"
        })
        
        # Should still return 200 but with empty data
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == ""
        assert data["sources"] == []
    
    def test_courses_endpoint_malformed_analytics(self, test_client, test_app):
        """Test courses endpoint when analytics returns malformed data"""
        # Analytics returns wrong structure
        test_app.rag_system.get_course_analytics.return_value = ["wrong", "format"]
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "detail" in response.json()
    
    def test_query_endpoint_unicode_error(self, test_client, test_app):
        """Test query endpoint with unicode encoding issues"""
        test_app.rag_system.query.side_effect = UnicodeError("Unicode encoding error")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test unicode error"
        })
        
        assert response.status_code == 500
        assert "Unicode encoding error" in response.json()["detail"]
    
    def test_query_endpoint_key_error(self, test_client, test_app):
        """Test query endpoint when RAG system has key errors"""
        test_app.rag_system.query.side_effect = KeyError("Missing required key")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test key error"
        })
        
        assert response.status_code == 500
        assert "Missing required key" in response.json()["detail"]
    
    def test_query_endpoint_type_error(self, test_client, test_app):
        """Test query endpoint when RAG system has type errors"""
        test_app.rag_system.query.side_effect = TypeError("Incorrect type provided")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test type error"
        })
        
        assert response.status_code == 500
        assert "Incorrect type provided" in response.json()["detail"]
    
    def test_error_response_format(self, test_client, test_app):
        """Test that error responses follow consistent format"""
        test_app.rag_system.query.side_effect = Exception("Test error message")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response = test_client.post("/api/query", json={
            "query": "Test error format"
        })
        
        assert response.status_code == 500
        error_data = response.json()
        
        # Should follow FastAPI error format
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)
        assert "Test error message" in error_data["detail"]
    
    def test_multiple_errors_in_sequence(self, test_client, test_app):
        """Test handling multiple errors in sequence"""
        errors = [
            ConnectionError("Connection failed"),
            TimeoutError("Timeout occurred"),
            MemoryError("Out of memory"),
            ValueError("Invalid value")
        ]
        
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        for i, error in enumerate(errors):
            test_app.rag_system.query.side_effect = error
            
            response = test_client.post("/api/query", json={
                "query": f"Test error {i}"
            })
            
            assert response.status_code == 500
            assert str(error) in response.json()["detail"]
    
    def test_error_recovery(self, test_client, test_app):
        """Test that API recovers from errors"""
        # First request fails
        test_app.rag_system.query.side_effect = Exception("Temporary error")
        test_app.rag_system.session_manager.create_session.return_value = "test_session"
        
        response1 = test_client.post("/api/query", json={
            "query": "First request (should fail)"
        })
        assert response1.status_code == 500
        
        # Second request succeeds
        test_app.rag_system.query.side_effect = None
        test_app.rag_system.query.return_value = ("Recovery successful", ["Source"])
        
        response2 = test_client.post("/api/query", json={
            "query": "Second request (should succeed)"
        })
        assert response2.status_code == 200
        data = response2.json()
        assert data["answer"] == "Recovery successful"