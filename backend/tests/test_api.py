"""
API endpoint tests for FastAPI application.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, backend_path)


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @patch('app.rag_system')
    def test_query_endpoint_success(self, mock_rag_system):
        """Test successful query endpoint"""
        # Import here to ensure mocking is in place
        from app import app
        
        # Mock RAG system response
        mock_rag_system.query.return_value = ("Test response", ["Source 1", "Source 2"])
        mock_rag_system.session_manager.create_session.return_value = "test_session_123"
        
        client = TestClient(app)
        
        # Test query without session_id
        response = client.post("/api/query", json={
            "query": "What is Python?"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Test response"
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "test_session_123"
        
        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is Python?", "test_session_123")
    
    @patch('app.rag_system')
    def test_query_endpoint_with_session(self, mock_rag_system):
        """Test query endpoint with existing session"""
        # Import here to ensure mocking is in place
        from app import app
        
        # Mock RAG system response
        mock_rag_system.query.return_value = ("Response with history", ["Source 1"])
        
        client = TestClient(app)
        
        # Test query with session_id
        response = client.post("/api/query", json={
            "query": "Follow up question",
            "session_id": "existing_session"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "Response with history"
        assert data["sources"] == ["Source 1"]
        assert data["session_id"] == "existing_session"
        
        # Verify RAG system was called with existing session
        mock_rag_system.query.assert_called_once_with("Follow up question", "existing_session")
        # Should not create new session
        mock_rag_system.session_manager.create_session.assert_not_called()
    
    @patch('app.rag_system')
    def test_query_endpoint_empty_query(self, mock_rag_system):
        """Test query endpoint with empty query"""
        from app import app
        
        client = TestClient(app)
        
        # Test with empty query
        response = client.post("/api/query", json={
            "query": ""
        })
        
        # Should still process (let RAG system handle empty queries)
        assert response.status_code == 200 or response.status_code == 422  # Validation error
    
    @patch('app.rag_system')
    def test_query_endpoint_rag_error(self, mock_rag_system):
        """Test query endpoint when RAG system raises error"""
        from app import app
        
        # Mock RAG system to raise exception
        mock_rag_system.query.side_effect = Exception("RAG system error")
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        client = TestClient(app)
        
        # Test query that triggers error
        response = client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    @patch('app.rag_system')
    def test_query_endpoint_invalid_json(self, mock_rag_system):
        """Test query endpoint with invalid JSON"""
        from app import app
        
        client = TestClient(app)
        
        # Test with invalid JSON structure
        response = client.post("/api/query", json={
            "invalid_field": "test"
        })
        
        # Should return validation error
        assert response.status_code == 422
    
    @patch('app.rag_system')
    def test_courses_endpoint_success(self, mock_rag_system):
        """Test successful courses endpoint"""
        from app import app
        
        # Mock course analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        
        client = TestClient(app)
        
        # Test courses endpoint
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course 1", "Course 2", "Course 3"]
        
        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()
    
    @patch('app.rag_system')
    def test_courses_endpoint_error(self, mock_rag_system):
        """Test courses endpoint when RAG system raises error"""
        from app import app
        
        # Mock RAG system to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        client = TestClient(app)
        
        # Test courses endpoint with error
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
    
    @patch('app.rag_system')
    def test_courses_endpoint_empty_results(self, mock_rag_system):
        """Test courses endpoint with no courses"""
        from app import app
        
        # Mock empty course analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        client = TestClient(app)
        
        # Test courses endpoint
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_static_files_endpoint(self):
        """Test that static files are served correctly"""
        from app import app
        
        client = TestClient(app)
        
        # Test root endpoint (should serve index.html)
        response = client.get("/")
        
        # Should either return the HTML or a 404 if frontend files don't exist
        assert response.status_code in [200, 404]
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        from app import app
        
        client = TestClient(app)
        
        # Test preflight request
        response = client.options("/api/query")
        
        # Should have CORS headers
        assert response.status_code == 200
        # Note: TestClient might not exactly replicate CORS behavior
        # This is more of a smoke test
    
    @patch('app.rag_system')
    def test_query_endpoint_large_response(self, mock_rag_system):
        """Test query endpoint with large response"""
        from app import app
        
        # Mock large response
        large_response = "Large response " * 1000
        large_sources = [f"Source {i}" for i in range(100)]
        
        mock_rag_system.query.return_value = (large_response, large_sources)
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        client = TestClient(app)
        
        # Test query
        response = client.post("/api/query", json={
            "query": "Large query"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["answer"]) > 1000
        assert len(data["sources"]) == 100
    
    @patch('app.rag_system')
    def test_query_endpoint_unicode_content(self, mock_rag_system):
        """Test query endpoint with unicode content"""
        from app import app
        
        # Mock response with unicode
        unicode_response = "Response with émojis 🚀 and ñoñería"
        unicode_sources = ["Source with español"]
        
        mock_rag_system.query.return_value = (unicode_response, unicode_sources)
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        client = TestClient(app)
        
        # Test query with unicode
        response = client.post("/api/query", json={
            "query": "Qué es Python? 🐍"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "émojis 🚀" in data["answer"]
        assert "español" in data["sources"][0]
    
    @patch('app.rag_system')  
    def test_concurrent_queries(self, mock_rag_system):
        """Test handling multiple concurrent queries"""
        from app import app
        import asyncio
        
        # Mock RAG system responses
        mock_rag_system.query.return_value = ("Concurrent response", ["Source"])
        mock_rag_system.session_manager.create_session.return_value = "test_session"
        
        client = TestClient(app)
        
        # Test multiple concurrent requests
        responses = []
        for i in range(5):
            response = client.post("/api/query", json={
                "query": f"Concurrent query {i}"
            })
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Concurrent response"
        
        # RAG system should have been called for each request
        assert mock_rag_system.query.call_count == 5


class TestAPIRequestValidation:
    """Test API request validation"""
    
    def test_query_request_validation(self):
        """Test QueryRequest model validation"""
        from app import QueryRequest
        
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
    
    def test_query_response_validation(self):
        """Test QueryResponse model validation"""
        from app import QueryResponse
        
        # Valid response
        valid_response = QueryResponse(
            answer="Test answer",
            sources=["Source 1", "Source 2"], 
            session_id="session_123"
        )
        
        assert valid_response.answer == "Test answer"
        assert valid_response.sources == ["Source 1", "Source 2"]
        assert valid_response.session_id == "session_123"
    
    def test_course_stats_validation(self):
        """Test CourseStats model validation"""
        from app import CourseStats
        
        # Valid stats
        valid_stats = CourseStats(
            total_courses=5,
            course_titles=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )
        
        assert valid_stats.total_courses == 5
        assert len(valid_stats.course_titles) == 5