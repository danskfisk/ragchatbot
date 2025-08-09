# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials built with FastAPI, ChromaDB, and Anthropic's Claude. The system enables semantic search and AI-powered responses about educational content.

## Core Architecture

### Backend Components
- **FastAPI App** (`backend/app.py`): Main web server with CORS, static file serving, and API endpoints
- **RAG System** (`backend/rag_system.py`): Central orchestrator that coordinates all components
- **Vector Store** (`backend/vector_store.py`): ChromaDB-based storage with semantic search capabilities
- **AI Generator** (`backend/ai_generator.py`): Anthropic Claude integration with tool-calling support
- **Document Processor** (`backend/document_processor.py`): Handles parsing and chunking of course materials
- **Search Tools** (`backend/search_tools.py`): Tool interface for Claude with course content search
- **Session Manager** (`backend/session_manager.py`): Conversation history management
- **Models** (`backend/models.py`): Pydantic models for Course, Lesson, and CourseChunk data structures

### Data Flow
1. Course documents (PDF/TXT/DOCX) are processed into chunks and stored in ChromaDB
2. User queries trigger semantic search through the VectorStore
3. Search results are passed to Claude via tool-calling interface
4. AI Generator synthesizes responses using search context
5. Session Manager tracks conversation history for follow-up questions

### Key Design Patterns
- **Tool-based Architecture**: Claude uses search tools rather than direct RAG context injection
- **Dual Collections**: Separate ChromaDB collections for course catalog (metadata) and content (chunks)
- **Smart Course Resolution**: Fuzzy matching of course names via semantic search
- **Chunked Content**: Large documents split into searchable segments with overlap

## Development Commands

### Environment Setup
```bash
# Use uv to install dependencies 
uv pip install chromadb==1.0.15 anthropic==0.58.2 sentence-transformers==5.0.0 fastapi==0.116.1 uvicorn==0.35.0 python-multipart==0.0.20 python-dotenv==1.1.1

# Downgrade NumPy for compatibility
uv pip install "numpy<2"

# Create environment file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Start development server
uv run uvicorn backend.app:app --reload --port 8000

# Or use the shell script 
./run.sh
```

### API Endpoints
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Query endpoint: `POST /api/query`
- Course stats: `GET /api/courses`

## Configuration

All settings are centralized in `backend/config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence-transformers)
- `CHUNK_SIZE`: 800 characters with 100 character overlap
- `CHROMA_PATH`: "./chroma_db" (persistent storage location)

## Data Management

### Course Document Processing
- Supports PDF, DOCX, and TXT files in `/docs` folder
- Documents auto-loaded on startup via `startup_event`
- Course metadata extracted and stored separately from content chunks
- Existing courses skipped to avoid duplicates

### Vector Store Structure
- **course_catalog**: Course titles, instructors, lesson metadata for semantic course matching
- **course_content**: Actual content chunks with course/lesson context for search

### Session Management
- Conversation history limited to last 2 exchanges (`MAX_HISTORY: 2`)
- Session IDs auto-generated if not provided
- History included in AI system prompts for context

## Platform-Specific Notes

### macOS x86_64 Compatibility
- PyTorch 2.7.1 is incompatible with macOS x86_64 - use uv as recommended package manager
- NumPy 2.x causes compatibility issues - downgrade to 1.x
- Python 3.12+ required (updated from original 3.13 requirement)

### Alternative Installation
If uv fails due to PyTorch compatibility:
1. Use uv to manage package installations
2. Modify `run.sh` to use `uv run uvicorn` if needed

## Tool System

The AI uses a structured tool interface:
- **CourseSearchTool**: Semantic search with course name resolution and lesson filtering
- **ToolManager**: Registers and executes tools, tracks search sources
- Tool calls are automatically handled by `ai_generator.py` with follow-up response generation

The search tool supports:
- Fuzzy course name matching (e.g., "MCP" matches "MCP Course Introduction")  
- Lesson number filtering for targeted queries
- Source tracking for UI display

## Development Workflow Guidelines
- Always use uv to run server and manage dependencies
- Do not use pip directly