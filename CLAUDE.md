# Local Code Indexing MCP Server

## Overview
This is an MCP (Model Context Protocol) server that indexes codebases using ChromaDB for semantic code search. It provides AI assistants with the ability to search through code using natural language queries.

## Architecture

### Core Components
- **FastMCP** (v2.14.1): SSE-based MCP protocol server running on port 8978
- **ChromaDB** (v1.4.0): Vector database for storing code embeddings
- **Sentence Transformers**: Jina embeddings model (`jinaai/jina-embeddings-v2-small-en`)
- **LlamaIndex**: Document loading and intelligent code splitting
- **Tree-sitter**: AST-based code parsing for semantic chunking
- **Watchdog**: Real-time file system monitoring

### Key Files
- `code_indexer_server.py`: Main MCP server implementation
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: Container build configuration
- `.env`: Environment variables (port, paths, exclusions)
- `requirements.txt`: Python dependencies with pinned versions
- `chroma_db/`: Persistent vector database storage
- `container_cache/`: Cached models and dependencies

## How It Works

### 1. Initialization Flow
1. Load configuration from environment variables
2. Initialize ChromaDB client with persistent storage
3. Load Jina embeddings model (384-dimensional vectors)
4. Auto-discover or use configured project folders
5. Start background indexing and file watching tasks
6. Launch MCP SSE server on port 8978

### 2. Indexing Process
1. **File Discovery**: Walk directory tree, filter by extensions and ignore patterns
2. **Smart Chunking**: Use Tree-sitter AST parsing for semantic code boundaries
   - Functions and classes kept intact when possible
   - 40 lines per chunk with 15-line overlap
   - Max 1500 characters per chunk
3. **Embedding Generation**: Convert chunks to vectors using Jina model
4. **Storage**: Store in ChromaDB with metadata (file path, line numbers, language)
5. **File Watching**: Monitor for changes and re-index automatically

### 3. Search Functionality
- **Tool**: `search_code` (exposed via MCP)
- **Parameters**: query, project, n_results (default 5), threshold (default 30.0)
- **Method**: Semantic similarity search using cosine distance
- **Returns**: Code snippets with file paths, line numbers, and relevance scores

## Configuration

### Environment Variables (.env)
```bash
FASTMCP_PORT=8978                          # MCP server port
PROJECTS_ROOT=~/git                        # Root directory containing projects
FOLDERS_TO_INDEX=                          # Comma-separated list (empty = auto-discover)
ADDITIONAL_IGNORE_DIRS=vendor,third_party  # Directories to exclude
ADDITIONAL_IGNORE_FILES=*.pb.go,*.min.js   # File patterns to exclude
DEBOUNCE_SECONDS=10.0                      # Debounce time for file change batching
```

### Default Exclusions
**Directories**: `__pycache__`, `node_modules`, `.git`, `build`, `dist`, `.venv`, `venv`, `env`, `.pytest_cache`, `.ipynb_checkpoints`

**Files**: `package-lock.json`, `yarn.lock`, `.env*`, `*.pyc`, `*.so`, `.DS_Store`, etc.

**Supported Languages**: Python, JavaScript/TypeScript, Java, C/C++, C#, Go, Ruby, PHP, Swift, Kotlin, Rust, Scala, HTML, CSS, SQL, Markdown, JSON, YAML

## Critical Configuration Rules

### Rule 1: Environment Variables Must Be in docker-compose.yml
**Problem**: Adding variables to `.env` alone doesn't work.
**Solution**: Every variable in `.env` must also be listed in `docker-compose.yml` environment section.

**Example**:
```yaml
environment:
  FASTMCP_PORT: ${FASTMCP_PORT}
  ADDITIONAL_IGNORE_DIRS: ${ADDITIONAL_IGNORE_DIRS}  # Must be explicitly listed!
```

### Rule 2: Always Exclude Vendor Directories
**Why**: Vendor/dependency directories contain thousands of files that slow indexing dramatically.
**Required exclusions**: `vendor`, `third_party`, `Godeps`, `_vendor`, `node_modules`, `external`

**Impact**:
- Without exclusions: Hours of indexing time
- With exclusions: 10-15 minutes for typical projects

### Rule 3: Restart After Configuration Changes
**Changes requiring restart**:
- Modifying `.env` file
- Changing embedding model
- Updating ignore patterns
- Modifying docker-compose.yml

**Commands**:
```bash
docker compose down
docker compose up -d
```

### Rule 4: Clear ChromaDB When Changing Exclusions
**When**: Adding/removing ignore patterns or changing folders to index
**Why**: Old data with different exclusion rules creates inconsistent index
**How**:
```bash
docker compose down
rm -rf chroma_db/
docker compose up -d
```

### Rule 5: Don't Follow Symlinks
**Security**: All file operations use `followlinks=False` to prevent directory traversal attacks
**Code**: See `load_documents()` function line 397 and 408

### Rule 6: Debouncing for Large Git Operations
**Why**: Git checkout/pull on large repos (e.g., Kubernetes) can trigger thousands of file change events
**Solution**: File changes are batched and deduplicated using the `DebouncedFileHandler` class
**Configuration**: Set `DEBOUNCE_SECONDS` in `.env` (default: 10.0 seconds)
**Behavior**:
- File changes are collected for the debounce period
- Same file changed multiple times = only processed once (latest state)
- Prevents overwhelming the indexer during git operations

## Docker Commands

### Basic Operations
```bash
# Start container (detached mode)
docker compose up -d

# Stop container (keeps container)
docker compose stop

# Stop and remove container
docker compose down

# Restart container
docker compose restart

# View logs (follow mode)
docker compose logs -f

# View last N lines
docker compose logs --tail=50

# Check container status
docker compose ps

# Check resource usage
docker stats code-indexer-mcp --no-stream
```

### Rebuild and Clean Start
```bash
# Rebuild after code changes
docker compose build

# Rebuild without cache (force fresh build)
docker compose build --no-cache

# Full clean restart
docker compose down
rm -rf chroma_db/
docker compose build
docker compose up -d
```

### Debugging
```bash
# Execute command in container
docker compose exec code-indexer-mcp bash

# Check environment variables
docker compose exec code-indexer-mcp env

# View Python processes
docker compose exec code-indexer-mcp ps aux
```

## Monitoring Indexing Progress

### Check if Indexing is Complete
Look for these log patterns:
```
INFO - Auto-discovered X folders: [list]
INFO - Starting to index folder: [name]
INFO - Loaded Y documents from /projects/[name]
INFO - Successfully indexed Z code chunks across Y files
INFO - Started watching [name]
INFO - File watching task started  ← ALL DONE!
```

### Monitor Progress
```bash
# Real-time progress
docker compose logs -f | grep -E "indexed|watching|Started"

# Count files processed
docker compose logs | grep -c "Processing /"

# Check CPU usage (high = still indexing)
docker stats code-indexer-mcp --no-stream
```

### Time Estimates
- Small project (100-500 files): 1-3 minutes
- Medium project (500-2000 files): 3-10 minutes
- Large project (2000+ files): 10-30 minutes
- **With vendor dirs**: Hours (DON'T DO THIS!)

## Common Issues and Solutions

### Issue 1: Slow Indexing (Processing vendor directories)
**Symptoms**: Logs show `Processing vendor/...`
**Cause**: Vendor directories not excluded
**Solution**:
1. Add to `.env`: `ADDITIONAL_IGNORE_DIRS=vendor,third_party,Godeps`
2. Add to `docker-compose.yml` environment section
3. Run: `docker compose down && rm -rf chroma_db/ && docker compose up -d`

### Issue 2: Configuration Changes Not Applied
**Symptoms**: Container ignores new `.env` settings
**Cause**: Environment variables not in docker-compose.yml
**Solution**: Add variables to both `.env` AND `docker-compose.yml` environment section, then restart

### Issue 3: Pydantic Warnings
**Symptoms**: `UnsupportedFieldAttributeWarning` messages
**Cause**: FastMCP 2.14.1 compatibility with Pydantic 2.12.5
**Impact**: Harmless, does not affect functionality
**Solution**: Ignore or suppress with warnings filter (not critical)

### Issue 4: Port 8978 Already in Use
**Symptoms**: Container fails to start with port conflict
**Cause**: Another instance running or stale container
**Solution**:
```bash
docker ps -a | grep code-indexer
docker rm -f code-indexer-mcp
docker compose up -d
```

### Issue 5: ChromaDB Collection Already Exists
**Symptoms**: Logs show "skipping initial indexing" but you want to re-index
**Cause**: Collection exists from previous run
**Solution**: Delete `chroma_db/` and restart

## MCP Integration

### Claude Code VSCode Extension
**Config file**: `/home/azureuser/.config/Code/User/mcp.json`

**Configuration**:
```json
{
  "mcpServers": {
    "local-code-indexer": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/home/azureuser/ai/local-indexing-mcp/docker-compose.yml",
        "exec",
        "-T",
        "code-indexer-mcp",
        "python",
        "code_indexer_server.py"
      ]
    }
  }
}
```

**Note**: Container must be running before MCP client connects.

### Claude Code Console (CLI)
**Config file**: `~/.config/claude-code/mcp.json`
**Configuration**: Same as VSCode extension

### Usage
Once configured, AI assistants can use the `search_code` tool:
```python
search_code(
    query="authentication logic",
    project="kubernetes",
    n_results=5,
    threshold=30.0
)
```

## Development Notes

### Changing Embedding Models
**Current**: `jinaai/jina-embeddings-v2-small-en` (optimized for code)
**Alternatives**:
- `all-MiniLM-L6-v2`: Faster, smaller (original default)
- `all-mpnet-base-v2`: Better quality, slower
- `sentence-transformers/gtr-t5-large`: Best quality, largest

**Important**: After changing models, MUST delete `chroma_db/` and re-index (embeddings are incompatible).

**Location**: `code_indexer_server.py` line 316 and 339

### Code Splitting Configuration
**Settings** (line 516-518):
```python
chunk_lines=40           # Lines per chunk
chunk_lines_overlap=15   # Overlap between chunks
max_chars=1500          # Max characters per chunk
```

Increase chunk size to reduce total chunks (faster indexing, less granular search).

### FastMCP API Notes
- Uses positional argument: `FastMCP("Server Name")` not `title=`
- Runs SSE server via `mcp.run_sse_async()`
- Tools defined with `@mcp.tool()` decorator

## Security Considerations

1. **Read-only mounts**: Projects mounted as `:ro` in docker-compose.yml
2. **No symlink following**: Prevents directory traversal attacks
3. **Containerized**: Isolated from host system
4. **No telemetry**: ChromaDB telemetry disabled
5. **Port binding**: Only exposes 8978, no web UI

## Performance Optimization

### Faster Indexing
1. Exclude vendor/dependency directories
2. Use faster embedding model (all-MiniLM-L6-v2)
3. Increase chunk size to reduce total chunks
4. Index only specific folders (not auto-discover)
5. Use SSD storage for chroma_db/

### Better Search Results
1. Use Jina embeddings (current default)
2. Adjust threshold parameter (lower = more results)
3. Increase n_results for broader search
4. Keep chunks smaller for precise line-number results

## Troubleshooting Checklist

Before asking for help, verify:
- [ ] Container is running: `docker compose ps`
- [ ] Port 8978 is listening: `netstat -tlnp | grep 8978`
- [ ] No errors in logs: `docker compose logs | grep -i error`
- [ ] Environment variables set: `docker compose exec code-indexer-mcp env`
- [ ] Vendor directories excluded: Check logs for `vendor/` paths
- [ ] ChromaDB exists: `ls -la chroma_db/`
- [ ] Sufficient disk space: `df -h`

## Future Enhancements (Ideas)

- [ ] Support for more languages (Elixir, Haskell, etc.)
- [ ] Web UI for browsing indexed code
- [ ] Multiple embedding models (quality vs speed)
- [ ] Incremental indexing (only changed files)
- [ ] Search result ranking improvements
- [ ] Multi-tenant support (user-specific indexes)
- [ ] API rate limiting
- [ ] Metrics and monitoring dashboard
- [ ] Integration with GitHub/GitLab webhooks

## Resources

- FastMCP: https://github.com/jlowin/fastmcp
- ChromaDB: https://docs.trychroma.com/
- MCP Protocol: https://modelcontextprotocol.io/
- Sentence Transformers: https://www.sbert.net/
- LlamaIndex: https://docs.llamaindex.ai/
- Tree-sitter: https://tree-sitter.github.io/

## Version History

- **v1.0** (Initial): Basic MCP server with ChromaDB
- **v1.1**: Added auto-discovery of folders
- **v1.2**: Switched to Jina embeddings model
- **v1.3**: Added vendor directory exclusions
- **v1.4**: Fixed environment variable passing in Docker
- **v1.5**: Fixed ChromaDB 1.4.0 `list_collections()` API compatibility
- **v1.6**: Added debounced file handler for batching and deduplication of file changes

Last updated: 2026-01-10
