#!/usr/bin/env python3

import os
import logging
import json
import time
from threading import Lock
from typing import List, Set, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tree_sitter_language_pack import get_parser
import asyncio
from fastmcp import FastMCP
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Import LlamaIndex components
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core import SimpleDirectoryReader
    print("LlamaIndex dependencies found.")
except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required dependencies:")
    print("pip install llama-index llama-index-readers-file "
          "llama-index-embeddings-huggingface")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default directories to ignore
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    "node_modules",
    ".git",
    "build",
    "dist",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".ipynb_checkpoints"
}

# Default files to ignore
DEFAULT_IGNORE_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "Gemfile.lock",
    "composer.lock",
    ".DS_Store",
    ".env",
    ".env.local",
    ".env.development",
    ".env.test",
    ".env.production",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    ".coverage",
    "coverage.xml",
    ".eslintcache",
    ".tsbuildinfo"
}

# Default file extensions to include
DEFAULT_FILE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".scala", ".sh",
    ".html", ".css", ".sql", ".md", ".json", ".yaml", ".yml", ".toml"
}

# Global variables
config = None
chroma_client = None
embedding_function = None
mcp = FastMCP("Code Indexer Server")
observers = []
debounced_handler: Optional["DebouncedFileHandler"] = None


class DebouncedFileHandler:
    """
    Handles file changes with debouncing and deduplication.

    - Collects file changes for a configurable debounce period
    - Deduplicates changes (same file changed multiple times = one operation)
    - Processes changes in batches for efficiency
    - Prevents concurrent processing with a lock
    """

    def __init__(self, debounce_seconds: float = 5.0):
        self.pending_changes: Dict[str, Tuple[str, str, float]] = {}  # {filepath: (action, folder_name, timestamp)}
        self.lock = Lock()
        self.debounce_seconds = debounce_seconds
        self.processing_lock = asyncio.Lock()
        self.last_change_time: float = 0
        self._debounce_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        """Start the background worker."""
        self._running = True
        logger.info(f"DebouncedFileHandler started with {self.debounce_seconds}s debounce")

    def stop(self):
        """Stop the background worker."""
        self._running = False
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        logger.info("DebouncedFileHandler stopped")

    def add_change(self, file_path: str, action: str, folder_name: str):
        """
        Add a file change to the pending queue.
        Automatically deduplicates - latest change for each file wins.
        """
        with self.lock:
            self.pending_changes[file_path] = (action, folder_name, time.time())
            self.last_change_time = time.time()
            pending_count = len(self.pending_changes)

        logger.debug(f"Queued {action} for {file_path} ({pending_count} pending)")

        # Schedule debounce processing
        self._schedule_debounce()

    def _schedule_debounce(self):
        """Schedule the debounce timer to process changes."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cancel existing debounce task if any
                if self._debounce_task and not self._debounce_task.done():
                    self._debounce_task.cancel()
                # Schedule new debounce task
                self._debounce_task = loop.create_task(self._debounce_wait())
        except RuntimeError:
            # No event loop running yet, will be processed when loop starts
            pass

    async def _debounce_wait(self):
        """Wait for debounce period then trigger processing."""
        try:
            await asyncio.sleep(self.debounce_seconds)

            # Check if more changes came in during the wait
            with self.lock:
                time_since_last = time.time() - self.last_change_time

            if time_since_last >= self.debounce_seconds:
                # No new changes, process the batch
                await self._process_batch()
            else:
                # More changes came in, reschedule
                self._schedule_debounce()
        except asyncio.CancelledError:
            pass  # Task was cancelled, new one will be scheduled

    def get_and_clear_changes(self) -> Dict[str, Tuple[str, str, float]]:
        """Atomically get all pending changes and clear the buffer."""
        with self.lock:
            changes = self.pending_changes.copy()
            self.pending_changes.clear()
            return changes

    async def _process_batch(self):
        """Process all pending changes in a batch."""
        if not self._running:
            return

        async with self.processing_lock:
            changes = self.get_and_clear_changes()

            if not changes:
                return

            logger.info(f"Processing batch of {len(changes)} file changes")

            # Group changes by action type for logging
            modifications = []
            deletions = []

            for file_path, (action, folder_name, _) in changes.items():
                if action == "deleted":
                    deletions.append((file_path, folder_name))
                else:
                    modifications.append((file_path, folder_name))

            # Process deletions first
            for file_path, folder_name in deletions:
                try:
                    await self._handle_deletion(file_path, folder_name)
                except Exception as e:
                    logger.error(f"Error processing deletion for {file_path}: {e}")

            # Process modifications/creations
            for file_path, folder_name in modifications:
                try:
                    await self._handle_modification(file_path, folder_name)
                except Exception as e:
                    logger.error(f"Error processing modification for {file_path}: {e}")

            logger.info(f"Batch processing complete: {len(modifications)} modified, {len(deletions)} deleted")

    async def _handle_modification(self, file_path: str, folder_name: str):
        """Handle a file modification or creation."""
        try:
            # Check if file still exists (might have been deleted after the change was queued)
            if not os.path.exists(file_path):
                logger.debug(f"File no longer exists, skipping: {file_path}")
                return

            collection_name = sanitize_collection_name(folder_name)
            rel_path = os.path.relpath(file_path, config["projects_root"])

            # Load and process the single file
            reader = SimpleDirectoryReader(input_files=[file_path])
            documents = reader.load_data()

            if documents:
                documents[0].metadata["file_path"] = rel_path
                process_and_index_documents(
                    documents,
                    collection_name,
                    "chroma_db"
                )
                logger.info(f"Indexed: {rel_path}")
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")

    async def _handle_deletion(self, file_path: str, folder_name: str):
        """Handle a file deletion."""
        try:
            collection_name = sanitize_collection_name(folder_name)
            rel_path = os.path.relpath(file_path, config["projects_root"])

            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            collection.delete(where={"file_path": rel_path})
            logger.info(f"Removed from index: {rel_path}")
        except Exception as e:
            logger.error(f"Error removing {file_path} from index: {e}")


def sanitize_collection_name(folder_name: str) -> str:
    """Convert folder name to a valid collection name by replacing forward slashes with underscores."""
    return folder_name.replace("/", "_")


class CodeIndexerEventHandler(FileSystemEventHandler):
    """
    File system event handler that delegates to the debounced handler.

    Instead of processing files immediately, it queues changes to be
    processed in batches after a debounce period.
    """

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.collection_name = sanitize_collection_name(folder_name)
        self.ignore_dirs = set(config["ignore_dirs"])
        self.ignore_files = set(config["ignore_files"])
        self.file_extensions = set(config["file_extensions"])

    def on_created(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._queue_change(event.src_path, "created")

    def on_modified(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._queue_change(event.src_path, "modified")

    def on_deleted(self, event):
        if event.is_directory:
            return
        if is_valid_file(
            event.src_path,
            self.ignore_dirs,
            self.file_extensions,
            self.ignore_files
        ):
            self._queue_change(event.src_path, "deleted")

    def _queue_change(self, file_path: str, action: str):
        """Queue a file change for debounced processing."""
        global debounced_handler
        if debounced_handler:
            debounced_handler.add_change(file_path, action, self.folder_name)
        else:
            logger.warning(f"Debounced handler not initialized, skipping {action} for {file_path}")


def auto_discover_folders(projects_root: str, ignore_dirs: Set[str]) -> List[str]:
    """
    Auto-discover all first-level subdirectories in projects_root.

    Args:
        projects_root: Root directory to scan
        ignore_dirs: Set of directory names to ignore

    Returns:
        List of relative folder paths to index
    """
    discovered_folders = []

    try:
        if not os.path.exists(projects_root):
            logger.warning(f"Projects root does not exist: {projects_root}")
            return []

        # List all items in projects_root
        for item in os.listdir(projects_root):
            item_path = os.path.join(projects_root, item)

            # Skip if not a directory
            if not os.path.isdir(item_path):
                continue

            # Skip if it's a symlink
            if os.path.islink(item_path):
                logger.info(f"Skipping symlink: {item}")
                continue

            # Skip if in ignore list or starts with dot
            if item in ignore_dirs or item.startswith('.'):
                logger.info(f"Skipping ignored directory: {item}")
                continue

            discovered_folders.append(item)

        logger.info(f"Auto-discovered {len(discovered_folders)} folders: {discovered_folders}")
        return discovered_folders

    except Exception as e:
        logger.error(f"Error during auto-discovery: {e}")
        return []


def get_config_from_env():
    """
    Get configuration from environment variables.

    Supports two modes:
    1. Manual mode: Set FOLDERS_TO_INDEX with specific folders (comma-separated)
    2. Auto mode (default): When FOLDERS_TO_INDEX is empty, automatically discovers
       all first-level subdirectories in PROJECTS_ROOT
    """
    # Validate FASTMCP_PORT is set
    fastmcp_port = os.getenv("FASTMCP_PORT")
    if not fastmcp_port:
        logger.error("FASTMCP_PORT environment variable is not set")
        raise ValueError("FASTMCP_PORT environment variable is required")

    try:
        port_value = int(fastmcp_port)
        if not (1 <= port_value <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {port_value}")
    except ValueError as e:
        logger.error(f"Invalid FASTMCP_PORT value: {fastmcp_port}")
        raise ValueError(f"FASTMCP_PORT must be a valid port number: {e}")

    projects_root = os.getenv("PROJECTS_ROOT", "/projects")

    # Get manual folder configuration
    folders_to_index = os.getenv("FOLDERS_TO_INDEX", "").split(",")
    folders_to_index = [f.strip() for f in folders_to_index if f.strip()]

    # Get additional ignore dirs and files from environment
    additional_ignore_dirs = os.getenv(
        "ADDITIONAL_IGNORE_DIRS", ""
    ).split(",")
    additional_ignore_dirs = [
        d.strip() for d in additional_ignore_dirs if d.strip()
    ]

    additional_ignore_files = os.getenv(
        "ADDITIONAL_IGNORE_FILES", ""
    ).split(",")
    additional_ignore_files = [
        f.strip() for f in additional_ignore_files if f.strip()
    ]

    # Get debounce configuration (seconds to wait before processing file changes)
    debounce_seconds_str = os.getenv("DEBOUNCE_SECONDS", "5.0")
    try:
        debounce_seconds = float(debounce_seconds_str)
        if debounce_seconds < 0.5:
            logger.warning(f"DEBOUNCE_SECONDS too low ({debounce_seconds}), using minimum 0.5")
            debounce_seconds = 0.5
        elif debounce_seconds > 60:
            logger.warning(f"DEBOUNCE_SECONDS too high ({debounce_seconds}), using maximum 60")
            debounce_seconds = 60
    except ValueError:
        logger.warning(f"Invalid DEBOUNCE_SECONDS value '{debounce_seconds_str}', using default 5.0")
        debounce_seconds = 5.0

    # Combine default and additional ignore patterns
    ignore_dirs = list(DEFAULT_IGNORE_DIRS | set(additional_ignore_dirs))
    ignore_files = list(DEFAULT_IGNORE_FILES | set(additional_ignore_files))

    # Determine which folders to index
    if folders_to_index:
        # Manual configuration provided
        logger.info(f"Using manually configured folders: {folders_to_index}")
    else:
        # Auto-discover folders (default behavior)
        logger.info("FOLDERS_TO_INDEX is empty, auto-discovering folders...")
        folders_to_index = auto_discover_folders(projects_root, set(ignore_dirs))
        if not folders_to_index:
            logger.warning("No folders discovered. Using root directory.")
            folders_to_index = [""]

    return {
        "projects_root": projects_root,
        "folders_to_index": folders_to_index,
        "ignore_dirs": ignore_dirs,
        "ignore_files": ignore_files,
        "file_extensions": list(DEFAULT_FILE_EXTENSIONS),
        "debounce_seconds": debounce_seconds
    }


async def initialize_chromadb():
    """Initialize ChromaDB and embedding function asynchronously."""
    global config, chroma_client, embedding_function

    try:
        # Get configuration from environment
        config = get_config_from_env()
        logger.info("Configuration loaded successfully")

        # Initialize ChromaDB client with telemetry disabled
        chroma_client = chromadb.PersistentClient(
            path="chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        logger.info("ChromaDB client initialized")

        # Initialize embedding function
        embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # Faster model for quicker indexing
            )
        )
        logger.info("Embedding function initialized")

        return True
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        # Still need to assign default values to global variables
        if config is None:
            config = {"projects_root": "", "folders_to_index": [""]}
        if chroma_client is None:
            # Create an empty client as a fallback
            try:
                chroma_client = chromadb.PersistentClient(path="chroma_db")
            except Exception as db_err:
                logger.error(
                    f"Failed to create fallback ChromaDB client: {db_err}"
                )
        if embedding_function is None:
            try:
                embedding_function = (
                    embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"  # Faster model for quicker indexing
                    )
                )
            except Exception as embed_err:
                logger.error(
                    f"Failed to create fallback embedding function: "
                    f"{embed_err}"
                )
        return False


def is_valid_file(
    file_path: str,
    ignore_dirs: Set[str],
    file_extensions: Set[str],
    ignore_files: Set[str] = None
) -> bool:
    """Check if a file should be processed based on its path and extension."""
    # Check if path contains ignored directory
    parts = file_path.split(os.path.sep)
    for part in parts:
        if part in ignore_dirs:
            return False

    # Get file name and check against ignored files
    file_name = os.path.basename(file_path)

    # Use provided ignore_files or fall back to default
    files_to_ignore = ignore_files if ignore_files is not None else DEFAULT_IGNORE_FILES

    # Check exact matches
    if file_name in files_to_ignore:
        return False

    # Check wildcard patterns
    for pattern in files_to_ignore:
        if pattern.startswith("*"):
            if file_name.endswith(pattern[1:]):
                return False

    # Check file extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in file_extensions if file_extensions else True


def load_documents(
    directory: str, 
    ignore_dirs: Set[str] = DEFAULT_IGNORE_DIRS,
    file_extensions: Set[str] = DEFAULT_FILE_EXTENSIONS,
    ignore_files: Set[str] = None
) -> List[Document]:
    """
    Load documents from a directory, filtering out ignored paths.
    Uses os.walk with followlinks=False to avoid following symbolic links.
    """
    try:
        # Get all files recursively - DO NOT follow symlinks
        all_files = []
        for root, dirs, files in os.walk(directory, followlinks=False):
            # Skip ignored directories
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.startswith('.')
            ]

            for file in files:
                abs_file_path = os.path.join(root, file)

                # Skip symlinks completely to avoid issues
                if os.path.islink(abs_file_path):
                    continue

                if is_valid_file(
                    abs_file_path,
                    ignore_dirs,
                    file_extensions,
                    ignore_files
                ):
                    # Calculate relative path from the directory being indexed
                    rel_file_path = os.path.relpath(abs_file_path, directory)
                    all_files.append((abs_file_path, rel_file_path))

        if not all_files:
            logger.warning(f"No valid files found in {directory}")
            return []

        # Load the filtered files using absolute paths for reading
        reader = SimpleDirectoryReader(
            input_files=[abs_path for abs_path, _ in all_files],
            exclude_hidden=True
        )
        documents = reader.load_data()

        # Update the metadata to use relative paths
        for doc, (_, rel_path) in zip(documents, all_files):
            doc.metadata["file_path"] = rel_path

        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def process_and_index_documents(
    documents: List[Document],
    collection_name: str,
    persist_directory: str
) -> None:
    """Process documents with CodeSplitter and index them in ChromaDB."""
    if not documents:
        logger.warning("No documents to process.")
        return

    try:
        # Try to get collection if it exists or create a new one
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return

    # Process each document
    total_nodes = 0
    for doc in documents:
        try:
            # Extract file path from metadata
            file_path = doc.metadata.get("file_path", "unknown")
            file_name = os.path.basename(file_path)

            # Determine language from file extension
            _, ext = os.path.splitext(file_name)
            language = ext[1:] if ext else "text"  # Remove the dot

            # Handle Markdown and other text files differently
            code_file_extensions = [
                "py", "python", "js", "jsx", "ts", "tsx", "java", "c", 
                "cpp", "h", "hpp", "cs", "go", "rb", "php", "swift", 
                "kt", "rs", "scala"
            ]

            if language in code_file_extensions:
                # Determine parser language based on file extension
                parser_language = "python"  # Default fallback
                if language in ["py", "python"]:
                    parser_language = "python"
                elif language in ["js", "jsx", "ts", "tsx"]:
                    parser_language = "javascript"
                elif language in ["java"]:
                    parser_language = "java"
                elif language in ["c", "cpp", "h", "hpp"]:
                    parser_language = "cpp"
                elif language in ["cs"]:
                    parser_language = "csharp"
                elif language in ["go"]:
                    parser_language = "go"
                elif language in ["rb"]:
                    parser_language = "ruby"
                elif language in ["php"]:
                    parser_language = "php"
                elif language in ["swift"]:
                    parser_language = "swift"
                elif language in ["kt"]:
                    parser_language = "kotlin"
                elif language in ["rs"]:
                    parser_language = "rust"
                elif language in ["scala"]:
                    parser_language = "scala"

                # Create parser and splitter for this specific language
                try:
                    code_parser = get_parser(parser_language)
                    splitter = CodeSplitter(
                        language=parser_language,
                        chunk_lines=40,
                        chunk_lines_overlap=15,
                        max_chars=1500,
                        parser=code_parser
                    )
                    nodes = splitter.get_nodes_from_documents([doc])
                except Exception as e:
                    logger.warning(
                        f"Could not create parser for {parser_language}, "
                        f"falling back to text-based splitting: {e}"
                    )
                    # Fall back to text-based splitting
                    nodes = []
                    lines = doc.text.split("\n")
                    chunk_size = 40
                    overlap = 15

                    for i in range(0, len(lines), chunk_size - overlap):
                        start_idx = i
                        end_idx = min(i + chunk_size, len(lines))

                        if start_idx >= len(lines):
                            continue

                        chunk_text = "\n".join(lines[start_idx:end_idx])

                        if not chunk_text.strip():
                            continue

                        from llama_index.core.schema import TextNode
                        node = TextNode(
                            text=chunk_text,
                            metadata={
                                "start_line_number": start_idx + 1,
                                "end_line_number": end_idx,
                                "file_path": file_path,
                                "file_name": file_name,
                            }
                        )
                        nodes.append(node)
            else:
                # For non-code files, manually split by lines
                nodes = []
                lines = doc.text.split("\n")
                chunk_size = 40
                overlap = 15

                for i in range(0, len(lines), chunk_size - overlap):
                    start_idx = i
                    end_idx = min(i + chunk_size, len(lines))

                    if start_idx >= len(lines):
                        continue

                    chunk_text = "\n".join(lines[start_idx:end_idx])

                    if not chunk_text.strip():
                        continue

                    from llama_index.core.schema import TextNode
                    node = TextNode(
                        text=chunk_text,
                        metadata={
                            "start_line_number": start_idx + 1,
                            "end_line_number": end_idx,
                            "file_path": file_path,
                            "file_name": file_name,
                        }
                    )
                    nodes.append(node)

            if not nodes:
                logger.warning(f"No nodes generated for {file_path}")
                continue

            logger.info(f"Processing {file_path}: {len(nodes)} chunks")

            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []

            for i, node in enumerate(nodes):
                start_line = node.metadata.get("start_line_number", 0)
                end_line = node.metadata.get("end_line_number", 0)

                if start_line == 0 or end_line == 0:
                    start_line = 1
                    end_line = len(node.text.split("\n"))

                chunk_id = f"{file_path}_{start_line}_{end_line}_{i}"

                metadata = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "language": language,
                    "start_line": start_line,
                    "end_line": end_line,
                }

                ids.append(chunk_id)
                texts.append(node.text)
                metadatas.append(metadata)

            # Add nodes to ChromaDB collection
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            total_nodes += len(nodes)

        except Exception as e:
            logger.error(
                f"Error processing document "
                f"{doc.metadata.get('file_path', 'unknown')}: {e}"
            )

    logger.info(
        f"Successfully indexed {total_nodes} code chunks "
        f"across {len(documents)} files"
    )


async def perform_initial_indexing(folder: str) -> bool:
    """Check if collection exists and perform initial indexing if needed."""
    try:
        folder_path = os.path.join(config["projects_root"], folder)
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return False

        collection_name = sanitize_collection_name(folder)

        # Check if collection exists
        try:
            chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            logger.info(f"Collection {collection_name} already exists, skipping initial indexing")
            return True
        except Exception:
            logger.info(f"Collection {collection_name} not found, performing initial indexing")

        # Load and process all documents in the folder
        documents = load_documents(
            folder_path,
            ignore_dirs=set(config["ignore_dirs"]),
            file_extensions=set(config["file_extensions"]),
            ignore_files=set(config["ignore_files"])
        )

        if documents:
            process_and_index_documents(documents, collection_name, "chroma_db")
            logger.info(f"Successfully performed initial indexing for {folder}")
            return True
        else:
            logger.warning(f"No documents found to index in {folder}")
            return False

    except Exception as e:
        logger.error(f"Error during initial indexing of {folder}: {e}")
        return False


async def index_projects():
    """Set up file system watchers for all configured projects."""
    global observers

    try:
        for folder in config["folders_to_index"]:
            # First perform initial indexing if needed
            success = await perform_initial_indexing(folder)
            if not success:
                logger.error(f"Failed to perform initial indexing for {folder}")
                continue

            folder_path = os.path.join(config["projects_root"], folder)
            logger.info(f"Setting up file watcher for {folder}")

            # Create an observer and event handler for this folder
            observer = Observer()
            event_handler = CodeIndexerEventHandler(folder)
            observer.schedule(event_handler, folder_path, recursive=True)

            # Start the observer
            observer.start()
            observers.append(observer)
            logger.info(f"Started watching {folder}")

        # Keep the main thread alive
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logger.error(f"Error in file watching setup: {e}")
        # Clean up observers on error
        for observer in observers:
            observer.stop()
        observers.clear()


@mcp.tool(
    name="search_code",
    description="""Search code using natural language queries.
        Args:
            query: Natural language query about the codebase
            project: Collection/folder name to search in. Use the current workspace name.
            n_results: Number of results to return (default: 5)
            threshold: Minimum relevance percentage to include results 
                (default: 30.0)
    """
)
async def search_code(
    query: str,
    project: str,
    n_results: int = 5,
    threshold: float = 30.0
) -> str:
    try:
        if not chroma_client or not embedding_function:
            logger.error("ChromaDB client or embedding function not initialized")
            return json.dumps({
                "error": "Search system not properly initialized",
                "results": [],
                "total_results": 0
            })

        # Get all collections
        collections = chroma_client.list_collections()

        # Find matching collections
        matching_collections = []
        project_name = project.lower()
        for collection in collections:
            # The collection name might be in format "customerX_project1" or just "project1"
            # We want to match if project_name fully matches the part after the last _ (if any)
            # Handle both old API (strings) and new API (Collection objects)
            collection_name = collection.name if hasattr(collection, 'name') else str(collection)
            collection_parts = collection_name.lower().split('_')
            if collection_parts[-1] == project_name:
                matching_collections.append(collection_name)

        if not matching_collections:
            logger.error(f"No collections found matching project {project}")
            return json.dumps({
                "error": f"No collections found matching project {project}",
                "results": [],
                "total_results": 0
            })

        # Search in all matching collections and combine results
        all_results = []

        for collection_name in matching_collections:
            collection = chroma_client.get_collection(collection_name)

            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    similarity = (1 - distance) * 100
                    if similarity >= threshold:
                        all_results.append({
                            "text": doc,
                            "file_path": meta.get("file_path", "Unknown file"),
                            "language": meta.get("language", "text"),
                            "start_line": int(meta.get("start_line", 0)),
                            "end_line": int(meta.get("end_line", 0)),
                            "relevance": round(similarity, 1),
                            "collection": collection.name  # Add collection name for debugging
                        })

        # Sort results by relevance
        all_results.sort(key=lambda x: x["relevance"], reverse=True)

        # Take top n_results
        final_results = all_results[:n_results]

        return json.dumps({
            "results": final_results,
            "total_results": len(final_results)
        })

    except Exception as e:
        logger.error(f"Error in search_code: {str(e)}")
        return json.dumps({
            "error": str(e),
            "results": [],
            "total_results": 0
        })


# Run initialization before starting MCP
async def main():
    global debounced_handler

    # Initialize ChromaDB before starting MCP
    success = await initialize_chromadb()

    if success:
        # Initialize the debounced file handler
        debounce_seconds = config.get("debounce_seconds", 5.0)
        debounced_handler = DebouncedFileHandler(debounce_seconds=debounce_seconds)
        debounced_handler.start()
        logger.info(f"Debounced file handler initialized ({debounce_seconds}s debounce)")

        # Start file watching in background
        asyncio.create_task(index_projects())
        logger.info("File watching task started")

    await mcp.run_async(transport="sse")

if __name__ == "__main__":
    asyncio.run(main())
