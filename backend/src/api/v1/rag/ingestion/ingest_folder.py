"""
Folder Ingestion Endpoint.

Triggers asynchronous ingestion of a local folder into the RAG system.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.core.ingest_config import ChunkConfig
from src.services.docling_service import docling_service
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
import asyncio
from pathlib import Path
from src.services.ingestion_task_manager import ingestion_task_manager, TaskStatus, FileResult
from datetime import datetime
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)
router = APIRouter()


def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested metadata for ChromaDB compatibility.

    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    This function converts nested dicts and lists to JSON strings.
    """
    flattened = {}
    for key, value in metadata.items():
        if value is None:
            flattened[key] = None
        elif isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (dict, list)):
            # Convert nested structures to JSON string
            import json
            flattened[key] = json.dumps(value)
        else:
            # Convert other types to string
            flattened[key] = str(value)
    return flattened


class IngestFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute path to the folder to ingest")
    collection_name: str = Field(..., description="Name of the target collection")
    profile: str = Field("default", description="Ingestion profile (codebase, documents, default)")
    recursive: bool = Field(True, description="Whether to scan recursively")
    allowed_extensions: Optional[List[str]] = Field(None, description="Specific extensions to include")
    default_chunking_strategy: Optional[str] = Field("sentence", description="Chunking strategy to use")
    incremental: bool = Field(True, description="Skip unchanged files")
    clean_sync: bool = Field(False, description="Remove documents from DB if file is deleted from folder")
    target_file: Optional[str] = Field(None, description="Specific file name to target inside the folder")

async def _process_files_background(
    task_id: str,
    files_to_ingest: List[str],
    request: IngestFolderRequest,
    rag_client
):
    """Background task to process files via generic IngestionProcessor and handle sync options."""
    from src.api.v1.rag.models.ingestion import ProcessOptions, ChunkingStrategy
    from src.services.ingestion_task_manager import ingestion_task_manager, TaskStatus
    import traceback
    
    processor = ingestion_task_manager.get_processor()
    task = ingestion_task_manager.get_task(task_id)

    try:
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        # 1. CLEAN SYNC: Remove documents not in the folder anymore
        if request.clean_sync:
            logger.info(f"Task {task_id}: Performing clean sync for collection {request.collection_name}")
            try:
                # get_collection will throw error if not exists, which is fine, means no clean sync needed
                actual_client = rag_client.chroma_manager.chroma_client.get_client() if hasattr(rag_client.chroma_manager.chroma_client, 'get_client') else rag_client.chroma_manager.chroma_client
                collection = await asyncio.to_thread(actual_client.get_collection, request.collection_name)
                
                # We need all documents in this collection
                all_data = await asyncio.to_thread(collection.get, include=["metadatas"])
                ids_to_delete = []
                valid_filenames = {os.path.basename(f) for f in files_to_ingest}
                
                for doc_id, metadata in zip(all_data["ids"], all_data["metadatas"]):
                    if doc_id == "__collection_metadata__":
                        continue
                    
                    # Usually "source" or "file_name" holds the filename
                    source_name = metadata.get("source") or metadata.get("file_name") or ""
                    # For chunks, the id might be filename_chunkindex
                    # Just check if source_name is still valid
                    if source_name and source_name not in valid_filenames:
                        ids_to_delete.append(doc_id)
                
                if ids_to_delete:
                    await asyncio.to_thread(collection.delete, ids=ids_to_delete)
                    logger.info(f"Clean Sync: Removed {len(ids_to_delete)} chunks/documents belonging to deleted files.")
            except Exception as e:
                logger.warning(f"Skipping clean sync (collection might not exist yet or error): {e}")

        # 2. Prepare assignments
        assignments = []
        chunking_strategy = ChunkingStrategy(request.default_chunking_strategy) if request.default_chunking_strategy else ChunkingStrategy.SENTENCE
        # If incremental is False, we force re-ingest (skip hash check)
        process_options = ProcessOptions(
            chunking_strategy=chunking_strategy,
            force_reingest=not request.incremental
        )
        
        for file_path in files_to_ingest:
            assignments.append({
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "collection": request.collection_name,
                "process_options": process_options
            })
            
        task.assignments = assignments
        
        # 3. Call processor
        # It handles duplicate skipping natively if ExtractionService with DuplicateDetector is used
        await processor.process_task_async(task_id, rag_client)
        
    except Exception as e:
        logger.error(f"Background task failed: {e}\n{traceback.format_exc()}")
        if task:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.now()


@router.post("/ingest-folder")
async def ingest_folder_endpoint(
    request: IngestFolderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    rag_client=Depends(get_rag_client)
):
    """
    Asynchronous ingestion of a local folder with real-time progress tracking.

    1. Scans the folder for matching files.
    2. Creates a background task to process them.
    3. Returns task_id for progress polling.
    """
    logger.info(f"Received folder ingestion request: {request.folder_path} -> {request.collection_name}")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")

    # 1. Determine extensions based on profile
    extensions = request.allowed_extensions
    if not extensions:
        if request.profile == "codebase":
            extensions = [".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".yml", ".yaml", ".html", ".css", ".sql"]
        elif request.profile == "documents":
            extensions = [".pdf", ".docx", ".txt", ".md"]
        else:
            # Default: broad set
            extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".js"]

    extensions = [ext.lower() for ext in extensions]
    logger.info(f"Using extensions filter: {extensions}")

    # 2. Scan for files
    files_to_ingest = []
    for root, dirs, filenames in os.walk(request.folder_path):
        # Skip common ignore dirs
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build"}]

        for filename in filenames:
            # Filter by target_file if specified
            if request.target_file and filename != request.target_file:
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            # Only filter by extension list (includes both documents and code files)
            if ext in extensions:
                full_path = os.path.join(root, filename)
                files_to_ingest.append(full_path)

        if not request.recursive:
            break

    if not files_to_ingest:
        raise HTTPException(status_code=400, detail="No matching files found in the specified folder.")

    logger.info(f"Found {len(files_to_ingest)} files to ingest.")

    # 3. Create background task
    task_id = ingestion_task_manager.create_task(
        file_count=len(files_to_ingest),
        collection_name=request.collection_name,
        user_id=current_user.id if hasattr(current_user, 'id') else None
    )

    # 4. Start background processing
    background_tasks.add_task(
        _process_files_background,
        task_id,
        files_to_ingest,
        request,
        rag_client
    )

    logger.info(f"Created task {task_id} for {len(files_to_ingest)} files")

    return {
        "task_id": task_id,
        "status": "processing",
        "files_found": len(files_to_ingest),
        "collection": request.collection_name,
        "message": f"Ingestion started. Poll /api/v1/rag/ingestion/ingest-status/{task_id} for progress."
    }
