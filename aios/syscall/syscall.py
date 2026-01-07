import time
import json
import threading
import base64
import mimetypes
import os
import shutil
import hashlib
import requests
import csv
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

# --- AI & DB Imports ---
import chromadb
from sentence_transformers import SentenceTransformer

# Update import to use the new location
from aios.memory.note import MemoryNote
from aios.syscall import Syscall
from aios.syscall.llm import LLMSyscall
from aios.syscall.storage import StorageSyscall, storage_syscalls
from aios.syscall.tool import ToolSyscall
from aios.syscall.memory import MemorySyscall
from aios.hooks.stores._global import (
    global_llm_req_queue_add_message,
    global_memory_req_queue_add_message,
    global_storage_req_queue_add_message,
    global_tool_req_queue_add_message,
)

from aios.hooks.types.llm import LLMRequestQueue
from aios.hooks.types.memory import MemoryRequestQueue
from aios.hooks.types.storage import StorageRequestQueue
from aios.hooks.types.tool import ToolRequestQueue

from cerebrum.llm.apis import LLMQuery, LLMResponse
from cerebrum.memory.apis import MemoryQuery, MemoryResponse
from cerebrum.storage.apis import StorageQuery, StorageResponse
from cerebrum.tool.apis import ToolQuery, ToolResponse

# --- CONFIG ---
CATEGORIES = [
    "screenshots", "handwritten_pages", "food", "selfie", "people_group",
    "landscape", "indoor_room", "outdoor_place", "products", "clothes",
    "random_objects", "memes", "others"
]

SUPPORTED_IMAGE_EXT = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tiff", ".tif", ".webp", ".jfif", ".heic", ".avif"
}

# Database path relative to project root
DB_PATH = "root/images/chroma_db"

# --- RAG CONFIG ---
RAG_IMAGES_DIR = "root/images"
RAG_IMAGE_COLLECTION_NAME = "images_index"
RAG_CHUNK_COLLECTION_NAME = "chunks_index"
RAG_TOP_K_IMAGES = 5
RAG_TOP_K_CHUNKS = 5

# --- FORM PROCESSING CONFIG ---
FORM_INPUT_DIR = "root/form_images"
FORM_OUTPUT_CSV = "root/form_images/forms_output.csv"

CSV_FIELDS = [
    "file_name",
    "elector_name","elector_epic","elector_address",
    "serial_no","part_no","part_name","ac_name","pc_name","state",
    "date_of_birth","aadhaar_no","mobile_no",
    "father_name","father_epic","mother_name","mother_epic","spouse_name","spouse_epic",
    "sir_elector_name","sir_elector_epic","sir_relative_name","sir_relationship","sir_district","sir_state","sir_ac_name","sir_ac_number","sir_part_no","sir_sr_no",
    "sir_relative2_name","sir_relative2_epic","sir_relative2_relation","sir_relative2_district","sir_relative2_state","sir_relative2_ac_name","sir_relative2_ac_number","sir_relative2_part_no","sir_relative2_sr_no",
    "elector_signature_present","elector_signature_text","signature_date","blo_name","blo_contact","blo_signature_present",
    "raw_text"
]

# --- GLOBALS (Lazy Loading) ---
_chroma_client = None
_embed_model = None
_collection = None
_rag_images_collection = None
_rag_chunks_collection = None

def get_chroma_collection():
    """Lazy load ChromaDB collection for basic indexing."""
    global _chroma_client, _collection
    if _collection is None:
        print("Initializing ChromaDB...")
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _chroma_client.get_or_create_collection(
            name="image_vectors",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection

def get_rag_collections():
    """Lazy load ChromaDB collections for RAG system."""
    global _chroma_client, _rag_images_collection, _rag_chunks_collection
    if _rag_images_collection is None or _rag_chunks_collection is None:
        print("Initializing RAG ChromaDB collections...")
        if not os.path.exists(DB_PATH):
            os.makedirs(DB_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
        _rag_images_collection = _chroma_client.get_or_create_collection(
            name=RAG_IMAGE_COLLECTION_NAME
        )
        _rag_chunks_collection = _chroma_client.get_or_create_collection(
            name=RAG_CHUNK_COLLECTION_NAME
        )
    return _rag_images_collection, _rag_chunks_collection

def get_embed_model():
    """Lazy load Sentence Transformer model."""
    global _embed_model
    if _embed_model is None:
        print("Loading Embedding Model (MiniLM-L6-v2)...")
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model

def encode_image_to_base64(image_path):
    """Encodes a local image file into a Base64 data URI."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        ext = image_path.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg']: mime_type = 'image/jpeg'
        elif ext == 'png': mime_type = 'image/png'
        elif ext == 'webp': mime_type = 'image/webp'
        else: return None

    try:
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base64_encoded_data = base64.b64encode(binary_data)
            base64_string = base64_encoded_data.decode('utf-8')
            return f"data:{mime_type};base64,{base64_string}"
    except Exception:
        return None

def compute_md5(image_path: Path) -> str:
    """Compute MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def embed_text(text: str) -> List[float]:
    """Generate embedding vector for text."""
    embed_model = get_embed_model()
    return embed_model.encode(text).tolist()

def build_form_extraction_prompt():
    """Build the prompt for form extraction."""
    return """
You are an OCR + structured extraction engine specialized in Indian government enumeration forms.

TASK:
Extract ALL structured fields from the image.
Output ONLY ONE JSON OBJECT. No extra text, no explanations, no markdown, no backticks.

Return the JSON object with these fields (use null for missing/unreadable fields):

{
  "elector_name": null,
  "elector_epic": null,
  "elector_address": null,

  "serial_no": null,
  "part_no": null,
  "part_name": null,
  "ac_name": null,
  "pc_name": null,
  "state": null,

  "date_of_birth": null,
  "aadhaar_no": null,
  "mobile_no": null,

  "father_name": null,
  "father_epic": null,
  "mother_name": null,
  "mother_epic": null,
  "spouse_name": null,
  "spouse_epic": null,

  "sir_elector_name": null,
  "sir_elector_epic": null,
  "sir_relative_name": null,
  "sir_relationship": null,
  "sir_district": null,
  "sir_state": null,
  "sir_ac_name": null,
  "sir_ac_number": null,
  "sir_part_no": null,
  "sir_sr_no": null,

  "sir_relative2_name": null,
  "sir_relative2_epic": null,
  "sir_relative2_relation": null,
  "sir_relative2_district": null,
  "sir_relative2_state": null,
  "sir_relative2_ac_name": null,
  "sir_relative2_ac_number": null,
  "sir_relative2_part_no": null,
  "sir_relative2_sr_no": null,

  "elector_signature_present": null,
  "elector_signature_text": null,
  "signature_date": null,
  "blo_name": null,
  "blo_contact": null,
  "blo_signature_present": null,

  "raw_text": null
}

Rules:
- If a field is missing or unreadable, set it to null.
- Do NOT add any text outside the JSON object.
- Keep JSON syntactically valid.
"""

class SyscallExecutor:
    def __init__(self):
        self.id = 0
        self.id_lock = threading.Lock()
    
    def create_syscall(self, agent_name: str, query) -> Dict[str, Any]:
        if isinstance(query, LLMQuery):
            return LLMSyscall(agent_name, query)
        elif isinstance(query, StorageQuery):
            return StorageSyscall(agent_name, query)
        elif isinstance(query, MemoryQuery):
            return MemorySyscall(agent_name, query)
        elif isinstance(query, ToolQuery):
            return ToolSyscall(agent_name, query)

    def _execute_syscall(self, agent_name: str, query) -> Dict[str, Any]:
        completed_response = ""
        start_times, end_times = [], []
        waiting_times, turnaround_times = [], []

        with self.id_lock:
            self.id += 1
            syscall_id = self.id
        
        syscall = self.create_syscall(agent_name, query)
        syscall.set_status("active")
        
        current_time = time.time()
        syscall.set_created_time(current_time)
        syscall.set_response(None)

        if not syscall.get_source():
            syscall.set_source(syscall.agent_name)
        
        if not syscall.get_pid():
            syscall.set_pid(syscall_id)
        
        if isinstance(syscall, LLMSyscall):
            global_llm_req_queue_add_message(syscall)
        elif isinstance(syscall, StorageSyscall):
            global_storage_req_queue_add_message(syscall)
        elif isinstance(syscall, MemorySyscall):
            global_memory_req_queue_add_message(syscall)
        elif isinstance(syscall, ToolSyscall):
            global_tool_req_queue_add_message(syscall)
        
        syscall.start()
        syscall.join()
        
        completed_response = syscall.get_response()
        
        start_time = syscall.get_start_time()
        end_time = syscall.get_end_time()
        
        if start_time and end_time:
            waiting_time = start_time - syscall.get_created_time()
            turnaround_time = end_time - syscall.get_created_time()
            start_times.append(start_time)
            end_times.append(end_time)
            waiting_times.append(waiting_time)
            turnaround_times.append(turnaround_time)

        return {
            "response": completed_response,
            "start_times": start_times,
            "end_times": end_times,
            "waiting_times": waiting_times,
            "turnaround_times": turnaround_times,
        }

    def execute_storage_syscall(self, agent_name: str, query: StorageQuery) -> Dict[str, Any]:
        return self._execute_syscall(agent_name, query)

    def execute_memory_syscall(self, agent_name: str, query: MemoryQuery) -> Dict[str, Any]:
        return self._execute_syscall(agent_name, query)

    def execute_tool_syscall(self, agent_name: str, query: ToolQuery) -> Dict[str, Any]:
        return self._execute_syscall(agent_name, query)

    def execute_llm_syscall(self, agent_name: str, query: LLMQuery) -> Dict[str, Any]:
        return self._execute_syscall(agent_name, query)

    def execute_file_operation(self, agent_name: str, query: LLMQuery) -> str:
        system_prompt = (
            "You are a file system agent. Your task is to output a JSON tool call to execute the user's request.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY valid JSON. No conversational text.\n"
            "2. Do not use markdown code blocks.\n"
            "3. If you need to write code, put it inside the 'content' parameter.\n\n"
            "Example Format:\n"
            "[{\"name\": \"write_file\", \"arguments\": {\"path\": \"test.txt\", \"content\": \"Hello World\"}}]"
        )
        query.messages = [{"role": "system", "content": system_prompt}] + query.messages
        query.tools = storage_syscalls
        
        result = self.execute_llm_syscall(agent_name, query)
        parser_response = result["response"]
        
        if not parser_response or not getattr(parser_response, 'tool_calls', None):
            error_msg = parser_response.response_message if parser_response else "System Error."
            return f"Operation Failed: {error_msg}"

        file_operations = parser_response.tool_calls
        operation_summaries = []
        for operation in file_operations:
            op_name = operation.get("name")
            op_params = operation.get("arguments") if "arguments" in operation else operation.get("parameters")
            storage_query = StorageQuery(operation_type=op_name, params=op_params)
            storage_response = self.execute_storage_syscall(agent_name, storage_query)
            operation_summaries.append(str(storage_response))
        
        return "Operations completed."

    # --- FUNCTION 1: CLASSIFY IMAGES ---
    def execute_image_classification(self, agent_name: str) -> str:
        root_dir = "root"
        images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(images_dir): return "Error: 'images' folder not found."

        files = os.listdir(images_dir)
        images = [f for f in files if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f.lower())[1] in SUPPORTED_IMAGE_EXT]
        if not images: return "No images found."

        log = []
        for img_name in images:
            img_path = os.path.join(images_dir, img_name)
            base64_img = encode_image_to_base64(img_path)
            if not base64_img: continue

            prompt = f"Classify this image into ONE category: {CATEGORIES}. Return only the category name."
            query = LLMQuery(messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": base64_img}}]}], action_type="chat")
            
            try:
                result = self.execute_llm_syscall(agent_name, query)
                raw_cat = result["response"].response_message if hasattr(result["response"], "response_message") else "others"
                category = "others"
                for cat in CATEGORIES:
                    if cat in raw_cat.lower().replace("-", "_"):
                        category = cat
                        break
                
                target = os.path.join(images_dir, category)
                os.makedirs(target, exist_ok=True)
                shutil.move(img_path, os.path.join(target, img_name))
                log.append(f"Moved {img_name} -> {category}")
            except:
                pass
        return "\n".join(log)

    # --- FUNCTION 2: INDEX IMAGES (Auto-Caption + Vector DB) ---
    def execute_image_indexing(self, agent_name: str) -> str:
        """Scans root/images, captions them via LLM, embeds caption, stores in ChromaDB."""
        root_dir = "root"
        images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(images_dir): return "Error: 'images' folder not found."

        # Recursively find images even in subfolders
        image_files = []
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in SUPPORTED_IMAGE_EXT:
                    image_files.append(os.path.join(root, file))

        if not image_files: return "No images found to index."

        # Lazy load DB and Model
        collection = get_chroma_collection()
        embed_model = get_embed_model()
        log = []

        for img_path in image_files:
            try:
                # 1. Generate ID (MD5 of path)
                img_id = hashlib.md5(img_path.encode()).hexdigest()
                
                # Check if already indexed
                existing = collection.get(ids=[img_id])
                if existing and existing['ids']:
                    continue # Skip if already indexed

                # 2. Get Caption from LLM
                base64_img = encode_image_to_base64(img_path)
                prompt = "Describe this image in one short factual sentence. No markup."
                query = LLMQuery(
                    messages=[{
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt}, 
                            {"type": "image_url", "image_url": {"url": base64_img}}
                        ]
                    }], 
                    action_type="chat"
                )
                
                # Reuse existing syscall to call LLM
                result = self.execute_llm_syscall(agent_name, query)
                caption = result["response"].response_message if hasattr(result["response"], "response_message") else "An image."
                
                # 3. Embed Caption using SentenceTransformers
                embedding = embed_model.encode(caption).tolist()

                # 4. Store in Chroma
                collection.add(
                    ids=[img_id],
                    embeddings=[embedding],
                    metadatas=[{"path": img_path, "caption": caption}],
                    documents=[caption]
                )
                log.append(f"Indexed: {os.path.basename(img_path)} -> '{caption[:30]}...'")
            except Exception as e:
                log.append(f"Failed {os.path.basename(img_path)}: {str(e)}")

        return "Indexing Complete:\n" + "\n".join(log) if log else "All images were already indexed."

    # --- FUNCTION 3: SEARCH IMAGES (Semantic Retrieval) ---
    def execute_image_search(self, query_text: str) -> str:
        """Embeds query and searches ChromaDB."""
        try:
            collection = get_chroma_collection()
            embed_model = get_embed_model()
            
            # Embed user query
            query_vec = embed_model.encode(query_text).tolist()
            
            # Query DB
            results = collection.query(query_embeddings=[query_vec], n_results=3)
            
            if not results['ids'][0]:
                return "No matching images found."

            response_text = "Found these images:\n"
            for i in range(len(results['ids'][0])):
                path = results['metadatas'][0][i]['path']
                caption = results['metadatas'][0][i]['caption']
                filename = os.path.basename(path)
                response_text += f"{i+1}. **{filename}** (Caption: {caption})\n"
            
            return response_text
        except Exception as e:
            return f"Search Error: {str(e)}"

    # --- FUNCTION 4: PROCESS FORMS ---
    def execute_form_processing(self, agent_name: str) -> str:
        """Process all form images in root/form_images and extract data to CSV."""
        if not os.path.exists(FORM_INPUT_DIR):
            return f"Error: Form images folder not found at {FORM_INPUT_DIR}"

        # Get all image files
        files = sorted(os.listdir(FORM_INPUT_DIR))
        images = [f for f in files if os.path.splitext(f.lower())[1] in SUPPORTED_IMAGE_EXT]

        if not images:
            return "No form images found to process."

        # Prepare CSV
        write_header = not os.path.exists(FORM_OUTPUT_CSV)
        log = []
        
        try:
            with open(FORM_OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()

                total = len(images)
                for idx, fname in enumerate(images, start=1):
                    path = os.path.join(FORM_INPUT_DIR, fname)
                    log.append(f"[{idx}/{total}] Processing: {fname}")

                    # Extract form data
                    parsed, raw_text = self._extract_form_from_image(agent_name, path)

                    # Build row
                    row = {k: None for k in CSV_FIELDS}
                    row["file_name"] = fname

                    if parsed is None:
                        row["raw_text"] = raw_text if raw_text is not None else ""
                        writer.writerow(row)
                        log.append(f"  -> Failed to parse. Saved raw text.")
                        continue

                    # Fill row with parsed values
                    for key in CSV_FIELDS:
                        if key == "file_name":
                            continue
                        row[key] = parsed.get(key, None)

                    row["raw_text"] = parsed.get("raw_text", raw_text if isinstance(raw_text, str) else "")
                    writer.writerow(row)
                    log.append(f"  -> Successfully extracted and saved.")

            result = "\n".join(log)
            result += f"\n\nForm processing complete. Output saved to: {FORM_OUTPUT_CSV}"
            return result
        except Exception as e:
            return f"Error during form processing: {str(e)}\n\nPartial log:\n" + "\n".join(log)

    def _extract_form_from_image(self, agent_name: str, image_path: str, max_retries: int = 3):
        """Extract structured data from a single form image with retries."""
        encoded = encode_image_to_base64(image_path)
        if not encoded:
            return None, "Failed to encode image"

        prompt = build_form_extraction_prompt()
        
        attempt = 0
        backoff_base = 1.5
        
        while attempt < max_retries:
            try:
                attempt += 1
                
                # Create LLM query
                query = LLMQuery(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": encoded}}
                        ]
                    }],
                    action_type="chat"
                )
                
                # Execute via syscall
                result = self.execute_llm_syscall(agent_name, query)
                raw = result["response"].response_message if hasattr(result["response"], "response_message") else ""
                
                # Extract JSON from response
                match = re.search(r"\{[\s\S]*?\}", raw)
                if not match:
                    return None, raw
                
                json_text = match.group(0)
                parsed = json.loads(json_text)
                return parsed, raw
                
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                if attempt >= max_retries:
                    return None, f"ERROR: {repr(e)} — Raw response: {raw if 'raw' in locals() else 'N/A'}"
                
                sleep_time = backoff_base ** attempt
                print(f"Warning: attempt {attempt} failed for {os.path.basename(image_path)}; retrying after {sleep_time:.1f}s — {e}")
                time.sleep(sleep_time)

        return None, "Max retries exceeded"

    # --- RAG FUNCTIONS ---
    
    def _vlm_short_summary(self, agent_name: str, image_path: Path) -> str:
        """Generate coarse image summary (10-20 words)."""
        base64_img = encode_image_to_base64(str(image_path))
        if not base64_img:
            return "Unable to process image."
        
        prompt = (
            "Describe what this image is in 10 to 20 words. "
            "Focus only on the type and high-level meaning."
        )
        
        query = LLMQuery(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_img}}
                ]
            }],
            action_type="chat"
        )
        
        result = self.execute_llm_syscall(agent_name, query)
        return result["response"].response_message if hasattr(result["response"], "response_message") else "An image."

    def _vlm_factual_extraction(self, agent_name: str, image_path: Path) -> List[str]:
        """Extract factual information as clean, embeddable chunks."""
        base64_img = encode_image_to_base64(str(image_path))
        if not base64_img:
            return []
        
        prompt = (
            "Carefully analyze the image.\n\n"
            "If this image is a document:\n"
            "- Identify the document type.\n"
            "- List all clearly visible factual information.\n"
            "- Include subject names, marks, dates, identifiers, labels, values.\n\n"
            "If this image is NOT a document:\n"
            "- Describe objects, actions, and context factually.\n\n"
            "Output ONLY plain text.\n"
            "Each fact on a new line.\n"
            "No JSON. No markdown. No explanations."
        )
        
        query = LLMQuery(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_img}}
                ]
            }],
            action_type="chat"
        )
        
        result = self.execute_llm_syscall(agent_name, query)
        raw_text = result["response"].response_message if hasattr(result["response"], "response_message") else ""
        
        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]
        
        return lines

    def execute_rag_indexing(self, agent_name: str) -> str:
        """Main RAG indexing pipeline: processes all images and creates two-level index."""
        images_collection, chunks_collection = get_rag_collections()
        
        image_paths = list(Path(RAG_IMAGES_DIR).glob("**/*"))
        image_paths = [p for p in image_paths if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXT]

        if not image_paths:
            return "No images found for RAG indexing."

        log = []
        for img_path in image_paths:
            try:
                log.append(f"\nProcessing image: {img_path.name}")
                
                image_id = compute_md5(img_path)
                log.append(f"  image_id = {image_id}")

                # Check if already indexed
                existing = images_collection.get(ids=[image_id])
                if existing and existing['ids']:
                    log.append(f"  -> Already indexed, skipping.")
                    continue

                # Coarse image-level index
                summary = self._vlm_short_summary(agent_name, img_path)
                images_collection.add(
                    ids=[image_id],
                    documents=[summary],
                    embeddings=[embed_text(summary)],
                    metadatas=[{
                        "image_id": image_id,
                        "file_path": str(img_path)
                    }]
                )

                # Fine chunk-level index
                chunks = self._vlm_factual_extraction(agent_name, img_path)

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{image_id}_chunk_{idx}"
                    chunks_collection.add(
                        ids=[chunk_id],
                        documents=[chunk],
                        embeddings=[embed_text(chunk)],
                        metadatas=[{
                            "image_id": image_id,
                            "chunk_index": idx,
                            "file_path": str(img_path)
                        }]
                    )

                log.append(f"  -> Indexed {len(chunks)} chunks.")
            except Exception as e:
                log.append(f"  -> Error: {str(e)}")

        return "\n".join(log) + "\n\nRAG indexing completed."

    def _retrieve_candidate_images(self, query: str) -> List[str]:
        """Coarse search: retrieve top-K candidate image IDs."""
        images_collection, _ = get_rag_collections()
        query_embedding = embed_text(query)

        results = images_collection.query(
            query_embeddings=[query_embedding],
            n_results=RAG_TOP_K_IMAGES
        )

        image_ids = []
        for meta in results["metadatas"][0]:
            image_ids.append(meta["image_id"])

        return list(set(image_ids))

    def _retrieve_relevant_chunks(self, query: str, image_ids: List[str]) -> List[str]:
        """Fine search: retrieve top-K chunks from candidate images."""
        _, chunks_collection = get_rag_collections()
        query_embedding = embed_text(query)

        results = chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=RAG_TOP_K_CHUNKS,
            where={"image_id": {"$in": image_ids}}
        )

        return results["documents"][0] if results["documents"] else []

    def _generate_rag_answer(self, agent_name: str, query: str, context_chunks: List[str]) -> str:
        """Generate final answer using LLM with retrieved context."""
        context = "\n".join(context_chunks)

        prompt = (
            "You are given factual context extracted from images.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Answer the following question using ONLY the context above.\n"
            "If the answer is not present, say 'Information not found.'\n\n"
            f"Question: {query}"
        )

        llm_query = LLMQuery(
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            action_type="chat"
        )

        result = self.execute_llm_syscall(agent_name, llm_query)
        return result["response"].response_message if hasattr(result["response"], "response_message") else "Unable to generate answer."

    def execute_rag_query(self, agent_name: str, query_text: str) -> str:
        """Main RAG query pipeline: retrieval and answer generation."""
        try:
            log = []
            log.append(f"Query: {query_text}\n")

            # Step 1: Coarse image retrieval
            image_ids = self._retrieve_candidate_images(query_text)
            log.append(f"Candidate image_ids: {image_ids}")

            if not image_ids:
                return "\n".join(log) + "\n\nNo relevant images found."

            # Step 2: Fine chunk retrieval
            chunks = self._retrieve_relevant_chunks(query_text, image_ids)
            log.append(f"Retrieved {len(chunks)} chunks.")

            if not chunks:
                return "\n".join(log) + "\n\nNo relevant information found."

            # Step 3: Generate final answer
            answer = self._generate_rag_answer(agent_name, query_text, chunks)

            log.append(f"\nAnswer:\n{answer}")
            return "\n".join(log)
        except Exception as e:
            return f"RAG Query Error: {str(e)}"

    def execute_request(self, agent_name: str, query: Any) -> Dict[str, Any]:
        """
        Execute a request based on its type.
        """
        if isinstance(query, LLMQuery):
            try:
                last_msg = query.messages[-1]["content"] if query.messages else ""
                
                if isinstance(last_msg, str):
                    lower_msg = last_msg.lower()
                    
                    # COMMAND 1: "Classify images"
                    if "classify" in lower_msg and "image" in lower_msg:
                        return {"response": self.execute_image_classification(agent_name)}
                    
                    # COMMAND 2: "Index images"
                    if "index" in lower_msg and "image" in lower_msg:
                        return {"response": self.execute_image_indexing(agent_name)}
                    
                    # COMMAND 3: "Find image of..." / "Search for..."
                    if ("find" in lower_msg or "search" in lower_msg) and "image" in lower_msg and "rag" not in lower_msg:
                        # Extract search term (Simple heuristic)
                        search_term = last_msg.replace("find image", "").replace("search for image", "").replace("of", "").strip()
                        return {"response": self.execute_image_search(search_term)}
                    
                    # COMMAND 4: "Process forms"
                    if ("process" in lower_msg or "extract" in lower_msg or "scan" in lower_msg) and "form" in lower_msg:
                        return {"response": self.execute_form_processing(agent_name)}
                    
                    # COMMAND 5: "RAG index" - Two-level indexing
                    if "rag" in lower_msg and "index" in lower_msg:
                        return {"response": self.execute_rag_indexing(agent_name)}
                    
                    # COMMAND 6: "RAG query" or "RAG search" or "answer from images"
                    if ("rag" in lower_msg and ("query" in lower_msg or "search" in lower_msg or "answer" in lower_msg)) or \
                       ("answer" in lower_msg and "from" in lower_msg and "image" in lower_msg):
                        # Extract the actual question
                        query_text = last_msg
                        # Remove command keywords to get clean query
                        for keyword in ["rag query", "rag search", "rag answer", "answer from images", "answer from image"]:
                            query_text = query_text.replace(keyword, "").strip()
                        # Remove leading conjunctions/prepositions
                        for word in [":", "about", "for", "on"]:
                            if query_text.startswith(word):
                                query_text = query_text[len(word):].strip()
                        return {"response": self.execute_rag_query(agent_name, query_text)}

                    # NORMAL VISION: Describe specific image file
                    root_dir = "root" 
                    images_dir = os.path.join(root_dir, "images") 
                    if os.path.exists(images_dir):
                        for filename in os.listdir(images_dir):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                if filename in last_msg:
                                    image_path = os.path.join(images_dir, filename)
                                    base64_image = encode_image_to_base64(image_path)
                                    if base64_image:
                                        query.messages[-1]["content"] = [
                                            {"type": "text", "text": last_msg},
                                            {"type": "image_url", "image_url": {"url": base64_image}}
                                        ]
                                        query.action_type = "chat"
                                        break 
            except Exception as e:
                print(f"Vision Error: {e}")
                pass 

            if query.action_type == "chat" or query.action_type == "chat_with_json_response":
                return self.execute_llm_syscall(agent_name, query)
            elif query.action_type == "tool_use":
                llm_response = self.execute_llm_syscall(agent_name, query)["response"]
                if not llm_response or not llm_response.tool_calls:
                    return {"response": "Error: No tools."}
                tool_query = ToolQuery(tool_calls=llm_response.tool_calls)
                return self.execute_tool_syscall(agent_name, tool_query)
            elif query.action_type == "operate_file":
                return {"response": self.execute_file_operation(agent_name, query)}

        elif isinstance(query, ToolQuery):
            return self.execute_tool_syscall(agent_name, query)
        elif isinstance(query, MemoryQuery):
            return self.execute_memory_syscall(agent_name, query)
        elif isinstance(query, StorageQuery):
            return self.execute_storage_syscall(agent_name, query)

def create_syscall_executor():
    """Create and return a SyscallExecutor instance and its wrapper."""
    executor = SyscallExecutor()
    
    class SyscallWrapper:
        """Wrapper class providing direct access to syscall methods."""
        llm = executor.execute_llm_syscall
        storage = executor.execute_storage_syscall
        memory = executor.execute_memory_syscall
        tool = executor.execute_tool_syscall

    return executor.execute_request, SyscallWrapper

# Maintain backwards compatibility
useSysCall = create_syscall_executor