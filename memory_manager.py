import os
import sys
# Force UTF-8 output for Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class QdrantMemory:
    def __init__(self, collection_name="phi3_memory", path="./qdrant_storage"):
        import os
        from qdrant_client.models import Distance, VectorParams
        
        try:
            # 1. Initialize Client (Local Storage Mode)
            self.client = QdrantClient(path=path)
        except Exception as e:
            if "already accessed" in str(e) or "Permission denied" in str(e):
                print("   [Memory] ⚠️ Storage lock detected. Attempting to clear lock...")
                lock_file = os.path.join(path, ".lock")
                if os.path.exists(lock_file):
                    try:
                        os.remove(lock_file)
                        print("   [Memory] 🔓 Lock removed. Retrying...")
                        self.client = QdrantClient(path=path)
                    except Exception as e2:
                        print(f"   [Memory] ❌ Could not clear lock: {e2}")
                        print("   [Memory] Falling back to In-Memory mode (Non-persistent).")
                        self.client = QdrantClient(location=":memory:")
                else:
                    self.client = QdrantClient(location=":memory:")
            else:
                raise e

        self.collection_name = collection_name
        
        # 2. Load Embedding Model (Lightweight & Fast)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Text Splitter (Chunks of 500 chars with 50 char overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # 4. Ensure Collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"📡 Creating new Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def add_memory(self, text, metadata=None):
        """
        Smart Memory: Checks if related context exists. If so, appends to it.
        Otherwise, creates new entry.
        """
        # 1. Check for existing related memory first
        # We use a threshold to decide if we should merge or create new
        SIMILARITY_THRESHOLD = 0.85 
        
        # Vectorize the input briefly to find the 'topic'
        query_vector = self.model.encode(text[:500]).tolist()
        
        existing_points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1
        ).points
        
        # 2. Logic: Merge or Create
        if existing_points and existing_points[0].score >= SIMILARITY_THRESHOLD:
            # MERGE STRATEGY
            best_match = existing_points[0]
            existing_content = best_match.payload['content']
            
            # Deduplication check (simple)
            if text in existing_content:
                print("   [Memory: Content already known. Skipping.]")
                return

            print(f"   [Memory: Found related topic (Score: {best_match.score:.2f}). Merging data...]")
            
            # Combine content
            new_content = existing_content + "\n\n---\n\n" + text
            
            # Re-vectorize the combined content (Note: this effectively updates the 'center' of the topic)
            new_vector = self.model.encode(new_content).tolist()
            
            # Update the point in Qdrant
            point = PointStruct(
                id=best_match.id,
                vector=new_vector,
                payload={
                    "content": new_content,
                    "metadata": best_match.payload.get('metadata', metadata or {})
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"   [Memory: Successfully updated single location for this topic.]")
            
        else:
            # NEW ENTRY STRATEGY 
            # Splits text into chunks, vectorizes them, and saves to Qdrant.
            chunks = self.splitter.split_text(text)
            points = []
            
            for i, chunk in enumerate(chunks):
                vector = self.model.encode(chunk).tolist()
                # Ensure unique ID for new chunks
                point_id = hash(chunk + str(metadata)) % (10**10) 
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "content": chunk,
                        "metadata": metadata or {}
                    }
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Saved {len(chunks)} new chunks to Qdrant memory.")

    def search(self, query, limit=3):
        """
        Finds the most relevant chunks in memory.
        """
        query_vector = self.model.encode(query).tolist()
        
        # In newer versions of qdrant-client local mode, search is handled via query_points
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points
        
        return [res.payload['content'] for res in results]

if __name__ == "__main__":
    # Test session
    memory = QdrantMemory()
    test_text = "The Phi-3 model is a lightweight, high-performance language model by Microsoft. It is designed to run on-device and can be fine-tuned for specific tasks like active learning."
    memory.add_memory(test_text)
    
    print("\n🔍 Searching Memory for 'What is Phi-3?'...")
    found = memory.search("What is Phi-3?")
    for i, chunk in enumerate(found):
        print(f"Chunk {i+1}: {chunk}")
