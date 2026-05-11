"""
OpenSearch Ingestion Script for Workflow Documents
Indexes workflow documents with IBM watsonx embeddings into OpenSearch
"""

import json
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

load_dotenv()

# Configuration
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "workflow_documents")
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true"
OPENSEARCH_VERIFY_CERTS = os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true"

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-large")

BATCH_SIZE = 20
RATE_LIMIT_DELAY = 0.5  # seconds between batches


def get_opensearch_client() -> OpenSearch:
    """Create and return OpenSearch client"""
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST.replace("https://", "").replace("http://", ""), "port": OPENSEARCH_PORT}],
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=OPENSEARCH_USE_SSL,
        verify_certs=OPENSEARCH_VERIFY_CERTS,
        connection_class=RequestsHttpConnection,
        timeout=30
    )
    return client


def create_index(client: OpenSearch, index_name: str):
    """Create OpenSearch index with KNN vector support"""
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16
                        }
                    }
                },
                "page_content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "metadata": {
                    "properties": {
                        "type": {"type": "keyword"},
                        "title": {"type": "text"},
                        "category": {"type": "keyword"},
                        "steps": {"type": "object", "enabled": True},
                        "estimated_duration": {"type": "keyword"},
                        "required_permissions": {"type": "keyword"},
                        "tags": {"type": "keyword"},
                        "created_date": {"type": "date"},
                        "last_updated": {"type": "date"}
                    }
                }
            }
        }
    }
    
    # Delete index if it exists
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)
    
    # Create new index
    print(f"Creating index: {index_name}")
    client.indices.create(index=index_name, body=index_body)
    print(f"Index {index_name} created successfully")


def get_embedding_model():
    """Initialize IBM watsonx embedding model"""
    credentials = {
        "url": WATSONX_URL,
        "apikey": WATSONX_API_KEY
    }
    
    embedding = Embeddings(
        model_id=EMBEDDING_MODEL_ID,
        params={
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {"input_text": False}
        },
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID
    )
    
    return embedding


def format_workflow_text(workflow: Dict[str, Any]) -> str:
    """Format workflow document as text for embedding"""
    text_parts = [
        f"Workflow: {workflow['title']}",
        f"Type: {workflow['type']}",
        f"Category: {workflow['category']}",
        f"Description: {workflow['description']}",
        f"Estimated Duration: {workflow['estimated_duration']}",
        "\nSteps:"
    ]
    
    for step in workflow['steps']:
        step_text = f"\nStep {step['step_number']}: {step['description']}"
        if step['required_inputs']:
            step_text += f"\n  Required inputs: {', '.join(step['required_inputs'])}"
        if step['dependencies']:
            step_text += f"\n  Dependencies: Steps {', '.join(map(str, step['dependencies']))}"
        step_text += f"\n  Completion criteria: {step['completion_criteria']}"
        text_parts.append(step_text)
    
    return "\n".join(text_parts)


def get_embeddings(texts: List[str], embedding_model) -> List[List[float]]:
    """Generate embeddings for a batch of texts"""
    try:
        embeddings = embedding_model.embed_documents(texts=texts)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise


def prepare_document(workflow: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    """Prepare document for OpenSearch ingestion"""
    return {
        "_id": workflow["id"],
        "page_content": format_workflow_text(workflow),
        "embedding": embedding,
        "metadata": {
            "type": workflow["type"],
            "title": workflow["title"],
            "category": workflow["category"],
            "steps": workflow["steps"],
            "estimated_duration": workflow["estimated_duration"],
            "required_permissions": workflow["required_permissions"],
            "tags": workflow["tags"],
            "created_date": workflow["created_date"],
            "last_updated": workflow["last_updated"]
        }
    }


def ingest_workflows(
    client: OpenSearch,
    index_name: str,
    workflows: List[Dict[str, Any]],
    embedding_model
):
    """Ingest workflows into OpenSearch with embeddings"""
    total = len(workflows)
    print(f"\nIngesting {total} workflows into {index_name}...")
    
    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = workflows[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        # Generate text representations
        texts = [format_workflow_text(wf) for wf in batch]
        
        # Generate embeddings
        try:
            embeddings = get_embeddings(texts, embedding_model)
        except Exception as e:
            print(f"Failed to generate embeddings for batch {batch_num}: {e}")
            continue
        
        # Prepare documents
        documents = [
            prepare_document(wf, emb)
            for wf, emb in zip(batch, embeddings)
        ]
        
        # Bulk index
        bulk_body = []
        for doc in documents:
            bulk_body.append({"index": {"_index": index_name, "_id": doc["_id"]}})
            bulk_body.append({
                "page_content": doc["page_content"],
                "embedding": doc["embedding"],
                "metadata": doc["metadata"]
            })
        
        try:
            response = client.bulk(body=bulk_body)
            if response.get("errors"):
                print(f"  Warning: Some documents in batch {batch_num} had errors")
            else:
                print(f"  ✓ Batch {batch_num} indexed successfully")
        except Exception as e:
            print(f"  ✗ Failed to index batch {batch_num}: {e}")
        
        # Rate limiting
        if i + BATCH_SIZE < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    # Refresh index
    client.indices.refresh(index=index_name)
    
    # Get document count
    count = client.count(index=index_name)["count"]
    print(f"\n✓ Ingestion complete! {count} documents in index {index_name}")


def main():
    """Main ingestion workflow"""
    print("=" * 60)
    print("Workflow Document Ingestion")
    print("=" * 60)
    
    # Load workflow data
    data_file = "../data/workflow_dataset.json"
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        print("Please run data_generator.py first to generate workflow data")
        return
    
    print(f"\nLoading workflows from {data_file}...")
    with open(data_file, 'r') as f:
        workflows = json.load(f)
    print(f"Loaded {len(workflows)} workflows")
    
    # Initialize OpenSearch client
    print("\nConnecting to OpenSearch...")
    client = get_opensearch_client()
    
    # Test connection
    try:
        info = client.info()
        print(f"Connected to OpenSearch {info['version']['number']}")
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        return
    
    # Create index
    create_index(client, OPENSEARCH_INDEX)
    
    # Initialize embedding model
    print("\nInitializing IBM watsonx embedding model...")
    try:
        embedding_model = get_embedding_model()
        print(f"Using model: {EMBEDDING_MODEL_ID}")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return
    
    # Ingest workflows
    ingest_workflows(client, OPENSEARCH_INDEX, workflows, embedding_model)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Made with Bob
