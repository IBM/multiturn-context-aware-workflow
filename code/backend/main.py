"""
FastAPI Backend for Multiturn Workflow Agent Demo
Provides endpoints for Standard RAG and Workflow Agent queries,
plus on-demand data population and session management.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import uuid
import asyncio
from dotenv import load_dotenv
import time
from datetime import datetime

from rag_pipelines import StandardRAG, WorkflowAgent, MetricsCollector

load_dotenv()

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Multiturn Workflow Agent API",
    description="Compare Standard RAG vs Workflow Agent for operational workflows",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Global instances
# ─────────────────────────────────────────────
standard_rag = None
workflow_agent = None
metrics_collector = MetricsCollector()

# Populate job state — tracks background ingestion progress
populate_state = {
    "status": "idle",       # idle | running | done | error
    "message": "",
    "progress": 0,          # 0-100
    "total_docs": 0,
    "ingested_docs": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


# ─────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question or instruction")
    session_id: Optional[str] = Field(None, description="Session ID for workflow agent")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    retrieval_time: float
    generation_time: float
    total_time: float
    num_sources: int


class WorkflowStatus(BaseModel):
    workflow_id: Optional[str] = None
    workflow_title: Optional[str] = None
    total_steps: Optional[int] = None
    current_step: Optional[int] = None
    completed_steps: List[int] = []
    estimated_duration: Optional[str] = None


class WorkflowResponse(BaseModel):
    answer: str
    status: str
    workflow_status: Optional[WorkflowStatus] = None
    sources: Optional[List[SourceDocument]] = None
    retrieval_time: float
    generation_time: float
    total_time: float
    session_id: str
    next_action: Optional[str] = None


class MetricsResponse(BaseModel):
    total_queries: int
    standard_rag_queries: int
    workflow_agent_queries: int
    avg_standard_retrieval_time: float
    avg_workflow_retrieval_time: float
    avg_standard_sources: float
    avg_workflow_sources: float


class HealthResponse(BaseModel):
    status: str
    message: str
    opensearch_connected: bool
    embedding_provider: str


class PopulateStatusResponse(BaseModel):
    status: str
    message: str
    progress: int
    total_docs: int
    ingested_docs: int
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]


class DataStatusResponse(BaseModel):
    populated: bool
    document_count: int
    index_name: str
    message: str


class SessionStateResponse(BaseModel):
    session_id: str
    has_active_workflow: bool
    workflow_title: Optional[str] = None
    current_step: Optional[int] = None
    completed_steps: List[int] = []
    conversation_length: int


# ─────────────────────────────────────────────
# Startup/Shutdown
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipelines on startup"""
    global standard_rag, workflow_agent

    print("\n" + "="*60)
    print("STARTUP CONNECTION CHECKS")
    print("="*60)
    
    # Test OpenSearch connection
    print("\n1. Testing OpenSearch connection...")
    try:
        from opensearchpy import OpenSearch, RequestsHttpConnection
        opensearch_client = OpenSearch(
            hosts=[{
                'host': os.getenv("OPENSEARCH_HOST", "localhost").replace("https://", "").replace("http://", ""),
                'port': int(os.getenv("OPENSEARCH_PORT", "9200"))
            }],
            http_auth=(os.getenv("OPENSEARCH_USERNAME", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin")),
            use_ssl=os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true",
            verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", "true").lower() == "true",
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=10
        )
        info = opensearch_client.info()
        print(f"   ✅ OpenSearch connected successfully!")
        print(f"   - Cluster: {info.get('cluster_name', 'N/A')}")
        print(f"   - Version: {info.get('version', {}).get('number', 'N/A')}")
        print(f"   - Host: {os.getenv('OPENSEARCH_HOST')}")
    except Exception as e:
        print(f"   ❌ OpenSearch connection FAILED: {e}")
        print(f"   - Host: {os.getenv('OPENSEARCH_HOST')}")
        print(f"   - Port: {os.getenv('OPENSEARCH_PORT')}")
        print(f"   - Username: {os.getenv('OPENSEARCH_USERNAME')}")
    
    # Test Watsonx connection
    print("\n2. Testing IBM Watsonx connection...")
    try:
        from ibm_watsonx_ai.foundation_models import ModelInference
        credentials = {
            "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            "apikey": os.getenv("WATSONX_API_KEY")
        }
        test_model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params={"max_new_tokens": 10},
            credentials=credentials,
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        # Try a simple generation to test connection
        result = test_model.generate_text(prompt="Hello")
        print(f"   ✅ Watsonx connected successfully!")
        print(f"   - URL: {os.getenv('WATSONX_URL')}")
        print(f"   - Project ID: {os.getenv('WATSONX_PROJECT_ID')}")
        print(f"   - Model: ibm/granite-3-8b-instruct")
    except Exception as e:
        print(f"   ❌ Watsonx connection FAILED: {e}")
        print(f"   - URL: {os.getenv('WATSONX_URL')}")
        print(f"   - Project ID: {os.getenv('WATSONX_PROJECT_ID')}")
    
    print("\n3. Initializing RAG pipelines...")
    try:
        standard_rag = StandardRAG()
        workflow_agent = WorkflowAgent()
        print("   ✅ RAG pipelines initialized successfully")
    except Exception as e:
        print(f"   ❌ Error initializing RAG pipelines: {e}")
        raise
    
    print("\n" + "="*60)
    print("STARTUP COMPLETE")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down...")


# ─────────────────────────────────────────────
# Health Endpoints
# ─────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    opensearch_connected = False
    try:
        if standard_rag and standard_rag.client:
            info = standard_rag.client.info()
            opensearch_connected = True
    except:
        pass

    return HealthResponse(
        status="healthy",
        message="Multiturn Workflow Agent API is running",
        opensearch_connected=opensearch_connected,
        embedding_provider=os.getenv("EMBEDDING_MODEL_ID", "watsonx")
    )


@app.get("/health")
async def health():
    """Lightweight health check for Docker/nginx healthcheck probes"""
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Data Status Endpoint
# ─────────────────────────────────────────────
@app.get("/data-status", response_model=DataStatusResponse)
async def data_status():
    """
    Check whether the OpenSearch index has been populated.
    The UI calls this on load to decide whether to show the Populate screen
    or the query interface.
    """
    if not standard_rag or not standard_rag.client:
        return DataStatusResponse(
            populated=False,
            document_count=0,
            index_name=os.getenv("OPENSEARCH_INDEX", "workflow_documents"),
            message="RAG pipeline not initialized"
        )

    try:
        # Check if index exists
        index_name = os.getenv("OPENSEARCH_INDEX", "workflow_documents")
        if not standard_rag.client.indices.exists(index=index_name):
            return DataStatusResponse(
                populated=False,
                document_count=0,
                index_name=index_name,
                message="Index does not exist — click Populate Data to generate and ingest workflow documents"
            )

        # Get document count
        count_result = standard_rag.client.count(index=index_name)
        count = count_result.get("count", 0)

        if count == 0:
            return DataStatusResponse(
                populated=False,
                document_count=0,
                index_name=index_name,
                message="Index is empty — click Populate Data to generate and ingest workflow documents"
            )

        return DataStatusResponse(
            populated=True,
            document_count=count,
            index_name=index_name,
            message=f"Index has {count} documents — ready to query"
        )
    except Exception as e:
        return DataStatusResponse(
            populated=False,
            document_count=0,
            index_name=os.getenv("OPENSEARCH_INDEX", "workflow_documents"),
            message=f"Could not check index: {str(e)}"
        )


# ─────────────────────────────────────────────
# Populate Endpoint — background ingestion
# ─────────────────────────────────────────────
def _run_populate():
    """
    Synchronous worker that generates workflow data in-memory and ingests it
    into OpenSearch.
    """
    global populate_state

    from dataclasses import asdict
    from data_generator import WorkflowDataGenerator
    from ingest import (
        get_opensearch_client,
        create_index,
        get_embedding_model,
        format_workflow_text,
        get_embeddings,
        prepare_document
    )

    try:
        populate_state["status"] = "running"
        populate_state["message"] = "Initializing data generator..."
        populate_state["progress"] = 2
        populate_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        print("\n" + "="*60)
        print("STARTING DATA POPULATION")
        print("="*60)

        # ── Step 1: Generate workflow data ──
        print("\n1. Generating workflow data...")
        generator = WorkflowDataGenerator()

        populate_state["message"] = "Generating operational procedures (300)..."
        populate_state["progress"] = 5
        print("   - Generating operational procedures (300)...")
        generator.generate_operational_procedures(300)

        populate_state["message"] = "Generating task plans (250)..."
        populate_state["progress"] = 15
        print("   - Generating task plans (250)...")
        generator.generate_task_plans(250)

        populate_state["message"] = "Generating troubleshooting guides (200)..."
        populate_state["progress"] = 25
        print("   - Generating troubleshooting guides (200)...")
        generator.generate_troubleshooting_guides(200)

        populate_state["message"] = "Generating configuration procedures (150)..."
        populate_state["progress"] = 35
        print("   - Generating configuration procedures (150)...")
        generator.generate_configuration_procedures(150)

        populate_state["message"] = "Generating inspection checklists (100)..."
        populate_state["progress"] = 45
        print("   - Generating inspection checklists (100)...")
        generator.generate_inspection_checklists(100)

        # ── Step 2: Prepare documents ──
        populate_state["message"] = "Preparing documents for ingestion..."
        populate_state["progress"] = 50
        print("\n2. Preparing documents for ingestion...")

        workflows = [asdict(wf) for wf in generator.workflows]
        total = len(workflows)
        populate_state["total_docs"] = total
        print(f"   Total documents: {total}")

        # ── Step 3: Connect to OpenSearch ──
        populate_state["message"] = "Connecting to OpenSearch..."
        populate_state["progress"] = 55
        print("\n3. Connecting to OpenSearch...")
        client = get_opensearch_client()
        print("   ✅ Connected successfully")

        # Create index
        index_name = os.getenv("OPENSEARCH_INDEX", "workflow_documents")
        print(f"\n4. Creating index: {index_name}")
        create_index(client, index_name)
        print("   ✅ Index created")

        # ── Step 4: Initialize embedding model ──
        populate_state["message"] = "Initializing embedding model..."
        populate_state["progress"] = 60
        print("\n5. Initializing embedding model...")
        embedding_model = get_embedding_model()
        print("   ✅ Embedding model ready")

        # ── Step 5: Ingest in batches ──
        populate_state["message"] = "Starting ingestion..."
        populate_state["progress"] = 65
        print("\n6. Starting ingestion...")

        BATCH_SIZE = 20
        ingested = 0
        batches = [workflows[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches, 1):
            texts = [format_workflow_text(wf) for wf in batch]
            embeddings = get_embeddings(texts, embedding_model)

            bulk_body = []
            for wf, emb in zip(batch, embeddings):
                doc = prepare_document(wf, emb)
                bulk_body.append({"index": {"_index": index_name, "_id": doc["_id"]}})
                bulk_body.append({
                    "page_content": doc["page_content"],
                    "embedding": doc["embedding"],
                    "metadata": doc["metadata"]
                })

            try:
                client.bulk(body=bulk_body)
            except Exception as e:
                err_str = str(e).lower()
                if "already exists" not in err_str and "duplicate" not in err_str:
                    raise

            ingested += len(batch)
            populate_state["ingested_docs"] = ingested
            # Progress: 65% → 95% during ingestion
            populate_state["progress"] = 65 + int((ingested / total) * 30)
            populate_state["message"] = (
                f"Ingesting batch {batch_idx}/{total_batches} "
                f"({ingested}/{total} documents)..."
            )
            time.sleep(0.3)

        # Refresh index
        print("\n7. Refreshing index...")
        client.indices.refresh(index=index_name)
        print("   ✅ Index refreshed")

        populate_state["status"] = "done"
        populate_state["progress"] = 100
        populate_state["ingested_docs"] = ingested
        populate_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        populate_state["message"] = (
            f"Population complete — {ingested} documents ingested into OpenSearch"
        )
        
        print("\n" + "="*60)
        print(f"✅ POPULATION COMPLETE — {ingested} documents ingested")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ POPULATION FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback:\n{traceback.format_exc()}")
        
        populate_state["status"] = "error"
        populate_state["error"] = str(e)
        populate_state["message"] = f"Population failed: {str(e)}"
        populate_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@app.post("/populate", response_model=PopulateStatusResponse)
async def populate_data(background_tasks: BackgroundTasks):
    """
    Trigger on-demand data generation and OpenSearch ingestion.
    Generates 1,000 synthetic workflow documents in-memory,
    embeds them with IBM watsonx, and inserts them into OpenSearch.
    Returns immediately — poll GET /populate/status for progress.
    """
    global populate_state

    if populate_state["status"] == "running":
        return PopulateStatusResponse(**populate_state)

    # Reset state for a fresh run
    populate_state = {
        "status": "running",
        "message": "Starting data generation...",
        "progress": 0,
        "total_docs": 0,
        "ingested_docs": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at": None,
        "error": None,
    }

    # Run in thread pool
    background_tasks.add_task(_run_populate)

    return PopulateStatusResponse(**populate_state)


@app.get("/populate/status", response_model=PopulateStatusResponse)
async def populate_status():
    """
    Poll the current status of the background populate job.
    Frontend polls this every 2 seconds while status == 'running'.
    """
    return PopulateStatusResponse(**populate_state)


# ─────────────────────────────────────────────
# Query Endpoints
# ─────────────────────────────────────────────
@app.post("/rag", response_model=QueryResponse)
async def query_standard_rag(request: QueryRequest):
    """
    Query using Standard RAG (vector similarity only)
    """
    if not standard_rag:
        raise HTTPException(status_code=503, detail="Standard RAG not initialized")

    try:
        start_time = time.time()

        # Retrieve documents
        retrieval_start = time.time()
        sources = standard_rag.retrieve(request.question, top_k=request.top_k or 5)
        retrieval_time = time.time() - retrieval_start

        # Generate answer
        generation_start = time.time()
        answer = standard_rag.generate_answer(request.question, sources)
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Convert sources to response format
        source_docs = [
            SourceDocument(
                id=src.get("id", ""),
                content=src.get("page_content", ""),
                metadata=src.get("metadata", {}),
                score=src.get("score")
            )
            for src in sources
        ]

        # Track metrics
        metrics_collector.record_query(
            query_type="standard",
            retrieval_time=retrieval_time,
            num_sources=len(sources)
        )

        return QueryResponse(
            answer=answer,
            sources=source_docs,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            num_sources=len(sources)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/workflow-agent", response_model=WorkflowResponse)
async def query_workflow_agent(request: QueryRequest):
    """
    Query using Workflow Agent (conversational + orchestration)
    """
    if not workflow_agent:
        raise HTTPException(status_code=503, detail="Workflow Agent not initialized")

    try:
        start_time = time.time()

        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())

        # Process instruction
        retrieval_start = time.time()
        result = workflow_agent.process_instruction(request.question, session_id)
        retrieval_time = time.time() - retrieval_start

        # Generate response
        generation_start = time.time()
        
        # Extract answer based on result status
        if result["status"] == "query_answered":
            answer = result["answer"]
            sources = result.get("sources", [])
        elif result["status"] in ["workflow_started", "step_completed", "workflow_complete"]:
            answer = result["message"]
            sources = []
        elif result["status"] == "awaiting_input":
            answer = result["message"]
            sources = []
        else:
            answer = result.get("message", "Processing...")
            sources = []
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time

        # Build workflow status
        session = workflow_agent.get_session(session_id)
        workflow_status = None
        if session["current_workflow"]:
            workflow_status = WorkflowStatus(
                workflow_id=session["current_workflow"]["id"],
                workflow_title=session["current_workflow"]["metadata"]["title"],
                total_steps=len(session["current_workflow"]["metadata"]["steps"]),
                current_step=session["current_step"] + 1,
                completed_steps=session["completed_steps"],
                estimated_duration=session["current_workflow"]["metadata"]["estimated_duration"]
            )

        # Convert sources
        source_docs = None
        if sources:
            source_docs = [
                SourceDocument(
                    id=src.get("id", ""),
                    content=src.get("page_content", ""),
                    metadata=src.get("metadata", {}),
                    score=src.get("score")
                )
                for src in sources
            ]

        # Track metrics
        metrics_collector.record_query(
            query_type="workflow",
            retrieval_time=retrieval_time,
            num_sources=len(sources) if sources else 0
        )

        return WorkflowResponse(
            answer=answer,
            status=result["status"],
            workflow_status=workflow_status,
            sources=source_docs,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            session_id=session_id,
            next_action=result.get("next_step_description")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ─────────────────────────────────────────────
# Session Management Endpoints
# ─────────────────────────────────────────────
@app.get("/session/{session_id}/state", response_model=SessionStateResponse)
async def get_session_state(session_id: str):
    """Get current session state"""
    if not workflow_agent:
        raise HTTPException(status_code=503, detail="Workflow Agent not initialized")

    session = workflow_agent.get_session(session_id)

    return SessionStateResponse(
        session_id=session_id,
        has_active_workflow=session["current_workflow"] is not None,
        workflow_title=session["current_workflow"]["metadata"]["title"] if session["current_workflow"] else None,
        current_step=session["current_step"] + 1 if session["current_workflow"] else None,
        completed_steps=session["completed_steps"],
        conversation_length=len(session["conversation_history"])
    )


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset session state"""
    if not workflow_agent:
        raise HTTPException(status_code=503, detail="Workflow Agent not initialized")

    workflow_agent.reset_session(session_id)
    return {"message": f"Session {session_id} reset successfully"}


# ─────────────────────────────────────────────
# Metrics Endpoints
# ─────────────────────────────────────────────
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get aggregated metrics for both RAG approaches
    """
    metrics = metrics_collector.get_metrics()

    return MetricsResponse(
        total_queries=metrics["total_queries"],
        standard_rag_queries=metrics["standard_rag_queries"],
        workflow_agent_queries=metrics["workflow_agent_queries"],
        avg_standard_retrieval_time=metrics["avg_standard_retrieval_time"],
        avg_workflow_retrieval_time=metrics["avg_workflow_retrieval_time"],
        avg_standard_sources=metrics["avg_standard_sources"],
        avg_workflow_sources=metrics["avg_workflow_sources"]
    )


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset metrics"""
    metrics_collector.reset()
    return {"message": "Metrics reset successfully"}


# ─────────────────────────────────────────────
# Run with: uvicorn main:app --reload --port 8000
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Made with Bob
