"""
RAG Pipelines for Multiturn Workflow Agent
Implements Standard RAG and Workflow Agent with conversational state management
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai import APIClient

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
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "ibm/granite-3-8b-instruct")


class BaseRAG:
    """Base class for RAG implementations"""
    
    def __init__(self):
        self.client = self._init_opensearch()
        self.index_name = OPENSEARCH_INDEX
        self.embedding_model = self._init_embedding_model()
        self.llm = self._init_llm()
    
    def _init_opensearch(self) -> OpenSearch:
        """Initialize OpenSearch client"""
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST.replace("https://", "").replace("http://", ""), "port": OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            use_ssl=OPENSEARCH_USE_SSL,
            verify_certs=OPENSEARCH_VERIFY_CERTS,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        return client
    
    def _init_embedding_model(self):
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
    
    def _init_llm(self):
        """Initialize IBM watsonx LLM"""
        credentials = {
            "url": WATSONX_URL,
            "apikey": WATSONX_API_KEY
        }
        
        llm = ModelInference(
            model_id=LLM_MODEL_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 1000,
                GenParams.MIN_NEW_TOKENS: 1,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 1
            },
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID
        )
        
        return llm
    
    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        embeddings = self.embedding_model.embed_documents(texts=[query])
        return embeddings[0]
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            content = doc.get("page_content", "")
            context_parts.append(f"[Source {idx}]\n{content}\n")
        return "\n".join(context_parts)


class StandardRAG(BaseRAG):
    """Standard RAG implementation - vector search only"""
    
    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents using vector similarity search"""
        # Embed the query
        query_embedding = self._embed_query(question)
        
        # KNN search in OpenSearch
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["page_content", "metadata"]
        }
        
        results = self.client.search(
            index=self.index_name,
            body=search_body
        )
        
        documents = []
        for hit in results["hits"]["hits"]:
            doc = hit["_source"]
            doc["id"] = hit["_id"]
            doc["score"] = hit["_score"]
            documents.append(doc)
        
        return documents
    
    def generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM"""
        context = self._format_context(documents)
        
        prompt = f"""You are an operational assistant helping users with workflow procedures and maintenance tasks.

Answer the question using ONLY the information provided in the context below. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.generate_text(prompt=prompt)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"


class WorkflowAgent(BaseRAG):
    """Workflow Agent with conversational state management and backend orchestration"""
    
    def __init__(self):
        super().__init__()
        # Session storage (in production, use Redis or database)
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session state"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "conversation_history": [],
                "current_workflow": None,
                "current_step": 0,
                "completed_steps": [],
                "workflow_state": {},
                "collected_inputs": {},
                "created_at": datetime.now().isoformat()
            }
        return self.sessions[session_id]
    
    def reset_session(self, session_id: str):
        """Reset session state"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def retrieve_workflows(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant workflows using vector search"""
        query_embedding = self._embed_query(query)
        
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["page_content", "metadata"]
        }
        
        results = self.client.search(
            index=self.index_name,
            body=search_body
        )
        
        workflows = []
        for hit in results["hits"]["hits"]:
            workflow = hit["_source"]
            workflow["id"] = hit["_id"]
            workflow["score"] = hit["_score"]
            workflows.append(workflow)
        
        return workflows
    
    def _classify_intent(self, user_input: str, current_workflow: Optional[Dict]) -> str:
        """Classify user intent"""
        lower_input = user_input.lower()
        
        # Check for workflow start keywords
        start_keywords = ["start", "begin", "initiate", "execute", "run", "perform"]
        if any(keyword in lower_input for keyword in start_keywords):
            return "start_workflow"
        
        # Check for reset keywords (check before continuation to allow explicit resets)
        reset_keywords = ["reset", "restart", "cancel", "stop", "abort"]
        if any(keyword in lower_input for keyword in reset_keywords):
            return "reset_workflow"
        
        # If there's an active workflow, check for continuation intent
        if current_workflow:
            # Explicit continuation keywords
            continue_keywords = ["next", "continue", "proceed", "yes", "done", "complete"]
            if any(keyword in lower_input for keyword in continue_keywords):
                return "continue_workflow"
            
            # Contextual questions that imply workflow continuation
            contextual_keywords = [
                "what did you find", "what's the result", "what happened",
                "show me", "tell me", "what's the status", "status",
                "analyze", "check", "apply", "fix", "execute"
            ]
            if any(keyword in lower_input for keyword in contextual_keywords):
                return "continue_workflow"
            
            # Short responses that likely refer to the workflow context
            # (e.g., "it", "that", "this", single word responses)
            if len(lower_input.split()) <= 3:
                pronoun_keywords = ["it", "that", "this", "them", "those"]
                if any(keyword in lower_input for keyword in pronoun_keywords):
                    return "continue_workflow"
        
        # Default to query
        return "query"
    
    def _extract_inputs(self, user_input: str, required_inputs: List[str]) -> Dict[str, str]:
        """Extract required inputs from user message"""
        import re
        inputs = {}
        lower_input = user_input.lower()
        
        for input_name in required_inputs:
            # Try multiple extraction strategies
            
            # Strategy 1: Look for "input_name value" or "input_name: value"
            pattern1 = rf"{input_name}[\s:]+([a-zA-Z0-9\-_]+)"
            match = re.search(pattern1, lower_input)
            if match:
                inputs[input_name] = match.group(1)
                continue
            
            # Strategy 2: Look for "for server X" or "server X"
            if "server" in input_name:
                # Match patterns like "server api-prod-01" or "for server web-01"
                pattern2 = r"(?:for\s+)?server\s+([a-zA-Z0-9\-_]+)"
                match = re.search(pattern2, lower_input)
                if match:
                    inputs[input_name] = match.group(1)
                    continue
            
            # Strategy 3: Look for "asset X" or "for asset X"
            if "asset" in input_name:
                pattern3 = r"(?:for\s+)?asset\s+(?:is\s+)?([a-zA-Z0-9\-_]+)"
                match = re.search(pattern3, lower_input)
                if match:
                    inputs[input_name] = match.group(1)
                    continue
            
            # Strategy 4: Look for action-related inputs (fix_action, action, etc.)
            if "action" in input_name or "fix" in input_name:
                # Match patterns like "apply the fix", "restart service", "yes"
                action_patterns = [
                    r"(?:apply|execute|perform|do)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",  # "apply the fix"
                    r"(restart|reboot|stop|start|fix|repair|update)\s+(\w+)",  # "restart service"
                    r"^(yes|ok|proceed|continue)$",  # Simple affirmations
                ]
                for pattern in action_patterns:
                    match = re.search(pattern, lower_input)
                    if match:
                        # Extract the action phrase
                        if match.group(0) in ["yes", "ok", "proceed", "continue"]:
                            inputs[input_name] = "apply_fix"  # Default action for affirmations
                        else:
                            # Combine matched groups into action phrase
                            action = " ".join([g for g in match.groups() if g]).strip()
                            inputs[input_name] = action.replace(" ", "_")
                        break
        
        return inputs
    
    def _start_workflow(self, session: Dict[str, Any], workflow: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """Start a new workflow"""
        session["current_workflow"] = workflow
        session["current_step"] = 0
        session["completed_steps"] = []
        session["collected_inputs"] = {}
        
        first_step = workflow["metadata"]["steps"][0]
        
        # Extract any inputs from the initial workflow start command
        if user_input and first_step.get("required_inputs"):
            initial_inputs = self._extract_inputs(user_input, first_step["required_inputs"])
            session["collected_inputs"].update(initial_inputs)
        
        return {
            "status": "workflow_started",
            "workflow": {
                "id": workflow["id"],
                "title": workflow["metadata"]["title"],
                "total_steps": len(workflow["metadata"]["steps"]),
                "estimated_duration": workflow["metadata"]["estimated_duration"]
            },
            "current_step": 1,
            "step_description": first_step["description"],
            "required_inputs": first_step["required_inputs"],
            "message": f"Starting workflow: {workflow['metadata']['title']}. Step 1: {first_step['description']}"
        }
    
    def _execute_step(
        self, 
        session: Dict[str, Any], 
        user_input: str
    ) -> Dict[str, Any]:
        """Execute current workflow step"""
        workflow = session["current_workflow"]
        current_step_idx = session["current_step"]
        
        if current_step_idx >= len(workflow["metadata"]["steps"]):
            return {
                "status": "workflow_complete",
                "message": "All workflow steps completed successfully!",
                "completed_steps": session["completed_steps"]
            }
        
        current_step = workflow["metadata"]["steps"][current_step_idx]
        
        # Check dependencies
        for dep in current_step["dependencies"]:
            if (dep - 1) not in session["completed_steps"]:
                return {
                    "status": "dependency_error",
                    "message": f"Cannot execute step {current_step_idx + 1}. Step {dep} must be completed first.",
                    "current_step": current_step_idx + 1
                }
        
        # Extract and validate inputs
        new_inputs = self._extract_inputs(user_input, current_step["required_inputs"])
        session["collected_inputs"].update(new_inputs)
        
        missing_inputs = [
            inp for inp in current_step["required_inputs"]
            if inp not in session["collected_inputs"]
        ]
        
        if missing_inputs:
            return {
                "status": "awaiting_input",
                "message": f"Please provide the following information: {', '.join(missing_inputs)}",
                "current_step": current_step_idx + 1,
                "required_inputs": missing_inputs
            }
        
        # Simulate backend API call
        api_result = self._call_backend_api(current_step, session["collected_inputs"])
        
        # Mark step as complete
        session["completed_steps"].append(current_step_idx)
        session["current_step"] = current_step_idx + 1
        
        # Check if workflow is complete
        if session["current_step"] >= len(workflow["metadata"]["steps"]):
            return {
                "status": "workflow_complete",
                "message": f"Step {current_step_idx + 1} completed. Workflow finished successfully!",
                "step_result": api_result,
                "completed_steps": session["completed_steps"]
            }
        
        # Move to next step
        next_step = workflow["metadata"]["steps"][session["current_step"]]
        
        return {
            "status": "step_completed",
            "message": f"Step {current_step_idx + 1} completed: {current_step['completion_criteria']}. Moving to step {session['current_step'] + 1}: {next_step['description']}",
            "step_result": api_result,
            "current_step": session["current_step"] + 1,
            "next_step_description": next_step["description"],
            "next_required_inputs": next_step["required_inputs"]
        }
    
    def _call_backend_api(self, step: Dict[str, Any], inputs: Dict[str, str]) -> Dict[str, Any]:
        """Simulate backend API call (in production, make real API calls)"""
        api_operation = step.get("api_operation")
        
        if not api_operation:
            return {"status": "success", "message": "Step completed (no API operation)"}
        
        # Simulate API call
        return {
            "status": "success",
            "operation": api_operation,
            "inputs": inputs,
            "message": f"Successfully executed {api_operation}",
            "timestamp": datetime.now().isoformat()
        }
    
    def process_instruction(
        self, 
        user_input: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Process user instruction with conversational context"""
        session = self.get_session(session_id)
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Classify intent
        intent = self._classify_intent(user_input, session["current_workflow"])
        
        if intent == "reset_workflow":
            session["current_workflow"] = None
            session["current_step"] = 0
            session["completed_steps"] = []
            session["collected_inputs"] = {}
            return {
                "status": "reset",
                "message": "Workflow reset. How can I help you?"
            }
        
        if intent == "start_workflow":
            # Retrieve relevant workflows
            workflows = self.retrieve_workflows(user_input, top_k=1)
            if not workflows:
                return {
                    "status": "no_workflow_found",
                    "message": "I couldn't find a workflow matching your request. Please try rephrasing."
                }
            return self._start_workflow(session, workflows[0], user_input)
        
        if intent == "continue_workflow" and session["current_workflow"]:
            return self._execute_step(session, user_input)
        
        # Default: answer query using retrieved workflows
        workflows = self.retrieve_workflows(user_input, top_k=5)
        answer = self.generate_answer(user_input, workflows)
        
        return {
            "status": "query_answered",
            "answer": answer,
            "sources": workflows
        }
    
    def generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM"""
        context = self._format_context(documents)
        
        prompt = f"""You are an intelligent workflow assistant helping users execute operational procedures.

Answer the question using the provided context. If the user wants to start a workflow, explain what the workflow does and ask if they want to proceed.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.generate_text(prompt=prompt)
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"


class MetricsCollector:
    """Collect and aggregate metrics for both RAG approaches"""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "standard_rag_queries": 0,
            "workflow_agent_queries": 0,
            "standard_retrieval_times": [],
            "workflow_retrieval_times": [],
            "standard_sources": [],
            "workflow_sources": []
        }
    
    def record_query(
        self, 
        query_type: str, 
        retrieval_time: float, 
        num_sources: int
    ):
        """Record query metrics"""
        self.metrics["total_queries"] += 1
        
        if query_type == "standard":
            self.metrics["standard_rag_queries"] += 1
            self.metrics["standard_retrieval_times"].append(retrieval_time)
            self.metrics["standard_sources"].append(num_sources)
        elif query_type == "workflow":
            self.metrics["workflow_agent_queries"] += 1
            self.metrics["workflow_retrieval_times"].append(retrieval_time)
            self.metrics["workflow_sources"].append(num_sources)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics"""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "standard_rag_queries": self.metrics["standard_rag_queries"],
            "workflow_agent_queries": self.metrics["workflow_agent_queries"],
            "avg_standard_retrieval_time": avg(self.metrics["standard_retrieval_times"]),
            "avg_workflow_retrieval_time": avg(self.metrics["workflow_retrieval_times"]),
            "avg_standard_sources": avg(self.metrics["standard_sources"]),
            "avg_workflow_sources": avg(self.metrics["workflow_sources"])
        }
    
    def reset(self):
        """Reset all metrics"""
        self.__init__()

# Made with Bob
