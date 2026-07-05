const API_BASE = "http://localhost:8000/api";

export interface DocumentInfo {
  id: string;
  filename: string;
  uploaded_at: string;
  file_size: number;
}

export interface SubSourceInfo {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  citation_index: number;
}

export interface SourceInfo {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  contributing_sources?: SubSourceInfo[];
}

export interface ChatMessage {
  id: string;
  session_id: string;
  sender: "user" | "system";
  text: string;
  timestamp: string;
  confidence_score?: number;
  source?: SourceInfo;
}

export interface TrainingStatus {
  status: "idle" | "training" | "completed" | "failed";
  progress: number;
  error: string | null;
  last_trained: string | null;
}

export interface SystemStats {
  cpu_utilization: number;
  ram_utilization: number;
  documents_count: number;
  chunks_count: number;
  queries_served: number;
  vocabulary_size: number;
  model_loaded: boolean;
  training_status: string;
}

export interface EvaluationQuery {
  query: string;
  retrieved_chunk_index: number;
  expected_chunk_index: number;
  retrieval_accuracy: number;
  precision: number;
  recall: number;
  hallucination_rate: number;
  citation_correctness: number;
}

export interface EvaluationResults {
  success: boolean;
  total_queries: number;
  retrieval_accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  hallucination_rate: number;
  citation_correctness: number;
  queries: EvaluationQuery[];
}

export interface ChatSession {
  id: string;
  created_at: string;
}

export const api = {
  // --- Documents Router ---
  async uploadDocument(file: File): Promise<DocumentInfo> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to upload document.");
    }
    return response.json();
  },

  async listDocuments(): Promise<DocumentInfo[]> {
    const response = await fetch(`${API_BASE}/documents`);
    if (!response.ok) throw new Error("Failed to fetch documents.");
    return response.json();
  },

  async deleteDocument(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/documents/${id}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Failed to delete document.");
  },

  // --- Training Router ---
  async startTraining(): Promise<TrainingStatus> {
    const response = await fetch(`${API_BASE}/training/start`, {
      method: "POST",
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to start training.");
    }
    return response.json();
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await fetch(`${API_BASE}/training/status`);
    if (!response.ok) throw new Error("Failed to fetch training status.");
    return response.json();
  },

  // --- Chat Router ---
  async createSession(): Promise<{ id: string }> {
    const response = await fetch(`${API_BASE}/chat/session`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to create chat session.");
    return response.json();
  },

  async listSessions(): Promise<ChatSession[]> {
    const response = await fetch(`${API_BASE}/chat/sessions`);
    if (!response.ok) throw new Error("Failed to list chat sessions.");
    return response.json();
  },

  async deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/chat/session/${sessionId}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new Error("Failed to delete chat session.");
  },

  async queryAssistant(
    sessionId: string,
    text: string,
    confidenceThreshold?: number,
    retrievalAlpha?: number
  ): Promise<ChatMessage> {
    const response = await fetch(`${API_BASE}/chat/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        session_id: sessionId, 
        text,
        confidence_threshold: confidenceThreshold,
        retrieval_alpha: retrievalAlpha
      }),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to query assistant.");
    }
    return response.json();
  },

  async getSessionMessages(sessionId: string): Promise<ChatMessage[]> {
    const response = await fetch(`${API_BASE}/chat/session/${sessionId}/messages`);
    if (!response.ok) throw new Error("Failed to fetch message history.");
    return response.json();
  },

  // --- System stats and evaluation ---
  async getSystemStats(): Promise<SystemStats> {
    const response = await fetch(`${API_BASE}/system/stats`);
    if (!response.ok) throw new Error("Failed to fetch system stats.");
    return response.json();
  },

  async getEvaluationStats(): Promise<EvaluationResults> {
    const response = await fetch(`${API_BASE}/system/evaluation`);
    if (!response.ok) throw new Error("Failed to fetch evaluation stats.");
    return response.json();
  },

  async runEvaluation(): Promise<EvaluationResults> {
    const response = await fetch(`${API_BASE}/system/evaluation/run`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to run evaluation suite.");
    return response.json();
  },

  async rebuildEmbeddings(): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE}/training/rebuild-embeddings`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to start embeddings rebuild.");
    return response.json();
  },

  async rebuildBM25(): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE}/training/rebuild-bm25`, {
      method: "POST",
    });
    if (!response.ok) throw new Error("Failed to start BM25 rebuild.");
    return response.json();
  },

  async exportModel(): Promise<Blob> {
    const response = await fetch(`${API_BASE}/training/export`);
    if (!response.ok) throw new Error("Failed to export model zip.");
    return response.blob();
  },

  async importModel(file: File): Promise<{ status: string; message: string }> {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${API_BASE}/training/import`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Failed to import model zip.");
    }
    return response.json();
  },
};
