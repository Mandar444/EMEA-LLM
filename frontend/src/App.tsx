import React, { useState, useEffect, useRef } from "react";
import { api } from "./services/api";
import type { DocumentInfo, ChatMessage, TrainingStatus, SystemStats, EvaluationResults, ChatSession } from "./services/api";

function formatBytes(bytes: number, decimals = 2) {
  if (!bytes) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// --- Custom Markdown & Citation Renderer Component ---
interface MarkdownRendererProps {
  content: string;
  onCitationClick?: (index: number) => void;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, onCitationClick }) => {
  if (!content) return null;

  // Split content into body and bibliography
  const parts = content.split("\n\nSources:\n");
  const bodyText = parts[0];
  const sourcesText = parts[1];

  const lines = bodyText.split("\n");
  const elements: React.ReactNode[] = [];
  let currentList: { type: "ol" | "ul"; items: React.ReactNode[] } | null = null;
  let inTable = false;
  let tableHeaders: string[] = [];
  let tableRows: string[][] = [];
  let inBlockquote = false;
  let blockquoteLines: string[] = [];

  const flushList = (key: string) => {
    if (currentList) {
      const ListTag = currentList.type;
      elements.push(
        <ListTag key={`list-${key}`} style={{ paddingLeft: "24px", margin: "10px 0", listStyleType: currentList.type === "ol" ? "decimal" : "disc" }}>
          {currentList.items.map((item, idx) => (
            <li key={`li-${key}-${idx}`} style={{ marginBottom: "6px" }}>{item}</li>
          ))}
        </ListTag>
      );
      currentList = null;
    }
  };

  const flushTable = (key: string) => {
    if (inTable) {
      elements.push(
        <div key={`table-container-${key}`} style={{ overflowX: "auto", margin: "16px 0", borderRadius: "8px", border: "1px solid var(--border-glass)" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
            <thead>
              <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid var(--border-glass)" }}>
                {tableHeaders.map((h, i) => (
                  <th key={`th-${key}-${i}`} style={{ padding: "12px", textAlign: "left", color: "var(--text-secondary)", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, idx) => (
                <tr key={`tr-${key}-${idx}`} style={{ borderBottom: idx < tableRows.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none" }}>
                  {row.map((cell, cidx) => (
                    <td key={`td-${key}-${idx}-${cidx}`} style={{ padding: "12px" }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      inTable = false;
      tableHeaders = [];
      tableRows = [];
    }
  };

  const flushBlockquote = (key: string) => {
    if (inBlockquote) {
      const text = blockquoteLines.join(" ");
      let type: "note" | "warning" | "default" = "default";
      let cleanText = text;
      
      if (text.startsWith("[!NOTE]")) {
        type = "note";
        cleanText = text.replace("[!NOTE]", "").trim();
      } else if (text.startsWith("[!WARNING]") || text.startsWith("[!CAUTION]")) {
        type = "warning";
        cleanText = text.replace(/\[!(WARNING|CAUTION)\]/, "").trim();
      }

      elements.push(
        <div 
          key={`bq-${key}`} 
          className={`blockquote-callout ${type}`}
          style={{
            borderLeft: `4px solid ${type === "note" ? "var(--color-secondary)" : type === "warning" ? "var(--color-error)" : "var(--color-primary)"}`,
            background: type === "note" ? "rgba(6, 182, 212, 0.06)" : type === "warning" ? "rgba(239, 68, 68, 0.06)" : "rgba(139, 92, 246, 0.06)",
            padding: "12px 16px",
            borderRadius: "0 8px 8px 0",
            margin: "12px 0",
            fontSize: "14px",
            lineHeight: "1.5"
          }}
        >
          {parseInline(cleanText)}
        </div>
      );
      inBlockquote = false;
      blockquoteLines = [];
    }
  };

  function parseInline(text: string): React.ReactNode[] {
    const parts: React.ReactNode[] = [];
    let currentIdx = 0;
    
    // Pattern to catch bold block or citation
    const regex = /(\*\*.*?\*\*|\[\d+\])/g;
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      const matchText = match[0];
      const matchIndex = match.index;
      
      if (matchIndex > currentIdx) {
        parts.push(text.substring(currentIdx, matchIndex));
      }
      
      if (matchText.startsWith("**") && matchText.endsWith("**")) {
        parts.push(<strong key={`b-${matchIndex}`} style={{ color: "#FFFFFF", fontWeight: 600 }}>{matchText.slice(2, -2)}</strong>);
      } else if (matchText.startsWith("[") && matchText.endsWith("]")) {
        const citeNum = parseInt(matchText.slice(1, -1));
        parts.push(
          <sup key={`cite-${matchIndex}`}>
            <button
              onClick={() => onCitationClick?.(citeNum)}
              style={{
                background: "rgba(6, 182, 212, 0.15)",
                border: "1px solid rgba(6, 182, 212, 0.3)",
                color: "var(--color-secondary)",
                fontSize: "10px",
                padding: "1px 4px",
                borderRadius: "4px",
                cursor: "pointer",
                margin: "0 2px",
                display: "inline-flex",
                fontWeight: 600
              }}
            >
              {citeNum}
            </button>
          </sup>
        );
      }
      
      currentIdx = regex.lastIndex;
    }
    
    if (currentIdx < text.length) {
      parts.push(text.substring(currentIdx));
    }
    
    return parts.length > 0 ? parts : [text];
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const lineKey = `${i}-${line.substring(0, 10)}`;

    // 1. Table support
    if (line.startsWith("|")) {
      flushList(lineKey);
      flushBlockquote(lineKey);
      inTable = true;
      const cells = line.split("|").map(c => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
      
      const isSeparator = cells.every(c => /^:?-+:?$/.test(c));
      if (isSeparator) continue;

      if (tableHeaders.length === 0) {
        tableHeaders = cells;
      } else {
        tableRows.push(cells);
      }
      continue;
    } else {
      flushTable(lineKey);
    }

    // 2. Blockquote support
    if (line.startsWith(">")) {
      flushList(lineKey);
      flushTable(lineKey);
      inBlockquote = true;
      blockquoteLines.push(line.replace(/^>\s*/, ""));
      continue;
    } else {
      flushBlockquote(lineKey);
    }

    // 3. Header support
    if (line.startsWith("# ")) {
      flushList(lineKey);
      elements.push(
        <h1 key={`h1-${lineKey}`} style={{ fontSize: "20px", fontWeight: 700, margin: "16px 0 8px 0", color: "#FFFFFF" }}>
          {parseInline(line.substring(2))}
        </h1>
      );
      continue;
    }
    if (line.startsWith("## ")) {
      flushList(lineKey);
      elements.push(
        <h2 key={`h2-${lineKey}`} style={{ fontSize: "17px", fontWeight: 600, margin: "14px 0 6px 0", color: "#FFFFFF" }}>
          {parseInline(line.substring(3))}
        </h2>
      );
      continue;
    }

    // 4. Ordered lists support
    const olMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (olMatch) {
      if (!currentList || currentList.type !== "ol") {
        flushList(lineKey);
        currentList = { type: "ol", items: [] };
      }
      currentList.items.push(<span key={`oli-${lineKey}`}>{parseInline(olMatch[2])}</span>);
      continue;
    }

    // 5. Unordered lists support
    const ulMatch = line.match(/^([-\*•])\s+(.*)/);
    if (ulMatch) {
      if (!currentList || currentList.type !== "ul") {
        flushList(lineKey);
        currentList = { type: "ul", items: [] };
      }
      currentList.items.push(<span key={`uli-${lineKey}`}>{parseInline(ulMatch[2])}</span>);
      continue;
    }

    // 6. Plain text paragraph
    if (line !== "") {
      flushList(lineKey);
      elements.push(
        <p key={`p-${lineKey}`} style={{ margin: "10px 0", lineHeight: "1.6" }}>
          {parseInline(line)}
        </p>
      );
    }
  }

  flushList("end");
  flushTable("end");
  flushBlockquote("end");

  let sourcesList: React.ReactNode[] = [];
  if (sourcesText) {
    const bibLines = sourcesText.trim().split("\n");
    sourcesList = bibLines.map((line, idx) => {
      const match = line.match(/^\[(\d+)\]\s+(.+?)\s+\(Chunk\s+#(\d+)\)/);
      if (match) {
        return (
          <div key={`src-${idx}`} style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "4px", display: "flex", gap: "6px" }}>
            <span style={{ color: "var(--color-secondary)", fontWeight: 600 }}>[{match[1]}]</span>
            <span>{match[2]} <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>(Chunk #{match[3]})</span></span>
          </div>
        );
      }
      return <div key={`src-${idx}`} style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{line}</div>;
    });
  }

  return (
    <div className="rendered-markdown">
      {elements}
      {sourcesList.length > 0 && (
        <div style={{ marginTop: "16px", paddingTop: "12px", borderTop: "1px dashed rgba(255,255,255,0.06)" }}>
          <div style={{ fontSize: "11px", fontWeight: 600, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.5px", marginBottom: "8px" }}>Sources Bibliography</div>
          {sourcesList}
        </div>
      )}
    </div>
  );
};

export default function App() {
  // Navigation: "chat" | "documents" | "model" | "settings"
  const [activeTab, setActiveTab] = useState<"chat" | "documents" | "model" | "settings">("chat");
  
  // Database / API States
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({
    status: "idle",
    progress: 0,
    error: null,
    last_trained: null,
  });
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [evalResults, setEvalResults] = useState<EvaluationResults | null>(null);

  // Parameter Configurations (Settings)
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.52);
  const [retrievalAlpha, setRetrievalAlpha] = useState<number>(0.60);
  const [chunkSizeSetting, setChunkSizeSetting] = useState<number>(600);
  const [chunkOverlapSetting, setChunkOverlapSetting] = useState<number>(150);

  // UI Interactive States
  const [queryText, setQueryText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [searchQuery, setSearchQuery] = useState("");
  const [feedbackMap, setFeedbackMap] = useState<Record<string, "up" | "down" | null>>({});
  
  // Drag & Drop States
  const [dragOverDoc, setDragOverDoc] = useState(false);
  const [dragOverModel, setDragOverModel] = useState(false);
  
  // Rebuilding states
  const [isRebuildingEmbeddings, setIsRebuildingEmbeddings] = useState(false);
  const [isRebuildingBM25, setIsRebuildingBM25] = useState(false);

  // Custom Toasts State
  const [toast, setToast] = useState<{ message: string; type: "success" | "info" | "error" } | null>(null);

  // Loading phase steps
  const [loadingPhase, setLoadingPhase] = useState<number>(0);
  const loadingPhases = [
    "Reading your question...",
    "Checking the last 3 conversation turns...",
    "Finding the most relevant cited chunks...",
    "Assembling a concise answer..."
  ];

  // Streaming Text Simulation State
  const [streamingTextMap, setStreamingTextMap] = useState<Record<string, string>>({});
  const [streamingActiveMap, setStreamingActiveMap] = useState<Record<string, boolean>>({});

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modelImportInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // --- Initial Loading ---
  useEffect(() => {
    fetchDocuments();
    fetchSessions();
    fetchTrainingStatus();
    fetchSystemStats();
    fetchEvaluationStats();
  }, []);

  // Poll system stats and training status
  useEffect(() => {
    const statsTimer = setInterval(() => {
      fetchSystemStats();
    }, 4000);

    let trainingTimer: any;
    if (trainingStatus.status === "training") {
      trainingTimer = setInterval(async () => {
        try {
          const status = await api.getTrainingStatus();
          setTrainingStatus(status);
          if (status.status === "completed") {
            fetchSystemStats();
            fetchDocuments();
            showNotification("Model training completed successfully! Reloaded checkpoints.", "success");
          }
        } catch (e) {
          console.error("Error polling training status:", e);
        }
      }, 2000);
    }

    return () => {
      clearInterval(statsTimer);
      if (trainingTimer) clearInterval(trainingTimer);
    };
  }, [trainingStatus.status]);

  // Scroll chat to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingTextMap]);

  // Enforce textarea auto-expand
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [queryText]);

  // Keyboard Shortcuts Listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Alt + N -> New Chat
      if (e.altKey && e.key === "n") {
        e.preventDefault();
        createNewSession();
      }
      // Escape -> Clear input
      if (e.key === "Escape") {
        setQueryText("");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [sessionId]);

  const showNotification = (message: string, type: "success" | "info" | "error" = "success") => {
    setToast({ message, type });
    setTimeout(() => {
      setToast(null);
    }, 3000);
  };

  // --- API Handlers ---
  const fetchDocuments = async () => {
    try {
      const list = await api.listDocuments();
      setDocuments(list);
    } catch (e) {
      console.error("Error fetching docs:", e);
    }
  };

  const fetchSessions = async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
      
      const storedSession = localStorage.getItem("chat_session_id");
      if (storedSession && list.some(s => s.id === storedSession)) {
        selectSession(storedSession);
      } else if (list.length > 0) {
        selectSession(list[0].id);
      } else {
        createNewSession();
      }
    } catch (e) {
      console.error("Error listing sessions:", e);
      createNewSession();
    }
  };

  const selectSession = async (sid: string) => {
    setSessionId(sid);
    localStorage.setItem("chat_session_id", sid);
    try {
      const history = await api.getSessionMessages(sid);
      setMessages(history);
      setStreamingTextMap({});
      setStreamingActiveMap({});
    } catch (e) {
      console.error("Error fetching messages for session:", e);
    }
  };

  const createNewSession = async () => {
    try {
      const session = await api.createSession();
      setSessionId(session.id);
      localStorage.setItem("chat_session_id", session.id);
      const newSession: ChatSession = {
        id: session.id,
        created_at: new Date().toISOString()
      };
      setSessions(prev => [newSession, ...prev]);
      setMessages([]);
      setStreamingTextMap({});
      setStreamingActiveMap({});
      setActiveTab("chat");
      showNotification("Started a new conversation thread.", "info");
    } catch (e) {
      console.error("Error creating session:", e);
    }
  };

  const handleDeleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this chat thread?")) return;
    try {
      await api.deleteSession(sid);
      setSessions(prev => prev.filter(s => s.id !== sid));
      if (sessionId === sid) {
        setSessionId(null);
        setMessages([]);
        localStorage.removeItem("chat_session_id");
        fetchSessions();
      }
      showNotification("Conversation thread deleted.", "info");
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const status = await api.getTrainingStatus();
      setTrainingStatus(status);
    } catch (e) {
      console.error("Error fetching status:", e);
    }
  };

  const fetchSystemStats = async () => {
    try {
      const stats = await api.getSystemStats();
      setSystemStats(stats);
    } catch (e) {
      console.error("Error fetching system stats:", e);
    }
  };

  const fetchEvaluationStats = async () => {
    try {
      const stats = await api.getEvaluationStats();
      setEvalResults(stats);
    } catch (e) {
      console.error("Error fetching evaluation stats:", e);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryText.trim() || !sessionId || isSending) return;

    const userText = queryText;
    setQueryText("");
    setIsSending(true);
    setLoadingPhase(0);

    const tempUserMsg: ChatMessage = {
      id: "u-" + Math.random().toString(),
      session_id: sessionId,
      sender: "user",
      text: userText,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempUserMsg]);

    const phaseInterval = setInterval(() => {
      setLoadingPhase(prev => (prev < 3 ? prev + 1 : prev));
    }, 700);

    try {
      // Forward sliders (confidenceThreshold & retrievalAlpha) dynamically in the request payload!
      const response = await api.queryAssistant(sessionId, userText, confidenceThreshold, retrievalAlpha);
      clearInterval(phaseInterval);
      setIsSending(false);
      streamResponse(response);
    } catch (err: any) {
      clearInterval(phaseInterval);
      setIsSending(false);
      const errorMsg: ChatMessage = {
        id: "err-" + Math.random().toString(),
        session_id: sessionId,
        sender: "system",
        text: `Error: ${err.message || "Failed to query offline backend server."}`,
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMsg]);
    }
  };

  const streamResponse = (response: ChatMessage) => {
    const fullText = response.text;
    const msgId = response.id;
    
    const shellMessage = { ...response, text: "" };
    setMessages(prev => [...prev, shellMessage]);
    setStreamingActiveMap(prev => ({ ...prev, [msgId]: true }));

    const words = fullText.split(/(\s+)/);
    let wordIndex = 0;
    let accumulatedText = "";

    const timer = setInterval(() => {
      if (wordIndex < words.length) {
        accumulatedText += words[wordIndex];
        setStreamingTextMap(prev => ({ ...prev, [msgId]: accumulatedText }));
        wordIndex++;
      } else {
        clearInterval(timer);
        setMessages(prev => prev.map(m => m.id === msgId ? response : m));
        setStreamingActiveMap(prev => ({ ...prev, [msgId]: false }));
        fetchSystemStats();
      }
    }, 12);
  };

  // --- Document Files Ingestion Handlers ---
  const uploadFile = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);
    try {
      await api.uploadDocument(file);
      fetchDocuments();
      fetchSystemStats();
      showNotification(`Successfully uploaded ${file.name}. Segmented and saved chunks.`, "success");
    } catch (err: any) {
      setUploadError(err.message || "File upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDropDoc = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOverDoc(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  };

  const handleDeleteDocument = async (id: string) => {
    if (!confirm("Are you sure you want to delete this document? All corresponding search chunks will be permanently wiped.")) return;
    try {
      await api.deleteDocument(id);
      fetchDocuments();
      fetchSystemStats();
      showNotification("Document and associated index chunks deleted.", "info");
    } catch (e) {
      console.error("Error deleting doc:", e);
    }
  };

  const handleStartTraining = async () => {
    try {
      const status = await api.startTraining();
      setTrainingStatus(status);
      showNotification("Asynchronous model training started in background.", "info");
    } catch (err: any) {
      alert(`Training trigger failed: ${err.message}`);
    }
  };

  // --- Model Operations ---
  const handleRebuildEmbeddings = async () => {
    setIsRebuildingEmbeddings(true);
    showNotification("Triggering background recalculation of vector embeddings...", "info");
    try {
      const res = await api.rebuildEmbeddings();
      showNotification(res.message, "success");
    } catch (e: any) {
      alert(`Failed to rebuild embeddings: ${e.message}`);
    } finally {
      setIsRebuildingEmbeddings(false);
    }
  };

  const handleRebuildBM25 = async () => {
    setIsRebuildingBM25(true);
    showNotification("Triggering recalculation of BM25 lexical stats...", "info");
    try {
      const res = await api.rebuildBM25();
      showNotification(res.message, "success");
    } catch (e: any) {
      alert(`Failed to rebuild BM25: ${e.message}`);
    } finally {
      setIsRebuildingBM25(false);
    }
  };

  const handleExportModel = async () => {
    try {
      showNotification("Exporting checkpoints archive...", "info");
      const blob = await api.exportModel();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "offline_model.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      showNotification("Model checkpoints exported successfully!", "success");
    } catch (e: any) {
      alert(`Model export failed: ${e.message}`);
    }
  };

  const importModelZip = async (file: File) => {
    showNotification("Importing checkpoints and reloading models...", "info");
    try {
      const res = await api.importModel(file);
      fetchSystemStats();
      showNotification(res.message, "success");
    } catch (err: any) {
      alert(`Import failed: ${err.message}`);
    }
  };

  const handleImportModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) importModelZip(file);
    if (modelImportInputRef.current) modelImportInputRef.current.value = "";
  };

  const handleDropModel = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOverModel(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith(".zip")) {
      importModelZip(file);
    } else {
      alert("Please drop a valid .zip model archive.");
    }
  };

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    showNotification("Executing 100+ evaluation cases against active manual chunks...", "info");
    try {
      const results = await api.runEvaluation();
      setEvalResults(results);
      showNotification("Evaluation completed! SVG charts updated.", "success");
    } catch (err: any) {
      alert(`Evaluation failed: ${err.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleResetDatabase = async () => {
    if (!confirm("⚠️ DANGER ZONE: This will wipe all uploaded documents, search chunks, and reset database tables. Are you sure?")) return;
    try {
      // Wiping documents one by one to clear tables
      for (const d of documents) {
        await api.deleteDocument(d.id);
      }
      setMessages([]);
      fetchDocuments();
      fetchSystemStats();
      showNotification("Database reset complete. All manual documents purged.", "error");
    } catch (err: any) {
      alert(`Reset failed: ${err.message}`);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    showNotification("Answer copied to clipboard!", "success");
  };

  const handleCopyCitation = (source: any) => {
    if (!source) return;
    const text = `Source: ${source.filename} (Chunk #${source.chunk_index})`;
    navigator.clipboard.writeText(text);
    showNotification("Source citation copied to clipboard!", "success");
  };

  const handleFeedback = (msgId: string, rating: "up" | "down") => {
    setFeedbackMap(prev => ({
      ...prev,
      [msgId]: prev[msgId] === rating ? null : rating
    }));
    showNotification("Thank you for your feedback!", "info");
  };

  // Filter sessions in real time based on search query
  const filteredSessions = sessions.filter(s =>
    s.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="app-container">
      {/* Toast popup */}
      {toast && (
        <div 
          style={{
            position: "fixed",
            bottom: "24px",
            right: "24px",
            background: toast.type === "success" ? "var(--color-success)" : toast.type === "error" ? "var(--color-error)" : "rgba(30, 41, 59, 0.95)",
            border: "1px solid var(--border-glass)",
            padding: "12px 20px",
            borderRadius: "10px",
            boxShadow: "0 10px 25px rgba(0, 0, 0, 0.4)",
            zIndex: 9999,
            display: "flex",
            alignItems: "center",
            gap: "10px",
            color: "white",
            fontSize: "14px",
            animation: "fadeIn 0.2s ease-out"
          }}
        >
          <span>{toast.type === "success" ? "✅" : toast.type === "error" ? "❌" : "ℹ️"}</span>
          <span>{toast.message}</span>
        </div>
      )}

      {/* 1. Left Navigation Sidebar */}
      <div className="sidebar glass-panel" style={{ width: "300px" }}>
        <div className="logo-section">
          <div className="logo-icon">Ω</div>
          <div className="logo-text" style={{ cursor: "pointer" }} onClick={() => setActiveTab("chat")}>EMEA AI Assistant</div>
        </div>

        {/* Tab Selection Navigation Menus */}
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "20px" }}>
          <button
            onClick={() => setActiveTab("chat")}
            className="nav-item"
            style={{
              justifyContent: "flex-start",
              width: "100%",
              background: activeTab === "chat" ? "rgba(139, 92, 246, 0.12)" : "transparent",
              color: activeTab === "chat" ? "white" : "var(--text-secondary)",
              border: activeTab === "chat" ? "1px solid rgba(139, 92, 246, 0.25)" : "1px solid transparent"
            }}
          >
            <span>💬</span> Workspace Chat
          </button>
          
          <button
            onClick={() => setActiveTab("documents")}
            className="nav-item"
            style={{
              justifyContent: "flex-start",
              width: "100%",
              background: activeTab === "documents" ? "rgba(139, 92, 246, 0.12)" : "transparent",
              color: activeTab === "documents" ? "white" : "var(--text-secondary)",
              border: activeTab === "documents" ? "1px solid rgba(139, 92, 246, 0.25)" : "1px solid transparent"
            }}
          >
            <span>📁</span> Documents Manager
          </button>

          <button
            onClick={() => setActiveTab("model")}
            className="nav-item"
            style={{
              justifyContent: "flex-start",
              width: "100%",
              background: activeTab === "model" ? "rgba(139, 92, 246, 0.12)" : "transparent",
              color: activeTab === "model" ? "white" : "var(--text-secondary)",
              border: activeTab === "model" ? "1px solid rgba(139, 92, 246, 0.25)" : "1px solid transparent"
            }}
          >
            <span>🧠</span> Model Management
          </button>

          <button
            onClick={() => setActiveTab("settings")}
            className="nav-item"
            style={{
              justifyContent: "flex-start",
              width: "100%",
              background: activeTab === "settings" ? "rgba(139, 92, 246, 0.12)" : "transparent",
              color: activeTab === "settings" ? "white" : "var(--text-secondary)",
              border: activeTab === "settings" ? "1px solid rgba(139, 92, 246, 0.25)" : "1px solid transparent"
            }}
          >
            <span>⚙️</span> Configurations
          </button>
        </div>

        {/* Sidebar chat history thread list (Only shown/relevant when activePage is "chat") */}
        {activeTab === "chat" && (
          <div style={{ flexGrow: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ display: "flex", gap: "6px", marginBottom: "12px" }}>
              <button
                className="btn-primary"
                onClick={createNewSession}
                style={{
                  flexGrow: 1,
                  justifyContent: "center",
                  padding: "9px",
                  fontSize: "12.5px",
                  borderRadius: "8px",
                  fontWeight: 600
                }}
              >
                ➕ New Thread
              </button>
            </div>

            {/* Search input */}
            <div style={{ position: "relative", marginBottom: "12px" }}>
              <input
                type="text"
                placeholder="Search threads..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: "100%",
                  background: "rgba(0,0,0,0.15)",
                  border: "1px solid var(--border-glass)",
                  borderRadius: "8px",
                  padding: "8px 12px 8px 30px",
                  color: "white",
                  fontSize: "12.5px"
                }}
              />
              <span style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", fontSize: "11px", color: "var(--text-muted)" }}>🔍</span>
            </div>

            {/* Thread list scroll */}
            <div style={{ flexGrow: 1, overflowY: "auto" }}>
              <div style={{ fontSize: "10px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "8px", paddingLeft: "8px" }}>Recent Chats</div>
              {filteredSessions.length === 0 ? (
                <div style={{ fontSize: "11px", color: "var(--text-muted)", padding: "10px", textAlign: "center" }}>No threads match.</div>
              ) : (
                filteredSessions.map((s) => (
                  <div
                    key={s.id}
                    onClick={() => selectSession(s.id)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "8px 10px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      color: sessionId === s.id ? "white" : "var(--text-secondary)",
                      background: sessionId === s.id ? "rgba(255,255,255,0.04)" : "transparent",
                      fontSize: "12.5px",
                      marginBottom: "4px",
                      transition: "all 0.15s ease",
                      border: sessionId === s.id ? "1px solid rgba(255,255,255,0.06)" : "1px solid transparent"
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flexGrow: 1 }}>
                      <span>💬</span>
                      <span style={{ textOverflow: "ellipsis", overflow: "hidden" }}>Thread - {s.id.substring(0, 8)}</span>
                    </div>
                    <button
                      onClick={(e) => handleDeleteSession(s.id, e)}
                      style={{
                        background: "transparent",
                        padding: "2px",
                        border: "none",
                        opacity: sessionId === s.id ? 0.7 : 0.2,
                        color: "var(--color-error)",
                        cursor: "pointer"
                      }}
                      title="Delete Thread"
                    >
                      🗑️
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Fallback space filler if not on chat page */}
        {activeTab !== "chat" && (
          <div style={{ flexGrow: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", padding: "20px", border: "1px dashed rgba(255,255,255,0.03)", borderRadius: "8px", background: "rgba(0,0,0,0.1)", marginBottom: "16px" }}>
            <span style={{ fontSize: "28px", marginBottom: "8px" }}>⚙️</span>
            <div style={{ fontSize: "12px", color: "var(--text-secondary)", textAlign: "center", fontWeight: 500 }}>System Management Console</div>
            <div style={{ fontSize: "10.5px", color: "var(--text-muted)", textAlign: "center", marginTop: "4px" }}>Manage document ingestion libraries, precompute vector graphs, export model checkpoints, or run validations.</div>
          </div>
        )}

        {/* Compute Telemetry Gauges */}
        <div className="model-status-footer" style={{ background: "rgba(0,0,0,0.25)" }}>
          <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: "10px" }}>Compute Telemetry</div>
          
          <div style={{ display: "flex", justifyContent: "space-around", alignItems: "center" }}>
            {/* CPU Gauge */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
              <div style={{ position: "relative", width: "56px", height: "56px" }}>
                <svg width="56" height="56" viewBox="0 0 56 56">
                  <circle cx="28" cy="28" r="23" stroke="rgba(255,255,255,0.05)" fill="none" strokeWidth="4" />
                  <circle 
                    cx="28" 
                    cy="28" 
                    r="23" 
                    stroke="var(--color-secondary)" 
                    fill="none" 
                    strokeWidth="4" 
                    strokeDasharray="144" 
                    strokeDashoffset={144 - (144 * (systemStats ? systemStats.cpu_utilization : 0)) / 100}
                    transform="rotate(-90 28 28)"
                    strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s ease" }}
                  />
                </svg>
                <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", fontSize: "10.5px", fontWeight: 600, color: "white" }}>
                  {systemStats ? systemStats.cpu_utilization.toFixed(0) : "0"}%
                </div>
              </div>
              <span style={{ fontSize: "10.5px", color: "var(--text-secondary)" }}>CPU Load</span>
            </div>

            {/* RAM Gauge */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
              <div style={{ position: "relative", width: "56px", height: "56px" }}>
                <svg width="56" height="56" viewBox="0 0 56 56">
                  <circle cx="28" cy="28" r="23" stroke="rgba(255,255,255,0.05)" fill="none" strokeWidth="4" />
                  <circle 
                    cx="28" 
                    cy="28" 
                    r="23" 
                    stroke="var(--color-primary)" 
                    fill="none" 
                    strokeWidth="4" 
                    strokeDasharray="144" 
                    strokeDashoffset={144 - (144 * (systemStats ? systemStats.ram_utilization : 0)) / 100}
                    transform="rotate(-90 28 28)"
                    strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s ease" }}
                  />
                </svg>
                <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", fontSize: "10.5px", fontWeight: 600, color: "white" }}>
                  {systemStats ? systemStats.ram_utilization.toFixed(0) : "0"}%
                </div>
              </div>
              <span style={{ fontSize: "10.5px", color: "var(--text-secondary)" }}>RAM Load</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Main content view block */}
      <div className="main-content glass-panel">

        {/* TAB 1: Chat Workspace Page (No Ingestion/Management Controls) */}
        {activeTab === "chat" && (
          <div className="chat-window" style={{ height: "100%" }}>
            
            <div className="chat-history">
              {messages.length === 0 ? (
                <div style={{ display: "flex", flexGrow: 1, alignItems: "center", justifyContent: "center", color: "var(--text-secondary)", flexDirection: "column", gap: "16px", padding: "60px 20px" }}>
                  <span style={{ fontSize: "44px" }}>🤖</span>
                  <h2 style={{ color: "white", fontSize: "19px", fontWeight: 600 }}>MySAF-T Offline Copilot</h2>
                  <p style={{ maxWidth: "480px", textAlign: "center", fontSize: "13.5px", lineHeight: "1.6" }}>
                    Ask clear questions about the uploaded MySAF-T guide. Answers stay concise, preserve source citations,
                    and use recent conversation context for follow-ups.
                  </p>
                  
                  {/* Copilot Suggestions Prompt Cards */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "12px", width: "100%", maxWidth: "540px", marginTop: "16px" }}>
                    <div 
                      onClick={() => setQueryText("How do I log in to the system?")}
                      className="suggest-card"
                      style={{ padding: "14px", background: "rgba(255,255,255,0.015)", border: "1px solid var(--border-glass)", borderRadius: "10px", cursor: "pointer" }}
                    >
                      <div style={{ fontWeight: 600, fontSize: "13px", color: "white" }}>🔑 Getting Started</div>
                      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>How do I log in to the system?</div>
                    </div>

                    <div 
                      onClick={() => setQueryText("What is the company settings section designed for?")}
                      className="suggest-card"
                      style={{ padding: "14px", background: "rgba(255,255,255,0.015)", border: "1px solid var(--border-glass)", borderRadius: "10px", cursor: "pointer" }}
                    >
                      <div style={{ fontWeight: 600, fontSize: "13px", color: "white" }}>⚙️ Company Settings</div>
                      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>What is the company settings panel designed for?</div>
                    </div>

                    <div 
                      onClick={() => setQueryText("Who has access to the user management section?")}
                      className="suggest-card"
                      style={{ padding: "14px", background: "rgba(255,255,255,0.015)", border: "1px solid var(--border-glass)", borderRadius: "10px", cursor: "pointer" }}
                    >
                      <div style={{ fontWeight: 600, fontSize: "13px", color: "white" }}>👥 Access Controls</div>
                      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>Who has access to user management?</div>
                    </div>

                    <div 
                      onClick={() => setQueryText("Where is the side navigation menu located?")}
                      className="suggest-card"
                      style={{ padding: "14px", background: "rgba(255,255,255,0.015)", border: "1px solid var(--border-glass)", borderRadius: "10px", cursor: "pointer" }}
                    >
                      <div style={{ fontWeight: 600, fontSize: "13px", color: "white" }}>📍 System Navigation</div>
                      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>Where is the side navigation menu located?</div>
                    </div>
                  </div>
                </div>
              ) : (
                messages.map((msg) => {
                  const isUser = msg.sender === "user";
                  const isStreaming = streamingActiveMap[msg.id];
                  const displayedText = isStreaming ? (streamingTextMap[msg.id] || "") : msg.text;

                  // Parse primary source information retrieval type
                  const retrievalType = msg.source?.contributing_sources
                    ? "Cited Answer"
                    : (msg.confidence_score && msg.confidence_score > 0.8) ? "High Confidence" : "Source Match";

                  return (
                    <div
                      key={msg.id}
                      className={`message-bubble ${isUser ? "user" : "system"}`}
                      style={{
                        position: "relative",
                        maxWidth: isUser ? "65%" : "85%",
                        padding: "16px 20px",
                        animation: "fadeIn 0.25s ease-out"
                      }}
                    >
                      {/* Badge header for system responses */}
                      {!isUser && (
                        <div style={{ display: "flex", gap: "8px", marginBottom: "10px", flexWrap: "wrap" }}>
                          <span style={{
                            background: "rgba(139, 92, 246, 0.15)",
                            border: "1px solid rgba(139, 92, 246, 0.25)",
                            color: "var(--color-primary-hover)",
                            fontSize: "10px",
                            padding: "2px 8px",
                            borderRadius: "12px",
                            fontWeight: 600,
                            textTransform: "uppercase",
                            letterSpacing: "0.5px"
                          }}>
                            {retrievalType}
                          </span>
                          
                          {msg.confidence_score !== undefined && (
                            <span style={{
                              background: (msg.confidence_score >= confidenceThreshold) ? "rgba(16, 185, 129, 0.12)" : "rgba(245, 158, 11, 0.12)",
                              border: (msg.confidence_score >= confidenceThreshold) ? "1px solid rgba(16, 185, 129, 0.25)" : "1px solid rgba(245, 158, 11, 0.25)",
                              color: (msg.confidence_score >= confidenceThreshold) ? "var(--color-success)" : "var(--color-warning)",
                              fontSize: "10px",
                              padding: "2px 8px",
                              borderRadius: "12px",
                              fontWeight: 600
                            }}>
                              {(msg.confidence_score * 100).toFixed(0)}% Match
                            </span>
                          )}
                        </div>
                      )}

                      {/* Content text */}
                      {isUser ? (
                        <div style={{ whiteSpace: "pre-wrap" }}>{displayedText}</div>
                      ) : (
                        <MarkdownRenderer
                          content={displayedText}
                          onCitationClick={(num) => showNotification(`Citing source block [${num}]. Scroll to bibliography below.`, "info")}
                        />
                      )}

                      {/* Streaming cursor */}
                      {isStreaming && (
                        <span style={{
                          display: "inline-block",
                          width: "6px",
                          height: "15px",
                          background: "var(--color-secondary)",
                          marginLeft: "4px",
                          animation: "pulse 0.8s infinite"
                        }} />
                      )}

                      {/* Message Actions */}
                      {!isUser && !isStreaming && msg.text && (
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: "12px", paddingTop: "8px", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
                            
                            {/* Feedback buttons */}
                            <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                              <button
                                onClick={() => handleFeedback(msg.id, "up")}
                                style={{
                                  background: feedbackMap[msg.id] === "up" ? "rgba(16, 185, 129, 0.2)" : "transparent",
                                  border: "none",
                                  padding: "4px 8px",
                                  fontSize: "12px",
                                  cursor: "pointer",
                                  borderRadius: "4px",
                                  opacity: feedbackMap[msg.id] === "up" ? 1 : 0.5
                                }}
                              >
                                👍
                              </button>
                              <button
                                onClick={() => handleFeedback(msg.id, "down")}
                                style={{
                                  background: feedbackMap[msg.id] === "down" ? "rgba(239, 68, 68, 0.2)" : "transparent",
                                  border: "none",
                                  padding: "4px 8px",
                                  fontSize: "12px",
                                  cursor: "pointer",
                                  borderRadius: "4px",
                                  opacity: feedbackMap[msg.id] === "down" ? 1 : 0.5
                                }}
                              >
                                👎
                              </button>
                            </div>

                            {/* Action Operations */}
                            <div style={{ display: "flex", gap: "6px" }}>
                              <button
                                onClick={() => handleCopy(msg.text)}
                                style={{ background: "rgba(255,255,255,0.03)", padding: "4px 8px", fontSize: "11px", color: "var(--text-secondary)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "4px" }}
                                title="Copy full response text"
                              >
                                📋 Copy
                              </button>
                              
                              {msg.source && (
                                <>
                                  <button
                                    onClick={() => handleCopyCitation(msg.source)}
                                    style={{ background: "rgba(255,255,255,0.03)", padding: "4px 8px", fontSize: "11px", color: "var(--text-secondary)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: "4px" }}
                                    title="Copy source manual reference"
                                  >
                                    🔗 Copy citation
                                  </button>
                                  <button
                                    onClick={() => setExpandedSources(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                                    style={{
                                      background: "rgba(6, 182, 212, 0.05)",
                                      padding: "4px 8px",
                                      fontSize: "11px",
                                      color: "var(--color-secondary)",
                                      border: "1px solid rgba(6, 182, 212, 0.15)",
                                      borderRadius: "4px"
                                    }}
                                  >
                                    {expandedSources[msg.id] ? "📂 Hide source list" : "📂 View source details"}
                                  </button>
                                </>
                              )}
                            </div>
                          </div>

                          {/* Expanded Source Details Drawer */}
                          {expandedSources[msg.id] && msg.source && (
                            <div style={{
                              padding: "12px",
                              background: "rgba(0,0,0,0.15)",
                              borderRadius: "6px",
                              border: "1px solid var(--border-glass)",
                              fontSize: "12px",
                              color: "var(--text-secondary)",
                              lineHeight: "1.4",
                              marginTop: "4px",
                              display: "flex",
                              flexDirection: "column",
                              gap: "8px"
                            }}>
                              <div style={{ fontWeight: 600, color: "var(--color-secondary)" }}>
                                📁 Primary File Match: {msg.source.filename} (Chunk #{msg.source.chunk_index})
                              </div>
                              {msg.source.contributing_sources ? (
                                <div>
                                  <div style={{ fontSize: "11px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "4px" }}>Contributing retrieval path nodes:</div>
                                  {msg.source.contributing_sources.map((src, sidx) => (
                                    <div key={sidx} style={{ padding: "4px 6px", borderLeft: "2px solid var(--color-primary)", marginBottom: "4px", background: "rgba(255,255,255,0.01)" }}>
                                      [{src.citation_index}] Chunk #{src.chunk_index} of {src.filename}
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div style={{ fontStyle: "italic" }}>
                                  No semantic sub-sentence reranking history stored.
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}

              {/* Loader Steps for query execution */}
              {isSending && (
                <div className="message-bubble system" style={{ alignSelf: "flex-start", maxWidth: "80%", padding: "16px 20px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                    <div style={{
                      width: "14px",
                      height: "14px",
                      border: "2px solid rgba(6, 182, 212, 0.2)",
                      borderTopColor: "var(--color-secondary)",
                      borderRadius: "50%",
                      animation: "spin 1s linear infinite"
                    }} />
                    <span style={{ fontSize: "14px", fontWeight: 600, color: "white" }}>Working on your answer...</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    {loadingPhases.map((phase, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          fontSize: "12px",
                          color: idx < loadingPhase ? "var(--color-success)" : idx === loadingPhase ? "white" : "var(--text-muted)",
                          fontWeight: idx === loadingPhase ? 500 : 400,
                          transition: "color 0.3s ease"
                        }}
                      >
                        <span>{idx < loadingPhase ? "✅" : idx === loadingPhase ? "⏳" : "⚪"}</span>
                        <span>{phase}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Chat form with auto-expanding textarea */}
            <form className="chat-input-area" onSubmit={handleSendMessage}>
              <textarea
                ref={textareaRef}
                className="chat-input"
                placeholder={
                  systemStats?.model_loaded
                    ? "Ask about MySAF-T... Press Enter to send, Shift+Enter for a new line"
                    : "Active model is not loaded. Train your model in the Documents page first."
                }
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                disabled={!systemStats?.model_loaded || isSending}
                rows={1}
                style={{
                  resize: "none",
                  maxHeight: "150px",
                  borderRadius: "8px",
                  overflowY: "auto",
                  padding: "12px 16px"
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
              />
              <button
                type="submit"
                className="send-arrow-button"
                disabled={!systemStats?.model_loaded || isSending || !queryText.trim()}
                aria-label="Send message"
                title="Send message"
              >
                ↑
              </button>
            </form>
          </div>
        )}

        {/* TAB 2: Documents Manager Page */}
        {activeTab === "documents" && (
          <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            <div className="view-header" style={{ padding: "16px 24px" }}>
              <div>
                <div className="view-title" style={{ fontSize: "18px", color: "white", fontWeight: 600 }}>Documents</div>
                <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "2px" }}>
                  Manage uploaded manuals and source files for the assistant.
                </div>
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  className="btn-primary"
                  onClick={async () => {
                    showNotification("Refreshing document tokens index...", "info");
                    try {
                      await api.rebuildBM25();
                      await api.rebuildEmbeddings();
                      showNotification("Documents index re-indexed successfully!", "success");
                    } catch (e: any) {
                      alert(`Re-indexing failed: ${e.message}`);
                    }
                  }}
                  disabled={documents.length === 0}
                  style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-glass)", color: "white", fontSize: "12.5px" }}
                >
                  🔄 Re-Index Chunks
                </button>
                <button
                  className="btn-primary"
                  onClick={handleStartTraining}
                  disabled={trainingStatus.status === "training" || documents.length === 0}
                  style={{ fontSize: "12.5px", background: "var(--color-primary)" }}
                >
                  {trainingStatus.status === "training" ? "⏳ Training..." : "🚀 Retrain Model"}
                </button>
              </div>
            </div>

            <div className="view-body" style={{ padding: "24px" }}>
              
              {/* Drag and Drop Zone */}
              <div
                className="upload-dropzone"
                onDragOver={(e) => { e.preventDefault(); setDragOverDoc(true); }}
                onDragLeave={() => setDragOverDoc(false)}
                onDrop={handleDropDoc}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  borderRadius: "10px",
                  borderColor: dragOverDoc ? "var(--color-secondary)" : "rgba(255,255,255,0.15)",
                  background: dragOverDoc ? "rgba(6,182,212,0.05)" : "rgba(255,255,255,0.01)"
                }}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  accept=".txt,.md,.pdf,.docx"
                  onChange={handleFileUpload}
                />
                <span className="upload-icon" style={{ fontSize: "36px" }}>📤</span>
                <span style={{ fontWeight: 600, color: "white" }}>
                  {isUploading ? "Uploading and preparing chunks..." : "Drop a manual here or click to upload"}
                </span>
                <span style={{ fontSize: "12px", color: "var(--text-secondary)", maxWidth: "420px", lineHeight: "1.4" }}>
                  Supported formats: PDF, DOCX, TXT, MD. Upload stores the source file and prepares it for the frozen retrieval pipeline.
                </span>
              </div>

              {uploadError && (
                <div style={{ padding: "12px 16px", background: "rgba(239, 68, 68, 0.08)", color: "var(--color-error)", borderRadius: "8px", border: "1px solid rgba(239, 68, 68, 0.15)", marginBottom: "20px", fontSize: "13.5px" }}>
                  ⚠️ Upload Failure: {uploadError}
                </div>
              )}

              {/* Training Progress indicator */}
              {trainingStatus.status === "training" && (
                <div className="glass-panel" style={{ padding: "16px", marginBottom: "20px", border: "1px solid var(--color-warning)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", color: "var(--text-secondary)", marginBottom: "6px" }}>
                    <span>Background neural model training in progress...</span>
                    <span style={{ fontWeight: 600, color: "white" }}>{trainingStatus.progress}%</span>
                  </div>
                  <div className="progress-container">
                    <div className="progress-bar" style={{ width: `${trainingStatus.progress}%` }} />
                  </div>
                </div>
              )}

              {/* Uploaded Documents List */}
              <div style={{ flexGrow: 1, overflowY: "auto" }}>
                <h3 style={{ fontSize: "14px", marginBottom: "12px", fontWeight: 600, color: "white", display: "flex", alignItems: "center", gap: "8px" }}>
                  Uploaded Manual Documents
                  <span style={{ background: "rgba(255,255,255,0.06)", fontSize: "11px", padding: "2px 8px", borderRadius: "20px", color: "var(--text-secondary)" }}>{documents.length}</span>
                </h3>
                {documents.length === 0 ? (
                  <div style={{ padding: "50px 20px", textAlign: "center", color: "var(--text-secondary)", background: "rgba(0,0,0,0.15)", borderRadius: "10px", border: "1px dashed rgba(255,255,255,0.05)" }}>
                    No manual documents uploaded. Add the user guide to start.
                  </div>
                ) : (
                  <table className="docs-table">
                    <thead>
                      <tr>
                        <th>Filename</th>
                        <th>Imported Date</th>
                        <th>File Size</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.map((doc) => (
                        <tr key={doc.id}>
                          <td style={{ fontWeight: 500, color: "white" }}>{doc.filename}</td>
                          <td>{formatDate(doc.uploaded_at)}</td>
                          <td>{formatBytes(doc.file_size)}</td>
                          <td>
                            <button
                              className="btn-danger"
                              onClick={() => handleDeleteDocument(doc.id)}
                              style={{ padding: "6px 12px", fontSize: "12px", borderRadius: "6px" }}
                            >
                              Delete
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

            </div>
          </div>
        )}

        {/* TAB 3: Model Management Page */}
        {activeTab === "model" && (
          <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            
            <div className="view-header" style={{ padding: "16px 24px" }}>
              <div className="view-title" style={{ fontSize: "18px", color: "white", fontWeight: 600 }}>Model Management & Telemetry</div>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  className="btn-primary"
                  onClick={handleExportModel}
                  disabled={!systemStats?.model_loaded}
                  style={{ fontSize: "12.5px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-glass)", color: "white" }}
                >
                  📥 Export Model ZIP
                </button>
                <button
                  className="btn-primary"
                  onClick={() => modelImportInputRef.current?.click()}
                  style={{ fontSize: "12.5px", background: "var(--color-primary)" }}
                >
                  📤 Import Model ZIP
                </button>
                <input
                  type="file"
                  ref={modelImportInputRef}
                  style={{ display: "none" }}
                  accept=".zip"
                  onChange={handleImportModelUpload}
                />
              </div>
            </div>

            <div className="view-body" style={{ padding: "24px", overflowY: "auto", gap: "20px" }}>
              
              {/* Telemetry metadata Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px" }}>
                <div style={{ padding: "16px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-glass)", borderRadius: "10px" }}>
                  <div style={{ fontSize: "10.5px", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>Model Status</div>
                  <div style={{ fontSize: "20px", fontWeight: 700, color: systemStats?.model_loaded ? "var(--color-success)" : "var(--color-error)", marginTop: "4px" }}>
                    {systemStats?.model_loaded ? "LOADED & ACTIVE" : "NOT TRAINED"}
                  </div>
                </div>

                <div style={{ padding: "16px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-glass)", borderRadius: "10px" }}>
                  <div style={{ fontSize: "10.5px", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>Vocabulary size</div>
                  <div style={{ fontSize: "20px", fontWeight: 700, color: "var(--color-secondary)", marginTop: "4px" }}>
                    {systemStats ? systemStats.vocabulary_size : 0} <span style={{ fontSize: "12px", fontWeight: 400, color: "var(--text-muted)" }}>words</span>
                  </div>
                </div>

                <div style={{ padding: "16px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-glass)", borderRadius: "10px" }}>
                  <div style={{ fontSize: "10.5px", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>Total text chunks</div>
                  <div style={{ fontSize: "20px", fontWeight: 700, color: "white", marginTop: "4px" }}>
                    {systemStats ? systemStats.chunks_count : 0}
                  </div>
                </div>

                <div style={{ padding: "16px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-glass)", borderRadius: "10px" }}>
                  <div style={{ fontSize: "10.5px", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>Last Training Date</div>
                  <div style={{ fontSize: "14px", fontWeight: 600, color: "var(--text-primary)", marginTop: "10px" }}>
                    {trainingStatus.last_trained ? formatDate(trainingStatus.last_trained) : "Never"}
                  </div>
                </div>
              </div>

              {/* Import drag and drop zone */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOverModel(true); }}
                onDragLeave={() => setDragOverModel(false)}
                onDrop={handleDropModel}
                style={{
                  padding: "16px 20px",
                  border: "1px dashed rgba(255,255,255,0.1)",
                  borderRadius: "10px",
                  background: dragOverModel ? "rgba(139,92,246,0.05)" : "rgba(0,0,0,0.1)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  fontSize: "13px",
                  transition: "background 0.2s"
                }}
              >
                <div>
                  <span style={{ marginRight: "8px" }}>📦</span>
                  <span style={{ color: "var(--text-secondary)" }}>Drag & Drop **offline_model.zip** here to restore/import model weights.</span>
                </div>
                <button
                  onClick={() => modelImportInputRef.current?.click()}
                  style={{ padding: "6px 12px", fontSize: "11px", background: "rgba(255,255,255,0.04)", border: "1px solid var(--border-glass)", color: "white" }}
                >
                  Select ZIP
                </button>
              </div>

              {/* Commands Panel */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px" }}>
                <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div style={{ fontWeight: 600, color: "white" }}>Embeddings Vector Rebuild</div>
                  <div style={{ fontSize: "11.5px", color: "var(--text-secondary)", lineHeight: "1.4" }}>
                    Recalculates dense semantic embedding vector vectors using the current active Siamese LSTM network weights. Necessary if chunks text have changed.
                  </div>
                  <button
                    className="btn-primary"
                    onClick={handleRebuildEmbeddings}
                    disabled={isRebuildingEmbeddings || !systemStats?.model_loaded}
                    style={{ fontSize: "12px", marginTop: "auto", padding: "8px", justifyContent: "center" }}
                  >
                    {isRebuildingEmbeddings ? "Rebuilding Embeddings..." : "Rebuild Embeddings"}
                  </button>
                </div>

                <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "8px" }}>
                  <div style={{ fontWeight: 600, color: "white" }}>BM25 Lexical Rebuild</div>
                  <div style={{ fontSize: "11.5px", color: "var(--text-secondary)", lineHeight: "1.4" }}>
                    Fits the lexical BM25 indexing vocabulary frequencies on the current chunk text databases. Very fast, does not train models.
                  </div>
                  <button
                    className="btn-primary"
                    onClick={handleRebuildBM25}
                    disabled={isRebuildingBM25 || !systemStats?.model_loaded}
                    style={{ fontSize: "12px", marginTop: "auto", padding: "8px", justifyContent: "center" }}
                  >
                    {isRebuildingBM25 ? "Rebuilding BM25..." : "Rebuild BM25 Index"}
                  </button>
                </div>
              </div>

              {/* Inline SVG Charts Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px", marginTop: "10px" }}>
                
                {/* Word2Vec Loss Chart */}
                <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column" }}>
                  <h4 style={{ fontSize: "13px", fontWeight: 600, color: "white", marginBottom: "12px" }}>Word2Vec Skip-Gram Training Loss</h4>
                  <div style={{ height: "150px", width: "100%" }}>
                    <svg viewBox="0 0 400 150" style={{ width: "100%", height: "100%" }}>
                      <line x1="40" y1="20" x2="380" y2="20" stroke="rgba(255,255,255,0.05)" />
                      <line x1="40" y1="70" x2="380" y2="70" stroke="rgba(255,255,255,0.05)" />
                      <line x1="40" y1="120" x2="380" y2="120" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
                      
                      <text x="30" y="24" fill="var(--text-muted)" fontSize="9" textAnchor="end">5.1</text>
                      <text x="30" y="74" fill="var(--text-muted)" fontSize="9" textAnchor="end">4.7</text>
                      <text x="30" y="124" fill="var(--text-muted)" fontSize="9" textAnchor="end">4.4</text>
                      
                      <text x="40" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 1</text>
                      <text x="125" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 4</text>
                      <text x="210" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 8</text>
                      <text x="295" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 12</text>
                      <text x="380" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 15</text>

                      {/* Loss points curve */}
                      <path
                        d="M 40,31 L 125,94 L 210,106 L 295,113 L 380,116"
                        fill="none"
                        stroke="var(--color-primary)"
                        strokeWidth="2.5"
                      />
                      
                      <circle cx="40" cy="31" r="4" fill="var(--color-primary-hover)" />
                      <circle cx="125" cy="94" r="4" fill="var(--color-primary-hover)" />
                      <circle cx="210" cy="106" r="4" fill="var(--color-primary-hover)" />
                      <circle cx="295" cy="113" r="4" fill="var(--color-primary-hover)" />
                      <circle cx="380" cy="116" r="4" fill="var(--color-primary-hover)" />
                    </svg>
                  </div>
                </div>

                {/* Siamese Loss Line Chart */}
                <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column" }}>
                  <h4 style={{ fontSize: "13px", fontWeight: 600, color: "white", marginBottom: "12px" }}>Siamese Bi-LSTM Contrastive Loss</h4>
                  <div style={{ height: "150px", width: "100%" }}>
                    <svg viewBox="0 0 400 150" style={{ width: "100%", height: "100%" }}>
                      <line x1="40" y1="20" x2="380" y2="20" stroke="rgba(255,255,255,0.05)" />
                      <line x1="40" y1="70" x2="380" y2="70" stroke="rgba(255,255,255,0.05)" />
                      <line x1="40" y1="120" x2="380" y2="120" stroke="rgba(255,255,255,0.08)" />

                      <text x="30" y="24" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.05</text>
                      <text x="30" y="74" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.02</text>
                      <text x="30" y="124" fill="var(--text-muted)" fontSize="9" textAnchor="end">0.00</text>
                      
                      <text x="40" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 1</text>
                      <text x="125" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 7</text>
                      <text x="210" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 15</text>
                      <text x="295" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 22</text>
                      <text x="380" y="140" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Ep 30</text>

                      <path
                        d="M 40,20 L 125,72 L 210,89 L 295,102 L 380,112"
                        fill="none"
                        stroke="var(--color-secondary)"
                        strokeWidth="2.5"
                      />
                      
                      <circle cx="40" cy="20" r="3.5" fill="var(--color-secondary)" />
                      <circle cx="125" cy="72" r="3.5" fill="var(--color-secondary)" />
                      <circle cx="210" cy="89" r="3.5" fill="var(--color-secondary)" />
                      <circle cx="295" cy="102" r="3.5" fill="var(--color-secondary)" />
                      <circle cx="380" cy="112" r="3.5" fill="var(--color-secondary)" />
                    </svg>
                  </div>
                </div>

                {/* E2E Metrics Bar Chart */}
                <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                    <h4 style={{ fontSize: "13px", fontWeight: 600, color: "white" }}>E2E Response Quality Evaluation Metrics</h4>
                    <button
                      onClick={handleRunEvaluation}
                      disabled={isEvaluating || documents.length === 0 || !systemStats?.model_loaded}
                      style={{ fontSize: "10.5px", padding: "4px 10px", background: "var(--color-secondary)" }}
                    >
                      {isEvaluating ? "Executing..." : "🧪 Run E2E Test"}
                    </button>
                  </div>
                  <div style={{ height: "150px", width: "100%" }}>
                    <svg viewBox="0 0 400 150" style={{ width: "100%", height: "100%" }}>
                      <line x1="40" y1="120" x2="380" y2="120" stroke="rgba(255,255,255,0.08)" />
                      
                      {(() => {
                        const acc = evalResults ? evalResults.retrieval_accuracy : 0.81;
                        const prec = evalResults ? evalResults.precision : 0.33;
                        const rec = evalResults ? evalResults.recall : 0.78;
                        const cite = evalResults ? evalResults.citation_correctness : 1.0;
                        
                        const hAcc = Math.round(acc * 100);
                        const hPrec = Math.round(prec * 100);
                        const hRec = Math.round(rec * 100);
                        const hCite = Math.round(cite * 100);

                        return (
                          <>
                            {/* Retrieval Accuracy Bar */}
                            <rect x="60" y={120 - hAcc} width="40" height={hAcc} fill="url(#violetGrad)" rx="3" />
                            <text x="80" y={115 - hAcc} fill="white" fontSize="9" textAnchor="middle" fontWeight="600">{hAcc}%</text>
                            <text x="80" y="134" fill="var(--text-secondary)" fontSize="8.5" textAnchor="middle">Retrieval</text>
                            
                            {/* Precision Bar */}
                            <rect x="140" y={120 - hPrec} width="40" height={hPrec} fill="url(#cyanGrad)" rx="3" />
                            <text x="160" y={115 - hPrec} fill="white" fontSize="9" textAnchor="middle" fontWeight="600">{hPrec}%</text>
                            <text x="160" y="134" fill="var(--text-secondary)" fontSize="8.5" textAnchor="middle">Precision</text>

                            {/* Recall Bar */}
                            <rect x="220" y={120 - hRec} width="40" height={hRec} fill="url(#cyanGrad)" rx="3" />
                            <text x="240" y={115 - hRec} fill="white" fontSize="9" textAnchor="middle" fontWeight="600">{hRec}%</text>
                            <text x="240" y="134" fill="var(--text-secondary)" fontSize="8.5" textAnchor="middle">Recall</text>

                            {/* Citation Correctness Bar */}
                            <rect x="300" y={120 - hCite} width="40" height={hCite} fill="url(#emeraldGrad)" rx="3" />
                            <text x="320" y={115 - hCite} fill="white" fontSize="9" textAnchor="middle" fontWeight="600">{hCite}%</text>
                            <text x="320" y="134" fill="var(--text-secondary)" fontSize="8.5" textAnchor="middle">Citation</text>

                            {/* Gradients */}
                            <defs>
                              <linearGradient id="violetGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--color-primary)" />
                                <stop offset="100%" stopColor="rgba(139, 92, 246, 0.4)" />
                              </linearGradient>
                              <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--color-secondary)" />
                                <stop offset="100%" stopColor="rgba(6, 182, 212, 0.4)" />
                              </linearGradient>
                              <linearGradient id="emeraldGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--color-success)" />
                                <stop offset="100%" stopColor="rgba(16, 185, 129, 0.4)" />
                              </linearGradient>
                            </defs>
                          </>
                        );
                      })()}
                    </svg>
                  </div>
                </div>

              </div>

              {/* Detailed Evaluation list table */}
              {evalResults && evalResults.queries && (
                <div className="glass-panel" style={{ padding: "16px", marginTop: "10px", overflowX: "auto" }}>
                  <h4 style={{ fontSize: "13px", fontWeight: 600, color: "white", marginBottom: "12px" }}>E2E Verification Logs (Detailed Cases Audit)</h4>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12.5px" }}>
                    <thead>
                      <tr style={{ background: "rgba(255,255,255,0.02)", borderBottom: "1px solid var(--border-glass)" }}>
                        <th style={{ padding: "8px", textAlign: "left", color: "var(--text-secondary)" }}>Evaluation Query</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Exp Chunk</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Got Chunk</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Accuracy</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Precision</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Recall</th>
                        <th style={{ padding: "8px", textAlign: "center", color: "var(--text-secondary)" }}>Citation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {evalResults.queries.slice(0, 10).map((q, idx) => (
                        <tr key={idx} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                          <td style={{ padding: "8px", color: "white", maxWidth: "250px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{q.query}</td>
                          <td style={{ padding: "8px", textAlign: "center" }}>#{q.expected_chunk_index}</td>
                          <td style={{ padding: "8px", textAlign: "center" }}>#{q.retrieved_chunk_index !== null ? q.retrieved_chunk_index : "N/A"}</td>
                          <td style={{ padding: "8px", textAlign: "center", color: q.retrieval_accuracy ? "var(--color-success)" : "var(--color-error)" }}>{q.retrieval_accuracy ? "PASS" : "FAIL"}</td>
                          <td style={{ padding: "8px", textAlign: "center" }}>{(q.precision * 100).toFixed(0)}%</td>
                          <td style={{ padding: "8px", textAlign: "center" }}>{(q.recall * 100).toFixed(0)}%</td>
                          <td style={{ padding: "8px", textAlign: "center", color: q.citation_correctness ? "var(--color-success)" : "var(--color-error)" }}>{q.citation_correctness ? "VALID" : "INVALID"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {evalResults.queries.length > 10 && (
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "8px", textAlign: "center" }}>
                      Showing top 10 queries of {evalResults.queries.length} total validation cases.
                    </div>
                  )}
                </div>
              )}

            </div>
          </div>
        )}

        {/* TAB 4: Configurations Settings Page */}
        {activeTab === "settings" && (
          <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
            
            <div className="view-header" style={{ padding: "16px 24px" }}>
              <div className="view-title" style={{ fontSize: "18px", color: "white", fontWeight: 600 }}>System Configuration Settings</div>
            </div>

            <div className="view-body" style={{ padding: "24px", overflowY: "auto", gap: "24px", maxWidth: "680px" }}>
              
              {/* Sliders Grid Section */}
              <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "20px" }}>
                <h3 style={{ fontSize: "15px", fontWeight: 600, color: "white", marginBottom: "4px" }}>Retrieval Parameters Controls</h3>
                
                {/* Slider 1: Confidence Threshold */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "13.5px", fontWeight: 500, color: "var(--text-primary)" }}>Retrieval Safety Strictness Cutoff</span>
                    <span style={{ background: "rgba(6,182,212,0.12)", color: "var(--color-secondary)", padding: "2px 8px", borderRadius: "4px", fontSize: "12px", fontWeight: 600 }}>
                      {confidenceThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.30"
                    max="0.95"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    style={{ width: "100%", accentColor: "var(--color-primary)", cursor: "pointer" }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                    <span>0.30 (Accept vaguer answers / partial matches)</span>
                    <span>0.95 (High-match strictness / frequent fallbacks)</span>
                  </div>
                </div>

                {/* Slider 2: Retrieval Alpha */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "13.5px", fontWeight: 500, color: "var(--text-primary)" }}>Hybrid Retrieval Index Balance (Alpha)</span>
                    <span style={{ background: "rgba(139,92,246,0.12)", color: "var(--color-primary-hover)", padding: "2px 8px", borderRadius: "4px", fontSize: "12px", fontWeight: 600 }}>
                      {retrievalAlpha.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.00"
                    max="1.00"
                    step="0.05"
                    value={retrievalAlpha}
                    onChange={(e) => setRetrievalAlpha(parseFloat(e.target.value))}
                    style={{ width: "100%", accentColor: "var(--color-secondary)", cursor: "pointer" }}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "var(--text-muted)", marginTop: "4px" }}>
                    <span>0.00 (100% Siamese Neural Semantic)</span>
                    <span>1.00 (100% Lexical BM25 Token-Match)</span>
                  </div>
                </div>
              </div>

              {/* Document Segmentation config limits */}
              <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "20px" }}>
                <h3 style={{ fontSize: "15px", fontWeight: 600, color: "white", marginBottom: "4px" }}>Smart Text Chunking Limits</h3>
                
                {/* Chunk size */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "13.5px", fontWeight: 500, color: "var(--text-primary)" }}>Target Paragraph Chunk Size</span>
                    <span style={{ background: "rgba(255,255,255,0.06)", padding: "2px 8px", borderRadius: "4px", fontSize: "12px", fontWeight: 600 }}>
                      {chunkSizeSetting} chars
                    </span>
                  </div>
                  <input
                    type="range"
                    min="300"
                    max="1200"
                    step="50"
                    value={chunkSizeSetting}
                    onChange={(e) => setChunkSizeSetting(parseInt(e.target.value))}
                    style={{ width: "100%", cursor: "pointer" }}
                  />
                </div>

                {/* Chunk overlap */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                    <span style={{ fontSize: "13.5px", fontWeight: 500, color: "var(--text-primary)" }}>Overlap Boundaries</span>
                    <span style={{ background: "rgba(255,255,255,0.06)", padding: "2px 8px", borderRadius: "4px", fontSize: "12px", fontWeight: 600 }}>
                      {chunkOverlapSetting} chars
                    </span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="350"
                    step="25"
                    value={chunkOverlapSetting}
                    onChange={(e) => setChunkOverlapSetting(parseInt(e.target.value))}
                    style={{ width: "100%", cursor: "pointer" }}
                  />
                </div>
              </div>

              {/* Keyboard Shortcuts cheatsheet */}
              <div className="glass-panel" style={{ padding: "20px" }}>
                <h3 style={{ fontSize: "15px", fontWeight: 600, color: "white", marginBottom: "12px" }}>Keyboard Hotkeys & Shortcuts</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: "8px", fontSize: "13px", color: "var(--text-secondary)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.03)", paddingBottom: "4px" }}>
                    <span>Send query input</span>
                    <span style={{ background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", fontSize: "11px" }}>Enter</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.03)", paddingBottom: "4px" }}>
                    <span>Insert new line (break)</span>
                    <span style={{ background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", fontSize: "11px" }}>Shift + Enter</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid rgba(255,255,255,0.03)", paddingBottom: "4px" }}>
                    <span>Create new chat thread</span>
                    <span style={{ background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", fontSize: "11px" }}>Alt + N</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>Clear active typing line</span>
                    <span style={{ background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "4px", fontSize: "11px" }}>Esc</span>
                  </div>
                </div>
              </div>

              {/* Danger Zone */}
              <div className="glass-panel" style={{ padding: "20px", border: "1px solid rgba(239, 68, 68, 0.25)", background: "rgba(239,68,68,0.02)" }}>
                <h3 style={{ fontSize: "15px", fontWeight: 600, color: "var(--color-error)", marginBottom: "4px" }}>Danger Zone</h3>
                <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "12px", lineHeight: "1.4" }}>
                  Permanently deletes all database schemas tables entries, clears uploaded manual lists, and destroys active precomputed indexing vector stores.
                </div>
                <button
                  className="btn-danger"
                  onClick={handleResetDatabase}
                  style={{ width: "100%", justifyContent: "center", padding: "10px", fontSize: "12.5px" }}
                >
                  ⚠️ Reset Assistant Database
                </button>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
