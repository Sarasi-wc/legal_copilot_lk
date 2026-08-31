import { useState } from "react";
import "./App.css";
import AskQuestion from "./components/AskQuestion";
import UploadPreview from "./components/UploadPreview";
import AnswerDiagnostics from "./components/AnswerDiagnostics";
import CorpusDataset from "./components/CorpusDataset";
import ResearchResults from "./components/ResearchResults";

const TABS = [
  { id: "ask", label: "⚖️ Ask a Question" },
  { id: "upload", label: "📤 Upload & Preview" },
  { id: "diagnostics", label: "📈 Answer Diagnostics" },
  { id: "corpus", label: "📂 Corpus & Dataset" },
  { id: "research", label: "📊 Research Results" },
];

function App() {
  const [activeTab, setActiveTab] = useState("ask");
  const [settings, setSettings] = useState({
    retrievalMethod: "hybrid_rerank",
    topK: 5,
    includeVerification: true,
  });
  const [lastResult, setLastResult] = useState(null);
  const [lastQuestion, setLastQuestion] = useState("");

  const handleAnswered = (result, question) => {
    setLastResult(result);
    setLastQuestion(question);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>⚖️ Sri Lankan Legal Copilot</h1>
        <p className="subtitle">
          AI-powered legal assistant with citation-grounded answers
        </p>
      </header>

      <div className="container">
        <div className="sidebar">
          <div className="settings-panel">
            <h3>⚙️ Settings</h3>

            <div className="setting-group">
              <label>Retrieval Method</label>
              <select
                value={settings.retrievalMethod}
                onChange={(e) =>
                  setSettings({ ...settings, retrievalMethod: e.target.value })
                }
              >
                <option value="hybrid_rerank">Hybrid + Rerank (Best)</option>
                <option value="hybrid">Hybrid</option>
                <option value="dense">Dense Only</option>
                <option value="bm25">BM25 Only</option>
              </select>
            </div>

            <div className="setting-group">
              <label>Top K Results: {settings.topK}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={settings.topK}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    topK: parseInt(e.target.value, 10),
                  })
                }
              />
            </div>

            <div className="setting-group">
              <label>
                <input
                  type="checkbox"
                  checked={settings.includeVerification}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      includeVerification: e.target.checked,
                    })
                  }
                />
                Enable Verification
              </label>
            </div>
          </div>

          <div className="info-panel">
            <h3>ℹ️ About</h3>
            <p>
              This system uses Retrieval-Augmented Generation (RAG) to answer
              legal questions based on Sri Lankan law.
            </p>

            <h4>Features:</h4>
            <ul>
              <li>Hybrid retrieval (BM25 + Dense)</li>
              <li>Citation-grounded answers</li>
              <li>NLI-based verification</li>
              <li>Abstention when uncertain</li>
            </ul>
          </div>

          <div className="warning-panel">
            <h3>⚠️ Disclaimer</h3>
            <p>
              This system provides legal information for educational purposes
              only. It is NOT legal advice. Consult a qualified legal
              professional for specific legal matters.
            </p>
          </div>
        </div>

        <div className="main-content">
          <nav className="tab-bar">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {activeTab === "ask" && (
            <AskQuestion settings={settings} onAnswered={handleAnswered} />
          )}
          {activeTab === "upload" && <UploadPreview />}
          {activeTab === "diagnostics" && (
            <AnswerDiagnostics
              lastResult={lastResult}
              lastQuestion={lastQuestion}
            />
          )}
          {activeTab === "corpus" && <CorpusDataset />}
          {activeTab === "research" && <ResearchResults />}
        </div>
      </div>

      <footer className="footer">
        <p>
          🤖 Powered by Retrieval-Augmented Generation (RAG) | Built with React
        </p>
        <p>⚖️ Sri Lankan Legal Copilot - Educational Research Project</p>
      </footer>
    </div>
  );
}

export default App;
