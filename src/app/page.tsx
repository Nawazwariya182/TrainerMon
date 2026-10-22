"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  Plus,
  Search,
  FolderPlus,
  Trash2,
  Play,
  Cpu,
  Zap,
  Sparkles,
  Layers,
  Activity,
  CheckCircle2,
  Clock,
  X,
  MessageSquare,
  AlertCircle,
  ExternalLink,
} from "lucide-react";

interface Project {
  id: string;
  name: string;
  description: string;
  model: string;
  taskType: string;
  maxTokens: number;
  contextWindow: number;
  status: string;
  createdAt: string;
}

interface BackendHealth {
  online: boolean;
  engine?: {
    current_model?: string;
    loaded?: boolean;
    load_time_seconds?: number;
  };
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [backendStatus, setBackendStatus] = useState<BackendHealth | null>(null);

  // New Project Form State
  const [name, setName] = useState<string>("");
  const [description, setDescription] = useState<string>("");
  const [model, setModel] = useState<string>("qwen3-8b");
  const [taskType, setTaskType] = useState<string>("Inference");
  const [maxTokens, setMaxTokens] = useState<number>(8192);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  // Quick Run Test Modal State
  const [activeProjectForRun, setActiveProjectForRun] = useState<Project | null>(null);
  const [promptInput, setPromptInput] = useState<string>("Explain the role of FlashAttention in scaling KV cache inference.");
  const [inferencing, setInferencing] = useState<boolean>(false);
  const [inferenceResult, setInferenceResult] = useState<any | null>(null);

  // Fetch projects from backend
  const fetchProjects = async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/projects");
      const data = await res.json();
      if (data.success && Array.isArray(data.projects)) {
        setProjects(data.projects);
      }
    } catch (err) {
      console.error("Failed to fetch projects:", err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch live backend health telemetry
  const fetchBackendHealth = async () => {
    try {
      const res = await fetch("/api/backend-status");
      const data = await res.json();
      setBackendStatus(data);
    } catch {
      setBackendStatus({ online: false });
    }
  };

  useEffect(() => {
    fetchProjects();
    fetchBackendHealth();
    const interval = setInterval(fetchBackendHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setErrorMsg("Please enter a project name.");
      return;
    }
    setSubmitting(true);
    setErrorMsg("");

    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim(),
          model,
          taskType,
          maxTokens: Number(maxTokens) || 8192,
          contextWindow: 16384,
        }),
      });

      const data = await res.json();
      if (data.success && data.project) {
        setProjects((prev) => [data.project, ...prev]);
        setIsModalOpen(false);
        setName("");
        setDescription("");
        setModel("qwen3-8b");
        setMaxTokens(8192);
      } else {
        setErrorMsg(data.error || "Failed to create project");
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Network error");
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteProject = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this project?")) return;
    try {
      const res = await fetch(`/api/projects?id=${id}`, { method: "DELETE" });
      const data = await res.json();
      if (data.success) {
        setProjects((prev) => prev.filter((p) => p.id !== id));
      }
    } catch (err) {
      console.error("Failed to delete project:", err);
    }
  };

  const handleRunInference = async () => {
    if (!activeProjectForRun || !promptInput.trim()) return;
    setInferencing(true);
    setInferenceResult(null);

    try {
      const apiKey = "sk-mltrain-live-engine-98f2b4c107e3a9d2";
      const res = await fetch("http://127.0.0.1:8000/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: activeProjectForRun.model,
          messages: [{ role: "user", content: promptInput.trim() }],
          max_tokens: activeProjectForRun.maxTokens || 8192,
          temperature: 0.7,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Inference server error");
      }

      const data = await res.json();
      setInferenceResult(data);
    } catch (err: any) {
      setInferenceResult({ error: err.message || "Failed to connect to inference server" });
    } finally {
      setInferencing(false);
    }
  };

  const filteredProjects = projects.filter((p) =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.model.toLowerCase().includes(searchQuery.toLowerCase()) ||
    p.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-sm font-semibold">
              <Sparkles className="w-5 h-5 text-indigo-100" />
            </div>
            <div>
              <span className="font-bold text-slate-900 text-lg tracking-tight">ML-Train Studio</span>
              <span className="ml-2 text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                v1.0
              </span>
            </div>
          </div>

          {/* Engine Status Pill */}
          <div className="hidden sm:flex items-center space-x-4">
            <div className="inline-flex items-center space-x-2 text-xs font-medium bg-slate-100 border border-slate-200 px-3 py-1.5 rounded-full text-slate-700">
              <span className={`w-2 h-2 rounded-full ${backendStatus?.online ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`} />
              <span>
                {backendStatus?.online
                  ? backendStatus.engine?.loaded
                    ? `Engine Active • ${backendStatus.engine.current_model || "Ready"}`
                    : "Engine Standby (Auto-offloaded)"
                  : "Engine Initializing (Port 8000)"}
              </span>
            </div>
            <div className="text-xs font-semibold px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-700 border border-indigo-100 flex items-center gap-1">
              <Zap className="w-3.5 h-3.5 text-indigo-500" />
              <span>8K Max Tokens Cap</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Page Hero Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-slate-200 pb-6">
          <div className="space-y-1">
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
              Projects
            </h1>
            <p className="text-sm text-slate-500 max-w-2xl">
              Organize, launch, and benchmark your machine learning pipelines. Optimized for fast inference and up to 8,192 maximum output tokens.
            </p>
          </div>

          {/* Primary Action Button: New Project */}
          <Link
            href="/projects/new"
            className="inline-flex items-center justify-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm shadow-sm transition-all duration-150 active:scale-95 shrink-0 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Project</span>
          </Link>
        </div>

        {/* Filter & Search Toolbar (Only visible when projects exist) */}
        {projects.length > 0 && (
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="relative w-full sm:w-80">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder="Search projects or models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-shadow"
              />
            </div>
            <div className="text-xs font-medium text-slate-500 self-end sm:self-center">
              Showing {filteredProjects.length} of {projects.length} project{projects.length !== 1 ? "s" : ""}
            </div>
          </div>
        )}

        {/* Project Content Area */}
        {loading ? (
          <div className="min-h-[360px] flex items-center justify-center">
            <div className="flex flex-col items-center space-y-3">
              <div className="w-8 h-8 border-3 border-indigo-600 border-t-transparent rounded-full animate-spin" />
              <p className="text-sm text-slate-500">Loading projects...</p>
            </div>
          </div>
        ) : projects.length === 0 ? (
          /* Initial Empty State: ZERO cards */
          <div className="min-h-[420px] rounded-2xl border-2 border-dashed border-slate-200 bg-white p-12 text-center flex flex-col items-center justify-center space-y-5 shadow-xs">
            <div className="w-16 h-16 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center shadow-inner">
              <FolderPlus className="w-8 h-8" />
            </div>
            <div className="space-y-1.5 max-w-md">
              <h3 className="text-xl font-bold text-slate-900 tracking-tight">
                No Projects Yet
              </h3>
              <p className="text-sm text-slate-500 leading-relaxed">
                You do not have any machine learning projects created yet. Start by creating a project to execute inference with up to 8k output tokens.
              </p>
            </div>
            <Link
              href="/projects/new"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-sm shadow-sm transition-all duration-150 active:scale-95"
            >
              <Plus className="w-4 h-4" />
              <span>New Project</span>
            </Link>
          </div>
        ) : filteredProjects.length === 0 ? (
          /* Search Empty State */
          <div className="py-16 text-center space-y-3">
            <AlertCircle className="w-10 h-10 text-slate-400 mx-auto" />
            <p className="text-base font-semibold text-slate-700">No matching projects found</p>
            <p className="text-xs text-slate-500">Try adjusting your search query.</p>
          </div>
        ) : (
          /* Project Cards Grid */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProjects.map((p) => (
              <div
                key={p.id}
                className="bg-white rounded-2xl border border-slate-200/90 p-6 flex flex-col justify-between hover:shadow-md hover:border-slate-300 transition-all duration-200 group relative"
              >
                <div className="space-y-4">
                  {/* Card Header */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <h3 className="text-base font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
                        {p.name}
                      </h3>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100">
                          {p.model}
                        </span>
                        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-100">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          {p.status}
                        </span>
                      </div>
                    </div>

                    <button
                      onClick={(e) => handleDeleteProject(p.id, e)}
                      title="Delete Project"
                      className="text-slate-400 hover:text-rose-600 p-1.5 rounded-lg hover:bg-rose-50 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Description */}
                  <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed">
                    {p.description}
                  </p>

                  {/* Specifications Grid */}
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-100 text-xs">
                    <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block text-[10px] uppercase font-bold tracking-wider">
                        Generation Cap
                      </span>
                      <span className="font-semibold text-slate-800">
                        {p.maxTokens.toLocaleString()} tokens
                      </span>
                    </div>
                    <div className="bg-slate-50 p-2 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block text-[10px] uppercase font-bold tracking-wider">
                        Task Architecture
                      </span>
                      <span className="font-semibold text-slate-800">{p.taskType}</span>
                    </div>
                  </div>
                </div>

                {/* Card Footer */}
                <div className="pt-4 mt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                  <div className="flex items-center space-x-1">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    <span>{new Date(p.createdAt).toLocaleDateString()}</span>
                  </div>

                  <button
                    onClick={() => {
                      setActiveProjectForRun(p);
                      setInferenceResult(null);
                    }}
                    className="inline-flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-indigo-600 text-white font-medium text-xs transition-colors shadow-xs"
                  >
                    <Play className="w-3 h-3" />
                    <span>Run Test</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Modal: Create New Project */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-lg w-full overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <FolderPlus className="w-4 h-4" />
                </div>
                <h3 className="font-bold text-slate-900 text-base">Create New Project</h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-md"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateProject} className="p-6 space-y-4">
              {errorMsg && (
                <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 shrink-0" />
                  <span>{errorMsg}</span>
                </div>
              )}

              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-700">Project Name *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Qwen3 Reasoning & Code Synthesis"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-700">Description</label>
                <textarea
                  rows={2}
                  placeholder="Describe your workload, prompts, or experimentation goals..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700">Base Model</label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="qwen3-8b">Qwen3-8B-Q8_0 (Code & Reasoning)</option>
                    <option value="qwen3.5-9b">Qwen3.5-9B-Q8_0 (Extended Context)</option>
                    <option value="qwen3-vl-8b">Qwen3-VL-8B-Instruct (Vision-Language)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700">Task Type</label>
                  <select
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value)}
                    className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="Inference">High-Speed Inference</option>
                    <option value="Pretraining">Training from Scratch</option>
                    <option value="Benchmarking">Context Benchmarking</option>
                    <option value="Vision-Language">Vision Analysis</option>
                  </select>
                </div>
              </div>

              <div className="space-y-1.5 bg-indigo-50/60 p-3.5 rounded-xl border border-indigo-100">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-indigo-900 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-indigo-600" />
                    <span>Max Output Token Cap: {maxTokens.toLocaleString()} Tokens (8K)</span>
                  </label>
                </div>
                <input
                  type="range"
                  min={1024}
                  max={8192}
                  step={512}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-indigo-700 font-medium">
                  <span>1,024</span>
                  <span>4,096</span>
                  <span className="font-bold">8,192 (Max Cap)</span>
                </div>
              </div>

              <div className="pt-2 flex items-center justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 rounded-lg border border-slate-300 text-xs font-medium text-slate-700 hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-xs disabled:opacity-50 transition-colors"
                >
                  {submitting ? "Creating..." : "Create Project"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Modal: Interactive Model Inference Runner */}
      {activeProjectForRun && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
          <div className="bg-white rounded-2xl border border-slate-200 shadow-xl max-w-2xl w-full max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <div>
                <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
                  <Play className="w-4 h-4 text-indigo-600" />
                  <span>Execute Model Test: {activeProjectForRun.name}</span>
                </h3>
                <p className="text-xs text-slate-500">
                  Target Model: <span className="font-semibold text-slate-700">{activeProjectForRun.model}</span> • Cap:{" "}
                  <span className="font-semibold text-indigo-700">{activeProjectForRun.maxTokens} Tokens</span>
                </p>
              </div>
              <button
                onClick={() => setActiveProjectForRun(null)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-md"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-4 overflow-y-auto flex-1">
              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-700">Prompt Input</label>
                <textarea
                  rows={3}
                  value={promptInput}
                  onChange={(e) => setPromptInput(e.target.value)}
                  placeholder="Enter a prompt to test generation..."
                  className="w-full px-3 py-2 bg-white border border-slate-300 rounded-lg text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="flex justify-end">
                <button
                  onClick={handleRunInference}
                  disabled={inferencing}
                  className="inline-flex items-center space-x-2 px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs shadow-xs transition-colors disabled:opacity-50"
                >
                  {inferencing ? (
                    <>
                      <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Generating response...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-3.5 h-3.5" />
                      <span>Execute Inference</span>
                    </>
                  )}
                </button>
              </div>

              {/* Inference Result Box */}
              {inferenceResult && (
                <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
                  {inferenceResult.error ? (
                    <div className="text-rose-600 text-xs font-medium">
                      ❌ {inferenceResult.error}
                    </div>
                  ) : (
                    <>
                      <div className="flex items-center justify-between text-xs font-semibold pb-2 border-b border-slate-200">
                        <span className="text-emerald-700 flex items-center gap-1">
                          <CheckCircle2 className="w-4 h-4" /> Generation Complete
                        </span>
                        <div className="flex items-center gap-3 text-slate-600">
                          <span>Tokens: {inferenceResult.usage?.completion_tokens || "N/A"}</span>
                          <span>Speed: {inferenceResult.inference_metrics?.tokens_per_second || "N/A"} tok/s</span>
                        </div>
                      </div>
                      <div className="text-xs text-slate-800 font-mono whitespace-pre-wrap max-h-60 overflow-y-auto bg-white p-3 rounded-lg border border-slate-200">
                        {inferenceResult.choices?.[0]?.message?.content || "No output returned"}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
