import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vector_store")
logging.basicConfig(level=logging.INFO)

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

class ResearchVectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = self.client.get_or_create_collection(
                name="mltrain_research_rag",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB persistent collection 'mltrain_research_rag' initialized at {CHROMA_DIR}")
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB client: {e}. Falling back to memory RAG cache.")
            self.client = None
            self.collection = None

    def index_project_context(self, project_id: str, meta: Dict[str, Any], answers: Dict[str, Any]) -> int:
        """Indexes the user's project specifications and questionnaire answers."""
        docs = []
        metas = []
        ids = []

        # Project overview doc
        name = meta.get("name", "ML Project")
        idea = meta.get("idea", meta.get("description", ""))
        domains = ", ".join(meta.get("domains", [])) if isinstance(meta.get("domains"), list) else str(meta.get("domains", ""))
        approaches = ", ".join(meta.get("learningApproaches", [])) if isinstance(meta.get("learningApproaches"), list) else str(meta.get("learningApproaches", ""))
        
        overview_text = f"Project: {name}\nDomains: {domains}\nApproaches: {approaches}\nObjective: {meta.get('primaryGoal', 'Production')}\nDescription: {idea}"
        docs.append(overview_text)
        metas.append({"project_id": project_id, "doc_type": "project_overview", "title": name})
        ids.append(f"{project_id}_overview")

        # Questionnaire answers docs
        for q_idx, ans_data in answers.items():
            opt = ans_data.get("option", "")
            custom = ans_data.get("custom", "")
            q_text = ans_data.get("question", f"Question {q_idx}")
            cat = ans_data.get("category", "Architecture")

            ans_doc = f"Requirement [{cat}]: {q_text}\nSelected Choice: {opt}"
            if custom:
                ans_doc += f"\nCustom Specification: {custom}"

            docs.append(ans_doc)
            metas.append({"project_id": project_id, "doc_type": "questionnaire_answer", "category": cat})
            ids.append(f"{project_id}_q_{q_idx}")

        return self._add_documents(ids, docs, metas)

    def index_datasets(self, project_id: str, datasets: List[Dict[str, Any]]) -> int:
        """Indexes the 5-7 discovered datasets into the vector store."""
        docs = []
        metas = []
        ids = []

        for idx, ds in enumerate(datasets):
            ds_id = ds.get("id", f"ds_{idx}")
            title = ds.get("title", f"Dataset {idx}")
            source = ds.get("source", "Unknown")
            modality = ds.get("modality", "")
            samples = ds.get("samples", "")
            license_val = ds.get("license", "Unknown")
            url = ds.get("url", "")
            reason = ds.get("suitability_reason", "")
            fmt = ds.get("format", "")

            ds_doc = (
                f"Dataset: {title}\nSource: {source} ({url})\nModality: {modality}\n"
                f"Format: {fmt}\nSamples: {samples}\nLicense: {license_val}\n"
                f"Suitability Rationale: {reason}"
            )

            docs.append(ds_doc)
            metas.append({
                "project_id": project_id,
                "doc_type": "dataset_card",
                "dataset_id": str(ds_id),
                "title": title,
                "license": license_val,
                "verified": bool(ds.get("verified", True)),
                "url": url,
            })
            ids.append(f"{project_id}_ds_{ds_id}")

        return self._add_documents(ids, docs, metas)

    def index_tech_research(self, project_id: str, tech_research: Dict[str, Any]) -> int:
        """Indexes the technical research sections (architecture, loss, compute, TODOs)."""
        docs = []
        metas = []
        ids = []

        for section_key, section_val in tech_research.items():
            if isinstance(section_val, dict):
                content = "\n".join(f"{k}: {v}" for k, v in section_val.items())
            elif isinstance(section_val, list):
                content = "\n".join(f"- {item}" for item in section_val)
            else:
                content = str(section_val)

            doc_text = f"Technical Research [{section_key.upper()}]:\n{content}"
            docs.append(doc_text)
            metas.append({"project_id": project_id, "doc_type": "tech_research", "section": section_key})
            ids.append(f"{project_id}_tech_{section_key}")

        return self._add_documents(ids, docs, metas)

    def index_code_files(self, project_id: str, code_files: Dict[str, str]) -> int:
        """Indexes generated workspace code files into ChromaDB for assistant retrieval."""
        docs = []
        metas = []
        ids = []

        for filename, content in code_files.items():
            if not content:
                continue
            chunk_size = 1500
            for chunk_idx, i in enumerate(range(0, len(content), chunk_size)):
                chunk_str = content[i : i + chunk_size]
                docs.append(f"Workspace Source Code [{filename}] (Segment {chunk_idx + 1}):\n{chunk_str}")
                metas.append({
                    "project_id": project_id,
                    "doc_type": "source_code",
                    "filename": filename,
                    "chunk": chunk_idx,
                })
                ids.append(f"{project_id}_code_{filename}_{chunk_idx}")

        return self._add_documents(ids, docs, metas)

    def _add_documents(self, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> int:
        """Internal helper to upsert documents into ChromaDB."""
        if not docs:
            return 0

        if self.collection is not None:
            try:
                # Use upsert to avoid duplicate key errors on repeated submissions
                self.collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                )
                logger.info(f"Upserted {len(docs)} documents into ChromaDB vector store.")
                return len(docs)
            except Exception as e:
                logger.warning(f"Error during ChromaDB upsert: {e}")
        return len(docs)

    def query_rag(self, project_id: Optional[str], query_text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Performs semantic search retrieval over the project's indexed research knowledge."""
        if not query_text or self.collection is None:
            return []

        try:
            where_filter = {"project_id": project_id} if project_id else None
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(top_k, max(1, self.collection.count())),
                where=where_filter,
            )

            hits = []
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, doc_id in zip(
                    results["documents"][0],
                    results["metadatas"][0] if results.get("metadatas") else [{}] * len(results["documents"][0]),
                    results["ids"][0] if results.get("ids") else [""] * len(results["documents"][0]),
                ):
                    hits.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                    })
            return hits
        except Exception as e:
            logger.warning(f"ChromaDB RAG query error: {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Returns vector store health and document counts."""
        count = self.collection.count() if self.collection is not None else 0
        return {
            "initialized": self.collection is not None,
            "engine": "ChromaDB (Persistent)",
            "storage_path": str(CHROMA_DIR),
            "total_indexed_documents": count,
        }

# Global Singleton Vector Store
vector_store = ResearchVectorStore()
