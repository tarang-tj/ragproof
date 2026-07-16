"""INTEGRATION STUB -- not run by the offline demo or test suite.

Shows the production seam: how to wire a Supabase/pgvector similarity
search retriever, plus Claude for generation, behind the same Retriever
interface used everywhere else in ragproof (see retrievers.py). This lets
you swap DenseRetriever -> PgVectorClaudeRetriever with zero changes to
eval_runner.py, metrics.py, or drift.py.

Setup (not executed here):
    pip install anthropic supabase voyageai
    export SUPABASE_URL="https://<project>.supabase.co"
    export SUPABASE_SERVICE_KEY="<service-role-key>"   # never commit this
    export ANTHROPIC_API_KEY="<your-key>"
    export VOYAGE_API_KEY="<your-key>"                 # or your embedder of choice

Expected Supabase schema (pgvector extension enabled):
    create table documents (
        id text primary key,
        content text not null,
        embedding vector(1024)  -- match your embedding model's dimension
    );
    create index on documents using ivfflat (embedding vector_cosine_ops);

    -- similarity search RPC, called via supabase-py .rpc(...)
    create or replace function match_documents(
        query_embedding vector(1024),
        match_count int
    ) returns table(id text, content text, similarity float)
    language sql stable as $$
        select id, content, 1 - (embedding <=> query_embedding) as similarity
        from documents
        order by embedding <=> query_embedding
        limit match_count;
    $$;

Fill in your keys via environment variables; nothing here executes network
calls at import time.
"""

from __future__ import annotations

import os
from typing import Any


class PgVectorClaudeRetriever:
    """Retriever backed by Supabase pgvector, generation via Claude.

    Implements the same `retrieve(query, top_k) -> list[str]` /
    `score(query) -> dict[str, float]` interface as SparseRetriever,
    DenseRetriever, and HybridRetriever in retrievers.py, so it drops into
    eval_runner.evaluate_retriever() and compare_retrievers() unchanged.
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        embedding_model: str = "voyage-3",
        match_rpc: str = "match_documents",
    ):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_SERVICE_KEY")
        self.embedding_model = embedding_model
        self.match_rpc = match_rpc

        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "PgVectorClaudeRetriever requires SUPABASE_URL and "
                "SUPABASE_SERVICE_KEY (env vars or constructor args). "
                "This is an integration stub -- fill in your project's keys."
            )

        # Lazy imports: keep these optional deps out of the offline demo path.
        try:
            from supabase import create_client  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pip install supabase to use PgVectorClaudeRetriever"
            ) from exc

        self._client = create_client(self.supabase_url, self.supabase_key)

    def _embed(self, text: str) -> list[float]:
        """Embed text via Voyage AI (or swap for your provider of choice).

        NOT CALLED by the offline demo/tests. Requires VOYAGE_API_KEY.
        """
        try:
            import voyageai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install voyageai to embed queries") from exc

        client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
        result = client.embed([text], model=self.embedding_model, input_type="query")
        return result.embeddings[0]

    def score(self, query: str) -> dict[str, float]:
        """Return cosine similarity per doc_id via the pgvector RPC."""
        query_embedding = self._embed(query)
        response = self._client.rpc(
            self.match_rpc,
            {"query_embedding": query_embedding, "match_count": 1000},
        ).execute()
        return {row["id"]: row["similarity"] for row in response.data}

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Top-k doc_ids via pgvector's ANN index (ivfflat/hnsw cosine search)."""
        query_embedding = self._embed(query)
        response = self._client.rpc(
            self.match_rpc,
            {"query_embedding": query_embedding, "match_count": top_k},
        ).execute()
        return [row["id"] for row in response.data]

    def generate_answer(self, query: str, context_doc_ids: list[str], model: str = "claude-sonnet-4-5") -> str:
        """Generate a grounded answer from retrieved context using Claude.

        NOT CALLED by the offline demo/tests. Requires ANTHROPIC_API_KEY.
        Pair this with cost.estimate_cost() using the actual usage object
        Claude returns (response.usage.input_tokens / output_tokens) to log
        real per-query spend instead of estimates.
        """
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pip install anthropic to generate answers") from exc

        docs = self._fetch_documents(context_doc_ids)
        context = "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in docs.items())

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system="Answer using only the provided context. Cite doc ids you used.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
        )
        return message.content[0].text

    def _fetch_documents(self, doc_ids: list[str]) -> dict[str, str]:
        response = (
            self._client.table("documents").select("id, content").in_("id", doc_ids).execute()
        )
        return {row["id"]: row["content"] for row in response.data}


def build_retriever_from_env() -> PgVectorClaudeRetriever:
    """Convenience factory: builds the retriever purely from env vars.

    Raises RuntimeError with a clear message if required env vars/deps are
    missing -- see the module docstring for the full setup checklist.
    """
    return PgVectorClaudeRetriever()
