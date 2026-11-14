import json
import math
from urllib.parse import unquote, urlparse

import pg8000.dbapi

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int):
        if dimensions != 1536:
            raise ValueError("This lab schema expects 1536 dimensions")

        url = urlparse(database_url)
        self.dimensions = dimensions
        self.settings = {
            "user": unquote(url.username or ""),
            "password": unquote(url.password or ""),
            "host": url.hostname,
            "port": url.port or 5432,
            "database": url.path.lstrip("/"),
            "timeout": 30,
        }

    def connect(self):
        return pg8000.dbapi.connect(**self.settings)

    def vector_text(self, vector):
        if len(vector) != self.dimensions:
            raise ValueError("Incorrect vector dimensions")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Invalid vector values")
        if not any(value != 0 for value in vector):
            raise ValueError("Zero vector cannot be used for cosine search")
        return json.dumps(vector)

    def create_schema(self):
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS day6_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR(1536) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
            """)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ):
        """Replace the complete contents of one lab strategy atomically."""
        if strategy not in {"A", "B"}:
            raise ValueError("Only A and B have physical storage")
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Chunks/vectors must be nonempty and aligned")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs in input")

        rows = []
        for chunk, vector in zip(chunks, vectors):
            rows.append((
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                json.dumps(chunk.metadata, ensure_ascii=False),
                self.vector_text(vector),
            ))

        conn = self.connect()
        try:
            cursor = conn.cursor()

            # Chỉ thay dữ liệu của strategy đang nạp trong bảng lab này.
            # Loại luôn chunk cũ không còn trong phiên bản mới.
            cursor.execute(
                "DELETE FROM day6_chunks WHERE strategy = %s",
                (strategy,),
            )
            cursor.executemany("""
                INSERT INTO day6_chunks (
                    strategy, chunk_id, document_id, source,
                    page, text, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            """, rows)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def snapshot(self, strategy):
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT count(*),
                       md5(string_agg(chunk_id, ',' ORDER BY chunk_id))
                FROM day6_chunks
                WHERE strategy = %s
            """, (strategy,))
            count, fingerprint = cursor.fetchone()
            return {"count": count, "chunk_ids_md5": fingerprint}
        finally:
            conn.close()

    def search(
        self,
        strategy,
        query_vector,
        top_k=3,
        filters=None,
    ):
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Unknown strategy")
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("Invalid top_k")
        if filters and strategy != "C":
            raise ValueError("Only C applies benchmark metadata filters")

        physical_strategy = "B" if strategy == "C" else strategy
        containment = {}

        if filters:
            if set(filters) != {"sections"}:
                raise ValueError("Unsupported filter field")
            section_filter = filters["sections"]
            if (
                not isinstance(section_filter, dict)
                or set(section_filter) != {"contains"}
                or not isinstance(section_filter["contains"], str)
            ):
                raise ValueError("Invalid sections filter")
            containment = {"sections": [section_filter["contains"]]}

        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, document_id, source, page,
                       text, metadata,
                       embedding <=> %s::vector AS distance
                FROM day6_chunks
                WHERE strategy = %s
                  AND metadata @> %s::jsonb
                ORDER BY distance, chunk_id
                LIMIT %s
            """, (
                self.vector_text(query_vector),
                physical_strategy,
                json.dumps(containment),
                top_k,
            ))

            hits = []
            for row in cursor.fetchall():
                metadata = row[5]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                hits.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source": row[2],
                    "page": row[3],
                    "text": row[4],
                    "metadata": metadata,
                    "distance": float(row[6]),
                    "score": 1.0 - float(row[6]),
                })
            return hits
        finally:
            conn.close()