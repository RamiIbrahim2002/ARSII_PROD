# core/ocr.py
import os
import json
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from fastapi import HTTPException, UploadFile
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from config import get_openai_client, logger
from core.schema import CNAM_FORM_SCHEMA, CLAIM_EVAL_SCHEMA
from core.prompts import OCR_SYSTEM_PROMPT, EVAL_SYSTEM_PROMPT, REPORT_SYSTEM_PROMPTS
from utils.image import get_blank_form_b64, encode_file_to_b64

class OcrProcessor:
    def __init__(self):
        self.client = get_openai_client()
        self.blank_form_b64 = get_blank_form_b64()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        
        # Initialize ChromaDB for RAG capabilities
        if self.openai_api_key:
            try:
                self.chroma_client = PersistentClient(path="data/chroma")
                self.embedding_function = OpenAIEmbeddingFunction(
                    api_key=self.openai_api_key,
                    model_name="text-embedding-3-small"
                )
                self.ocr_collection = self.chroma_client.get_or_create_collection(
                    name="ocr_documents",
                    embedding_function=self.embedding_function
                )
                logger.info("ChromaDB initialized successfully for OCR RAG")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB: {e}")
                self.chroma_client = None
                self.ocr_collection = None
        else:
            logger.warning("OpenAI API key not found - RAG features will be limited")
            self.chroma_client = None
            self.ocr_collection = None
            
        if self.blank_form_b64 is None:
            logger.error("Failed to load blank form template")
    
    async def process_form(self, filled_form: UploadFile):
        """Process a filled form and extract information using OCR."""
        request_id = os.urandom(4).hex()
        logger.info(f"[{request_id}] Extracting {filled_form.filename}")
        
        if self.blank_form_b64 is None:
            raise HTTPException(status_code=500, detail="Blank form template missing.")
        
        # Prepare the input images
        inputs = [{"type": "input_image", "image_url": f"data:image/png;base64,{self.blank_form_b64}"}]
        filled_data = await filled_form.read()
        filled_b64 = await encode_file_to_b64(filled_data)
        inputs.append({"type": "input_image", "image_url": f"data:image/png;base64,{filled_b64}"})
        
        try:
            # Send to OpenAI for OCR processing
            resp = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {"role": "user", "content": inputs}
                ],
                text={"format": {"type": "json_schema", "name": "cnam_form", "schema": CNAM_FORM_SCHEMA, "strict": True}}
            )
        except Exception as e:
            logger.error(f"[{request_id}] OpenAI error: {e}")
            raise HTTPException(status_code=500, detail="OCR service error.")
        
        try:
            result = json.loads(resp.output_text)
            logger.info(f"[{request_id}] Successfully extracted data from form")
            return filled_form.filename, result
        except Exception as e:
            logger.error(f"[{request_id}] JSON parse error: {e}")
            raise HTTPException(status_code=500, detail="Invalid OCR output.")
        
    async def evaluate_claim(self, extracted: dict) -> dict:
        """Ask OpenAI to run through reimbursement questions with structured JSON output."""
        # 1) JSON‑encode for the LLM
        body = json.dumps(extracted, ensure_ascii=False)
        
        try:
            resp = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system",  "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user",    "content": body}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "claim_evaluation",
                        "schema": CLAIM_EVAL_SCHEMA,
                        "strict": True
                    }
                }
            )
        except Exception as e:
            logger.error(f"Evaluation call failed: {e}")
            raise HTTPException(status_code=500, detail="Evaluation service error.")
        
        try:
            evaluation = json.loads(resp.output_text)
        except Exception as e:
            logger.error(f"Invalid evaluation JSON: {e}")
            raise HTTPException(status_code=500, detail="Invalid evaluation output.")
        
        logger.info(f"Claim evaluation result: {evaluation}")
        return evaluation

    async def generate_rag_report(
        self,
        documents: List[Dict[str, Any]],
        report_type: str = "summary"
    ) -> str:
        if not documents:
            return "No documents available to generate report."

        if not self.ocr_collection:
            logger.warning("ChromaDB not initialized: falling back to direct report.")
            return await self._generate_direct_report(documents, report_type)

        # 1) Sanitize inputs
        inputs: List[str] = []
        for idx, doc in enumerate(documents):
            raw = doc.get("content")
            if not isinstance(raw, str):
                logger.error(f"[RAG] raw content #{idx} is {type(raw)}")
                raw = json.dumps(raw, ensure_ascii=False) if isinstance(raw, (dict, list)) else str(raw)
            inputs.append(raw)

        # 2) Chunk them
        ids, docs_chunks, metas = [], [], []
        for i, text in enumerate(inputs):
            chunks = self._text_to_chunks(text, 1000, 200)
            for j, chunk in enumerate(chunks):
                ids.append(f"doc_{i}_chunk_{j}")
                docs_chunks.append(chunk)
                m = documents[i].get("metadata", {}).copy()
                m.update({
                    "original_doc_index": i,
                    "chunk_index": j,
                    "total_chunks": len(chunks)
                })
                metas.append(m)

        # **NEW**: Log how many chunks we ended up with
        total = len(docs_chunks)
        logger.info(f"[RAG] embedding total of {total} chunks")

        # 3) Embed in batches
        all_vectors = []
        batch_size = 50
        for start in range(0, total, batch_size):
            batch = docs_chunks[start:start+batch_size]
            logger.info(f"[RAG] embedding batch {start}–{start+len(batch)-1}")
            # right before your embeddings.create(...)
            for idx, chunk in enumerate(docs_chunks):
                try:
                    # embed a single‑item list
                    self.client.embeddings.create(
                        model=self.embed_model,
                        input=[chunk]
                    )
                    logger.info(f"[RAG] chunk #{idx} OK")
                except Exception as e:
                    logger.error(f"[RAG] chunk #{idx} FAILED: {e}")
                    logger.error(f"[RAG] >>> {chunk!r}")
                    raise  # stop here
            try:
                resp = self.client.embeddings.create(
                    model=self.embed_model,
                    input=batch
                )
                vecs = [d.embedding for d in resp.data]
                all_vectors.extend(vecs)
                logger.info(f"[RAG] batch {start}–{start+len(batch)-1} succeeded ({len(vecs)} vectors)")
            except Exception as e:
                logger.error(f"[RAG] embeddings.create failed on batch {start}: {e}")
                # Spot‑check that every item in `batch` really is a str
                for k, chk in enumerate(batch):
                    if not isinstance(chk, str):
                        logger.error(f"[RAG] BAD chunk in batch {start}+{k}: {type(chk)}")
                raise

        # 4) Ingest with those vectors
        temp_name = f"temp_report_{datetime.now().timestamp()}"
        temp_col = self.chroma_client.get_or_create_collection(
            name=temp_name,
            embedding_function=self.ocr_collection.embedding_function
        )
        temp_col.add(
            ids=ids,
            documents=docs_chunks,
            metadatas=metas,
            embeddings=all_vectors
        )

        # 5) Your existing retrieval & summarization code…
        queries = self._generate_report_queries(report_type)
        contexts = []
        for q in queries:
            res = temp_col.query(
                query_texts=[q],
                n_results=5,
                include=["documents","metadatas","distances"]
            )
            ctx = f"Query: {q}\n\n"
            for txt, md, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                rel = 1.0 - min(dist, 1.0)
                ctx += (
                    f"Doc {md['original_doc_index']} (chunk {md['chunk_index']}, "
                    f"type {md.get('document_type')}, date {md.get('timestamp')}, rel {rel:.2f}):\n"
                    f"{txt}\n\n"
                )
            contexts.append(ctx)

        # 6) Final chat completion (synchronous)
        stats = self._compute_report_stats(documents)
        system_prompt = REPORT_SYSTEM_PROMPTS.get(report_type, REPORT_SYSTEM_PROMPTS["default"])
        user_msg = f"Generate a {report_type} report with these stats:\n{stats}\n\n{''.join(contexts)}"

        try:
            summary = self.client.chat.completions.create(
                model=self.embed_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=0.2
            )
            return summary.choices[0].message.content
        finally:
            # clean up
            try:
                self.chroma_client.delete_collection(temp_name)
            except Exception as e:
                logger.error(f"Error deleting temp collection: {e}")


    def _text_to_chunks(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks of approximately chunk_size characters."""
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end with overlap
            end = min(start + chunk_size, text_length)
            
            # If we're not at the end of the text, try to find a good split point
            if end < text_length:
                # Look for sentence-ending punctuation
                for i in range(min(end + 50, text_length) - 1, max(end - 50, start) - 1, -1):
                    if i < text_length and text[i] in '.!?':
                        end = i + 1
                        break
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move the start position for the next chunk, with overlap
            start = max(end - overlap, start + 1)
            
            # If we couldn't move forward, prevent infinite loop
            if start >= end:
                start = end
        
        return chunks
    def _generate_report_queries(self, report_type: str) -> List[str]:
        """Generate queries for the given report type."""
        if report_type == "summary":
            return [
                "What are the main document types in the data?",
                "What are the key pieces of information across all documents?",
                "What common patterns or themes exist in the documents?",
                "What are the most notable insights from the data?"
            ]
        elif report_type == "analysis":
            return [
                "How would you categorize the documents in this dataset?",
                "What are the key data points and their relationships?",
                "Are there any anomalies or special cases in the documents?",
                "What is the quality of the OCR data and confidence levels?",
                "What deeper insights can be derived from analyzing the documents?"
            ]
        elif report_type == "trends":
            return [
                "How have document contents changed over time?",
                "What quantitative trends are visible in the data?",
                "What qualitative shifts are noticeable in document content?",
                "What recommendations would you make based on observed trends?",
                "Are there any seasonal or periodic patterns in the data?"
            ]
        else:
            return [
                "What are the main types of documents?",
                "What key information is contained in the documents?",
                "What insights can be derived from the documents?",
                "What is the overall quality of the document data?"
            ]

    async def generate_rag_report(
        self,
        documents: List[Dict[str, Any]],
        report_type: str = "summary"
    ) -> str:
        """
        Generate a report using RAG techniques from OCR-extracted documents.
        Concatenates all document contents into a single string for embedding.
        """
        if not documents:
            return "No documents available to generate report."

        # 1) Concatenate all document contents
        full_text = "\n\n---\n\n".join(doc["content"] for doc in documents)
        logger.info(f"[RAG] concatenated full_text length={len(full_text)}")

        # 2) Embed using ChromaDB's embedding_function
        try:
            embeddings = self.embedding_function([full_text])
            vector = embeddings[0]  # type: ignore
            logger.info(f"[RAG] full_text embed OK, vector length={len(vector)}")
        except Exception as e:
            logger.error(f"[RAG] full_text embedding failed: {e}")
            return f"Error embedding document text: {e}"

        # 3) Create temporary collection and add the concatenated document
        temp_name = f"temp_report_{datetime.now().timestamp()}"
        temp_col = self.chroma_client.get_or_create_collection(
            name=temp_name,
            embedding_function=self.embedding_function
        )
        temp_col.add(
            ids=["full_doc"],
            documents=[full_text],
            metadatas=[{"document_count": len(documents), "report_type": report_type}],
            embeddings=[vector]
        )

        # 4) Retrieve contexts for each query
        queries = self._generate_report_queries(report_type)
        retrieved = []
        for q in queries:
            res = temp_col.query(
                query_texts=[q],
                n_results=1,
                include=["documents", "distances"]
            )
            doc = res["documents"][0][0]
            dist = res["distances"][0][0]
            rel = 1.0 - min(dist, 1.0)
            retrieved.append(f"Query: {q}\nDocument (relevance={rel:.2f}):\n{doc}\n")

        # 5) Compute basic statistics
        stats = (
            f"Report Statistics:\n"
            f"- Document count: {len(documents)}\n"
            f"- Text length: {len(full_text)} chars"
        )

        # 6) Assemble prompts and call chat completion
        system = REPORT_SYSTEM_PROMPTS.get(report_type, REPORT_SYSTEM_PROMPTS["default"])
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": stats + "\n\n" + "\n".join(retrieved)}
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )
            report = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"[RAG] chat completion failed: {e}")
            report = f"Error generating report: {e}"

        # 7) Clean up temporary collection
        try:
            self.chroma_client.delete_collection(temp_name)
        except Exception as e:
            logger.error(f"Error deleting temp collection: {e}")

        return report


    async def _generate_direct_report(self, documents: List[Dict], report_type: str) -> str:
        """
        Fallback method to generate a report directly using OpenAI without RAG
        when ChromaDB is not available.
        """
        # Get system prompt based on report type
        system_prompt = REPORT_SYSTEM_PROMPTS.get(report_type, REPORT_SYSTEM_PROMPTS["default"])
        
        # Format document data
        doc_summaries = []
        for i, doc in enumerate(documents[:10]):  # Limit to 10 documents
            content = doc["content"]
            metadata = doc.get("metadata", {})
            
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)[:500]  # Truncate long content
            else:
                content_str = str(content)[:500]
                
            doc_summary = f"Document {i+1}:\n"
            for key, value in metadata.items():
                doc_summary += f"- {key}: {value}\n"
            doc_summary += f"- Content: {content_str}...\n\n"
            doc_summaries.append(doc_summary)
            
        context = f"""
        Documents to analyze ({len(documents)} total, showing first {min(10, len(documents))}):
        
        {''.join(doc_summaries)}
        
        Generate a {report_type} report based on these documents.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return f"Error generating report: {str(e)}"