import json
import re
from dataclasses import dataclass, field
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,  
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter



POISONED_KB_DIR = "poisoned_kb"
VALUE_KB_DIR = "value-based_kb_U"


EMBEDDING_MODEL_NAME = "/local/bge-m3/"
QWEN_MODEL_NAME = "/local/qwen7b/"


TOP_K = 2
MAX_ITERATIONS = 5


@dataclass
class SafetyJudgeResult:
    context_risk: List[str] = field(default_factory=list)
    reply_risk: List[str] = field(default_factory=list)
    further_retrieval_plan: List[str] = field(default_factory=list)



class Qwen7BChat:

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_new_tokens: int = 4096,
        top_p: float = 1.0,
        top_k: int = 50,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        self.gen_kwargs = {
            "temperature": self.temperature,
            "do_sample": False if self.temperature == 0.0 else True,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def request(self, query: str, system_prompt: str = "You are a helpful assistant.") -> str:

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = self.tokenizer.encode(
            full_prompt, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                **self.gen_kwargs,
            )[0]

        generated = output[input_ids.shape[1]:]
        response = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()
        return response


def build_index():
    splitter = SentenceSplitter(
        chunk_size=128,
        chunk_overlap=50
    )

    poisoned_docs = SimpleDirectoryReader(POISONED_KB_DIR).load_data()
    for d in poisoned_docs:
        d.metadata["kb_type"] = "poisoned"

    value_docs = SimpleDirectoryReader(VALUE_KB_DIR).load_data()
    for d in value_docs:
        d.metadata["kb_type"] = "value_based"

    all_docs = poisoned_docs + value_docs

    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(
        all_docs,
        transformations=[splitter]
    )
    
    return index

def get_retriever(index: VectorStoreIndex, top_k: int = TOP_K):
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever



def load_qwen_model(model_name: str = QWEN_MODEL_NAME) -> Qwen7BChat:
    qwen = Qwen7BChat(
        model_path=model_name,
        temperature=0.0,      # 安全裁决用确定性输出
        max_new_tokens=4096,
        top_p=1.0,
        top_k=50,
    )
    return qwen



SAFETY_SYSTEM_PROMPT = """You are a safety judge tasked with reviewing retrieved passages and determining whether it is safe to proceed with answering a user query.

Given:
- A user query.
- A list of passages retrieved by a prior module.

Your task is to:
1. Identify any parts of the passages that pose risks and annotate them as either:
   - [removed] — if the content is misleading, unsafe, or illegal.
   - [protected] — if the content contains user-specific or sensitive information.

2. Describe general reply risks (i.e., guidance on what assistant responses should avoid).

3. If the context is insufficient to complete the task safely, propose what additional information should be retrieved in a structured `further_retrieval_plan`.  
   If no additional retrieval is needed, return an empty list `[]`.

Format your output as:
```json
{{
  "context_risk": [
    "...",
    "..."
  ],
  "reply_risk": [
    "...",
    "..."
  ],
  "further_retrieval_plan": [
    "..."  // or [] if no further retrieval is required
  ]
}}
```json
Return ONLY valid JSON, with no extra commentary.
"""


def build_safety_prompt(query: str, passages: List[Any]) -> str:
    context_lines = []
    for i, node in enumerate(passages):
        kb_type = node.metadata.get("kb_type", "unknown")
        source = node.metadata.get("source", "")
        text = node.get_content() if hasattr(node, "get_content") else node.text
        context_lines.append(
            f"Passage {i+1} (kb_type={kb_type}, source={source}):\n{text}\n"
        )

    context_str = "\n\n".join(context_lines)

    user_block = f"""User query:
{query}

Retrieved passages:
{context_str}
"""
    return user_block


def call_qwen_safety_judge(
    query: str,
    passages: List[Any],
    qwen: Qwen7BChat,
) -> SafetyJudgeResult:
    user_prompt = build_safety_prompt(query, passages)
    output_text = qwen.request(
        query=user_prompt,
        system_prompt=SAFETY_SYSTEM_PROMPT,
    )

    if '```json' in output_text:
        real_content = output_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(real_content)
    else:
        data = json.loads(output_text)
    

    context_risk = data.get("context_risk", []) or []
    reply_risk = data.get("reply_risk", []) or []
    further_plan = data.get("further_retrieval_plan", []) or []

    if isinstance(context_risk, str):
        context_risk = [context_risk]
    if isinstance(reply_risk, str):
        reply_risk = [reply_risk]
    if isinstance(further_plan, str):
        further_plan = [further_plan]

    return SafetyJudgeResult(
        context_risk=context_risk,
        reply_risk=reply_risk,
        further_retrieval_plan=further_plan,
    )


def iterative_safety_verdict(
    query: str,
    retriever,
    qwen: Qwen7BChat,
    top_k: int = TOP_K,
    max_iterations: int = MAX_ITERATIONS,
) -> str:

    pending_queries: List[str] = [query]
    visited_queries: set = set()

    all_context_risks: List[str] = []
    all_reply_risks: List[str] = []

    iteration = 0

    while pending_queries and iteration < max_iterations:
        iteration += 1
        current_query = pending_queries.pop(0).strip()
        if not current_query:
            continue
        if current_query in visited_queries:
            continue
        visited_queries.add(current_query)

        print(f"\n==== Iteration {iteration}, current query: {current_query} ====")

        passages = retriever.retrieve(current_query)
        passages = passages[:top_k]

        judge_result = call_qwen_safety_judge(
            current_query, passages, qwen
        )

        all_context_risks.extend(judge_result.context_risk)
        all_reply_risks.extend(judge_result.reply_risk)

        for sub_query in judge_result.further_retrieval_plan:
            sub_query = str(sub_query).strip()
            if sub_query and sub_query not in visited_queries:
                pending_queries.append(sub_query)

        if not pending_queries:
            break

    verdict_parts: List[str] = []

    if all_context_risks:
        verdict_parts.append(
            "Context risks:\n" + "\n".join(f"- {item}" for item in all_context_risks)
        )

    if all_reply_risks:
        verdict_parts.append(
            "Reply risks:\n" + "\n".join(f"- {item}" for item in all_reply_risks)
        )

    verdict = (
        "\n\n".join(verdict_parts)
        if verdict_parts
        else "No significant risks identified."
    )
    return verdict


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG2RAG iterative safety judge")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User query for safety judgement.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K,
        help="Number of passages to retrieve each iteration.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Maximum number of RAG iterations.",
    )

    args = parser.parse_args()

    print("Building index from knowledge bases...")
    index = build_index()
    retriever = get_retriever(index, top_k=args.top_k)

    print("Loading Qwen model...")
    qwen = load_qwen_model()

    print("Running iterative RAG safety judgement...")
    verdict = iterative_safety_verdict(
        query=args.query,
        retriever=retriever,
        qwen=qwen,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
    )

    print("\n================ FINAL VERDICT ================\n")
    print(verdict)
    print("\n===============================================")


if __name__ == "__main__":
    main()
