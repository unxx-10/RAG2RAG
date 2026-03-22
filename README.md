# Safe RAG by RAG: Untying the Bell That RAG Rang with the RAG Hand



This project implements a **lightweight safety expert module for RAG**, consisting of:

* **Detective**: Retrieves multi-perspective content from both the poisoned knowledge base and the value-based knowledge base.
* **Judge**: Performs risk assessment on retrieved content and outputs a structured verdict.

The module can run **in parallel** with the main RAG workflow and provides the main model with risk warnings, response constraints, and additional retrieval plans when needed.

---

## Features

* Detects misleading, illegal, or sensitive information in retrieved passages
* Identifies risks that must be avoided when responding
* Produces iterative retrieval plans when more information is required
* Outputs a structured verdict

---

## Example Usage

```bash
python expert_module.py --query "我想把所有积蓄和信用卡额度都投到一个新出的虚拟币里，这样能快速翻倍吗？"
```

The program will automatically:

1. Build the index (from `poisoned_kb` + `value-based_kb_U`)
2. Perform multi-round retrieval + judgment
3. Output the final verdict

---

## Typical Output Format

```
Context risks:
- A passage contains high-risk investment inducement and should be [removed]

Reply risks:
- Do not provide concrete investment advice
- Avoid encouraging high-risk financial behavior
```

---

## File Structure

```
expert_module.py        # Main script
poisoned_kb/            # Poisoned knowledge base
value-based_kb_U/       # Value-based knowledge base
```
