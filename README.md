# RAG4Drug Multi-Agent Prototype


###

sentence_transformers==3.0.1
torch_geometric==2.7.0
python==3.10.16
torch==2.1.0




This project implements a lightweight code design for the architecture you described:

- **Reasoning Planner**: retrieves top-K reference drugs by structure+KG similarity and chooses symbolic/analogical strategy.
- **Knowledge Retriever**: retrieves and refines topological evidence paths and builds dynamic graph anchors.
- **Biological Annotator**: converts graph paths into traceable biological evidence narratives.
- **Unified Predictor**: fuses molecular/path/semantic embeddings and applies lightweight task-specific prediction heads.
- **Task Router**: dispatches between multi-task settings (currently `DTI` and `DDI`).

## Project Structure

- `src/rag4drug/reasoning_planner.py`: reference retrieval + strategy planning.
- `src/rag4drug/knowledge_retriever.py`: retrieval and path embedding.
- `src/rag4drug/knowledge_graph.py`: PrimeKG-backed graph backend.
- `src/rag4drug/biological_annotator.py`: biological evidence annotation + semantic embedding.
- `src/rag4drug/unified_predictor.py`: fused representation and task heads.
- `src/rag4drug/generator.py`: evidence-constrained generation output.
- `src/rag4drug/pipeline.py`: end-to-end orchestration.
- `src/rag4drug/task_router.py`: task dispatch and task specification resolution.
- `src/rag4drug/tasks.py`: default task registry (`DTI`, `DDI`).
- `examples/run_demo.py`: runnable demonstration.


###Data
##包括PrimeKG和drugbank



###Drugbank
https://github.com/zcc1203/RAG4Drug
##保存路径
RAG4Drug/

###PrimeKG
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
##保存路径
RAG4Drug/data/primekg/

## Quick Start

```bash
pip install -e .
CUDA_VISIBLE_DEVICES=3 python examples/train_rag4drug_5fold.py   --config configs/retrieval.json   --input-dir data/DDI/data/drugbank/cv5_warm_few   --output-dir outputs/rag4drug_all   --folds 5 --epochs 30 --eval-every 10 --batch-size 64 --lr 1e-3 --threshold 0.5   --feature-mode all   --struct-feature-mode both --n-bits 256   --semantic-model-path /home/data/43940/codes/RAG4Drug/v2/RAG4Drug/model/all-MiniLM-L6-v2

```






