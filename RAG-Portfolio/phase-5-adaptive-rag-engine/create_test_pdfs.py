"""Generate test PDFs for the Adaptive RAG Engine."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

OUT = Path(__file__).parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=12)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=8)


def make_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    doc = SimpleDocTemplate(str(OUT / filename), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = [Paragraph(title, h1), Spacer(1, 0.4*cm)]
    for heading, text in sections:
        story.append(Paragraph(heading, h2))
        for para in text.strip().split("\n\n"):
            story.append(Paragraph(para.strip(), body))
        story.append(Spacer(1, 0.3*cm))
    doc.build(story)
    print(f"Created: {filename}")


# ── PDF 1: Transformer Architecture ──────────────────────────────────────────
make_pdf("transformers.pdf", "Transformer Architecture and Attention Mechanisms", [
    ("Introduction", """
The Transformer architecture, introduced by Vaswani et al. in the 2017 paper 'Attention Is All You Need',
revolutionised natural language processing by replacing recurrent networks with a pure attention-based design.
Unlike RNNs which process tokens sequentially, Transformers process all tokens in parallel, enabling much faster
training and better capture of long-range dependencies.

The key innovation is the self-attention mechanism, which allows every token in a sequence to attend to every
other token, computing a weighted representation based on relevance. This enables the model to capture complex
relationships regardless of distance in the sequence.
"""),
    ("Self-Attention Mechanism", """
Self-attention computes three vectors for each token: Query (Q), Key (K), and Value (V). These are linear
projections of the input embeddings. The attention score between two tokens is computed as the dot product of
their Query and Key vectors, scaled by the square root of the dimension to prevent gradient vanishing.

The formula is: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Where d_k is the dimension of the key vectors. The softmax converts scores to probabilities, and the output
is a weighted sum of Value vectors. This allows the model to focus on relevant parts of the sequence.
"""),
    ("Multi-Head Attention", """
Multi-head attention runs self-attention multiple times in parallel with different learned projections. Each
'head' can attend to different aspects of the input — one head might focus on syntactic relationships while
another captures semantic similarity. The outputs of all heads are concatenated and projected back to the
model dimension.

With h heads and model dimension d_model, each head operates on dimension d_k = d_model / h. This allows
the model to jointly attend to information from different representation subspaces at different positions.
"""),
    ("Positional Encoding", """
Since Transformers process all tokens in parallel without inherent order, positional information must be
explicitly injected. The original paper uses sinusoidal positional encodings added to the input embeddings.
For position pos and dimension i: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)).

Modern models like BERT and GPT use learned positional embeddings instead. Relative positional encodings
(used in T5, RoPE) encode the relative distance between tokens rather than absolute positions, showing
better generalisation to sequences longer than those seen during training.
"""),
    ("BERT and GPT Variants", """
BERT (Bidirectional Encoder Representations from Transformers) uses a masked language modelling objective,
predicting randomly masked tokens using context from both directions. This bidirectional training makes BERT
excellent for understanding tasks like question answering and classification.

GPT (Generative Pre-trained Transformer) uses a causal (left-to-right) language modelling objective,
predicting the next token given previous tokens. This autoregressive design makes GPT naturally suited for
text generation. GPT-3 (175B parameters) and GPT-4 demonstrated emergent capabilities at scale including
few-shot learning, code generation, and complex reasoning.
"""),
    ("Scaling Laws", """
Neural scaling laws describe how model performance improves predictably with compute, data, and parameters.
The Chinchilla scaling laws (Hoffmann et al., 2022) showed that optimal training requires roughly 20 tokens
per parameter. A 70B parameter model should be trained on approximately 1.4 trillion tokens.

Larger models are not always better per unit of compute — efficient training of smaller models on more data
often outperforms larger models undertrained. This insight drove models like Llama 2 (trained on 2 trillion
tokens) and Mistral (using sliding window attention for efficient long-context processing).
"""),
])

# ── PDF 2: Vector Databases and Embeddings ───────────────────────────────────
make_pdf("vector_databases.pdf", "Vector Databases and Embedding Models", [
    ("What Are Vector Databases", """
Vector databases store high-dimensional numerical vectors (embeddings) and enable fast approximate nearest
neighbour (ANN) search. Unlike traditional databases that store structured rows, vector databases optimise
for similarity search: given a query vector, find the k most similar vectors in the database.

This capability is fundamental to semantic search, recommendation systems, RAG pipelines, and anomaly
detection. Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and Chroma. PostgreSQL
with the pgvector extension provides vector search within a relational database.
"""),
    ("Embedding Models", """
Embedding models convert raw data (text, images, audio) into dense numerical vectors where semantic
similarity corresponds to geometric proximity. For text, models like all-MiniLM-L6-v2 produce 384-dimensional
vectors, while OpenAI's text-embedding-3-large produces 3072-dimensional vectors.

The all-MiniLM-L6-v2 model was fine-tuned using contrastive learning on sentence pairs, making it highly
efficient for semantic similarity tasks. It achieves strong performance on retrieval benchmarks while being
6x smaller than large embedding models, making it ideal for local deployment in RAG systems.
"""),
    ("HNSW Index", """
Hierarchical Navigable Small World (HNSW) is a graph-based ANN algorithm. It builds a multi-layer graph
where higher layers have fewer nodes with longer edges (coarse navigation) and lower layers have more nodes
with shorter edges (fine-grained search). Queries start at the top layer and greedily navigate to the
nearest neighbour, then drop to lower layers for refinement.

Key parameters: ef_construction (graph quality during build, higher = better recall, slower build),
m (number of connections per node, higher = better recall, more memory). Typical production settings are
ef_construction=128, m=16 for a balance of recall and memory usage.
"""),
    ("Cosine Similarity vs Dot Product", """
Cosine similarity measures the angle between two vectors, ignoring magnitude. It equals the dot product of
normalised vectors. This makes it robust to vectors of different magnitudes — a short and long document on
the same topic will have high cosine similarity even though their magnitudes differ greatly.

Dot product similarity rewards both directional alignment and vector magnitude. If embeddings are normalised
(unit vectors), cosine similarity and dot product are equivalent. Most embedding models output normalised
vectors, making the choice equivalent in practice. pgvector supports both operators.
"""),
    ("Chunking Strategies", """
Chunking splits documents into smaller pieces for indexing. Fixed-size chunking splits every N tokens with
overlap, guaranteeing uniform chunk sizes but potentially splitting sentences mid-thought. Semantic chunking
detects natural boundaries like paragraph breaks and sentence endings, keeping meaning intact.

Optimal chunk size depends on the embedding model's context window and the query type. Short chunks (128-256
tokens) improve precision for specific fact retrieval but lose surrounding context. Long chunks (512-1024
tokens) retain more context but may include irrelevant information. Overlapping chunks (50-100 token overlap)
ensure boundary sentences appear in at least one chunk.
"""),
    ("Hybrid Search Approaches", """
Hybrid search combines dense vector search with sparse keyword search (BM25). Vector search excels at
semantic matching — finding documents about the same topic even with different terminology. BM25 excels at
exact keyword matching — finding documents containing specific product codes, names, or technical terms.

Reciprocal Rank Fusion (RRF) merges result lists from multiple retrievers without requiring score
calibration. For each document, RRF score = sum(1 / (k + rank_i)) where rank_i is the document's rank in
retriever i and k=60 is a constant. This rank-based approach is robust to the incompatible score scales
of BM25 and cosine similarity.
"""),
])

# ── PDF 3: LLM Evaluation Metrics ────────────────────────────────────────────
make_pdf("llm_evaluation.pdf", "LLM Evaluation: RAGAS, Benchmarks, and Quality Metrics", [
    ("Why LLM Evaluation Is Hard", """
Evaluating LLM outputs is fundamentally harder than evaluating traditional ML models. Classification models
have clear ground truth labels. LLMs generate open-ended text where multiple valid responses exist for the
same query. A factually correct answer expressed differently from the reference is still correct.

Traditional string-matching metrics like BLEU and ROUGE are inadequate for open-ended generation because
they penalise valid paraphrases. Modern evaluation frameworks use LLMs as judges (LLM-as-judge), leveraging
language understanding to assess semantic correctness, coherence, and faithfulness.
"""),
    ("RAGAS Framework", """
RAGAS (Retrieval Augmented Generation Assessment) is an evaluation framework specifically designed for RAG
systems. It evaluates both the retrieval component and the generation component using three core metrics.

Faithfulness measures whether the generated answer is factually grounded in the retrieved context. It works
by decomposing the answer into individual atomic claims, then checking each claim against the context using
an LLM judge. Faithfulness = number of claims supported by context / total claims. A score below 0.7
indicates hallucination risk.

Answer relevancy measures whether the answer addresses the original question. RAGAS generates hypothetical
questions from the answer, then measures semantic similarity between those questions and the original query.
High answer relevancy means the answer stays on topic.

Context recall measures whether the retrieved chunks contain the information needed to answer the question.
It requires a reference answer for comparison and is computed by checking how many sentences in the
reference answer can be attributed to the retrieved context.
"""),
    ("Hallucination Detection", """
Hallucination in LLMs refers to generating factually incorrect or unsupported statements that appear
confident and plausible. In RAG systems, hallucination occurs when the model generates claims not supported
by the retrieved context — either fabricating facts or contradicting the source documents.

Faithfulness scoring is the primary defence against hallucination in RAG. A strictly grounded system prompt
that instructs the model to answer only from provided context, combined with faithfulness monitoring,
creates a feedback loop that can detect and alert on hallucination trends.

NLI (Natural Language Inference) models can also detect hallucination by classifying whether each generated
statement is entailed, neutral, or contradicted by the context.
"""),
    ("Benchmark Datasets", """
MMLU (Massive Multitask Language Understanding) tests knowledge across 57 academic subjects including
mathematics, law, medicine, and history. It uses multiple-choice questions and measures broad factual
knowledge. GPT-4 achieves 87% on MMLU while smaller models like Llama 3.1 8B achieve around 68%.

HotpotQA is a multi-hop question answering dataset requiring reasoning across multiple documents. Questions
like 'Was the director of Inception born before the director of Interstellar?' require retrieving facts
about both films and comparing them.

BEIR (Benchmarking IR) is a diverse benchmark for information retrieval containing 18 datasets across
domains including biomedical, financial, and conversational search. It tests whether retrieval methods
generalise across domains.
"""),
    ("LLM-as-Judge Pattern", """
LLM-as-judge uses a capable LLM (typically GPT-4 or Claude) to evaluate the outputs of another LLM. The
judge receives the query, context, and response, then scores quality on dimensions like accuracy,
coherence, helpfulness, and safety.

This approach correlates well with human judgement (Spearman correlation ~0.85 between GPT-4 judgements
and human ratings on MT-Bench). However, LLM judges exhibit biases: preferring longer responses,
self-preferring their own style, and being inconsistent on borderline cases.

Mitigation strategies include using reference-based evaluation, running multiple judge calls and averaging,
chain-of-thought reasoning before scoring, and using diverse judge models.
"""),
    ("Continuous Evaluation in Production", """
Production LLM systems require continuous evaluation, not just pre-deployment benchmarking. Model behaviour
drifts as the document corpus changes, user query patterns evolve, and underlying model APIs update. A
system healthy at deployment can degrade silently over weeks.

Sliding window evaluation computes quality metrics over the most recent N queries rather than all-time
averages, making it sensitive to recent degradation. Typical window sizes are 50-200 queries. Alert
thresholds of faithfulness < 0.7 and context recall < 0.6 are standard starting points.

Prometheus and Grafana provide the infrastructure for surfacing these metrics in real time, enabling
on-call teams to detect and respond to quality regressions with the same tooling used for latency and
error rate monitoring.
"""),
])

print("\nAll PDFs created in data/raw/")
print("Copy them to the data/raw/ folder and click Rebuild Index in the dashboard.")