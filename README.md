# Multimodal Visual Search Engine Using SigLIP

# Overview
Developed a multimodal retrieval system that aligns textual and visual representations for crossmodal search. The system enables natural languagebased image retrieval from largescale datasets by leveraging SigLIP’s multimodal embeddings and FAISSbased vector indexing for efficient similarity search.

# Framework
Models: SigLIP, FAISS
Libraries: PyTorch, OpenCV, Transformers, NumPy, Matplotlib

# Scope
 Build a texttoimage and imagetotext retrieval system using SigLIP.
 Utilize joint vision–language embeddings for multimodal alignment.
 Implement FAISS for scalable similarity search over large vector spaces.
 Evaluate on Flickr30k and MSCOCO benchmarks.
 Optimize retrieval latency and embedding space coherence.

# Datasets Used:
 Flickr30k: 31,000 images with 5 textual captions each.
 MSCOCO: 120,000 images with multiple humanannotated descriptions.

# Preprocessing Steps:
 Image resizing and normalization (224×224).
 Caption cleaning, tokenization, and lowercasing.
 Embedding extraction for all text–image pairs via SigLIP.
 Normalization and FAISS index construction for retrieval.

 # Methodology

 1. Embedding Extraction

 Visual Encoder: SigLIP Vision Transformer extracts visual embeddings.
 Text Encoder: SigLIP Text Transformer generates corresponding textual embeddings.
 Both embeddings projected into a shared multimodal latent space.

 2. Embedding Alignment

 Finetuned on Flickr30k captions using contrastive loss to maximize alignment of matching pairs while separating nonmatching pairs.

 3. Indexing and Retrieval

 Constructed FAISS IndexFlatIP (Inner Product) for highspeed similarity search.
 Stored normalized embeddings (unit vectors) for cosine similarity equivalence.
 Implemented bidirectional retrieval (text→image and image→text).

 4. Evaluation and Visualization

 Metrics: Recall@1, Recall@5, Recall@10, mAP.
 Visualized query–retrieval pairs for interpretability and alignment validation.

# Architecture (Textual Diagram)

            ┌────────────────────────────┐
            │ Input Query (Text or Image)│
            └──────────────┬─────────────┘
                           │
          ┌────────────────▼────────────────┐
          │ SigLIP Encoders                 │
          │   Text Transformer             │
          │   Vision Transformer           │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │ Shared Multimodal Embedding Space│
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │ FAISS Vector Index               │
          │   Similarity Search             │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │ Retrieved Results (Topk Images) │
          └─────────────────────────────────┘

#  Results
| Dataset   | Metric   | Performance |
| Flickr30k | Recall@1 | 82%         |
| Flickr30k | Recall@5 | 92%         |
| MSCOCO    | Recall@1 | 79%         |
| MSCOCO    | Recall@5 | 89%         |

# Qualitative Findings:
 High precision in retrieval for natural queries (e.g., “a man riding a surfboard”).
 Demonstrated robustness to linguistic paraphrasing.
 Fast retrieval (<150ms per query) using FAISS flat index.

# Conclusion
The Multimodal Visual Search Engine effectively bridges text–image understanding via SigLIP’s unified embedding space and FAISSbased nearestneighbor search, achieving 82% Recall@1 on Flickr30k. The system enables natural languagedriven image exploration, supporting applications in visual data management, media analytics, and semantic search engines.

# Future Work
 Integrate crossmodal reranking for semantic refinement.
 Deploy GPUoptimized FAISS clustering for billionscale retrieval.
 Extend to video–text retrieval using framelevel embeddings.
 Experiment with multilingual SigLIP for crosslingual visual search.

# References
1. Zhai, X. et al. (2023). SigLIP: Scaling VisionLanguage Models with Sigmoid Loss. Google Research.
2. Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). OpenAI.
3. Johnson, J. et al. (2021). BillionScale Similarity Search with GPUs (FAISS). Facebook AI.
4. Young, P. et al. (2014). From Image Descriptions to Visual Denotations: Flickr30k Entities. CVPR.
5. Lin, T.Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV.

# Closest Research Paper:
> “SigLIP: Scaling VisionLanguage Models with Sigmoid Loss” — Google Research, 2023.
> This work forms the theoretical foundation for the project’s multimodal embedding alignment and retrieval mechanism.
