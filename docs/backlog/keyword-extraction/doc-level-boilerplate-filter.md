# Document-level boilerplate filter for abstracts

(Migrated from `artifacts_scripts/curriculum/TODO.md`.)

For the keyword path, this is now subsumed by the branching-entropy phrase miner — boilerplate fragments like `Articles citing`, `Reflect usage`, `Get alerts` get demoted by NPMI×count×BE without an explicit blacklist.

**Still relevant for**:
- Embedding pipelines: papers whose entire `abstract` field is scraped UI/page content (not the actual abstract) get noisy embeddings. A regex set applied at snapshot-ingest time would benefit every downstream consumer, not just keywords.
- LLM proposal generation: when feeding gap context to the LLM, scraped-content abstracts are noise.

**Approach**: regex blacklist of known UI/scraped-content fragments, applied at `openalex_ingest.py`'s ingest step. Either drop the abstract entirely (treat as missing) or strip the matched spans.
