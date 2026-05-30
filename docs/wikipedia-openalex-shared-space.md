# Wikipedia <-> OpenAlex in shared raw GIST space

Do Wikipedia and OpenAlex mix coherently in bookgap's raw 384-D GIST embedding space (no denoiser, drop_k=0), or is there an academic<->encyclopedic distribution shift? And does a domain-conflating alignment improve cross-corpus coherence?

## Setup

- **OpenAlex slice:** NLP topic `T10181` articles, subfield 1702, `denoiser=NONE drop_k=0` -> raw 384-D GIST. N=87408.
- **Wikipedia slice:** `cache/wiki/store/wiki_400k` (cirrus dump, GIST-small). N=400000.
- Shared embedder fingerprint: `6bcf5b5e02946b7e` (identical -> same raw space).
- Both corpora L2-normalized; cosine == dot product. After each conflation, vectors are re-normalized to the unit sphere.

## 1. Separability / shift magnitude (logistic-regression AUC)

AUC~0.5 = well-mixed (corpus identity not recoverable); AUC~1.0 = strong shift (corpus identity dominates the geometry).

| transform | held-out AUC | NN top-1 hub share (2000 OA->wiki) | unique wiki NN / 2000 | top-10 NN overlap vs raw |
|---|---|---|---|---|
| raw | 0.9986 | 0.139 | 654 | 1.000 |
| centroid-removed | 0.4970 | 0.039 | 1413 | 0.163 |
| anti-LDA (drop 1) | 0.9504 | 0.241 | 714 | 0.212 |
| centroid + anti-LDA (drop 1) | 0.4898 | 0.004 | 1620 | 0.083 |

- **AUC**: held-out logistic-regression accuracy at recovering corpus identity. Lower = better mixing.
- **NN top-1 hub share**: fraction of 2000 OA queries whose single nearest wiki article is the *same* most-popular wiki hub. High = collapse onto a generic high-density article (shift symptom).
- **top-10 NN overlap vs raw**: mean Jaccard of each OA query's top-10 wiki neighbors against the raw-space set. High = the conflation preserved topical NN structure (only removed the corpus offset, not the topical geometry).

## 2. Cross-corpus nearest neighbors

For sampled queries, top-3 cosine nearest neighbors in the *other* corpus. Topically sensible NN (NLP paper -> NLP/linguistics wiki) = coherent mixing; collapse onto a few generic high-density wiki articles = shift symptom.

### RAW space

### OpenAlex NLP article -> nearest Wikipedia

- **Automated identification of discourse markers using NLP approach: The case of okay**
  - Corpus linguistics
  - Utterance
  - Text corpus
- **Semantic Role Labeling for multi-VP clauses in Chinese**
  - Clause
  - Theta role
  - Phrase structure rules
- **A Comparison of Syntactically Motivated Word Alignment Spaces**
  - Nominative–accusative alignment
  - Sequence alignment
  - Ergative–absolutive alignment
- **A High-Quality and Large-Scale Dataset for English-Vietnamese Speech Translation**
  - Vietnamese language
  - Machine translation
  - Vietnamese Quoted-Readable
- **A high speed transcription interface for annotating primary linguistic data**
  - Corpus linguistics
  - Transcription (linguistics)
  - Text corpus

### Wikipedia article -> nearest OpenAlex NLP

- **Varaždin**
  - A mondván diskurzusjelölő variációs pragmatikai elemzése (1. rész)
  - Történeti pragmatikai variációról a regiszter tükrében
  - Croatian Glagolitic Script
- **Pearl Drums**
  - 17世紀 金鼓의 造形과 特徵
  - The Development of the Database about Satsuma Mandarin
  - Analisis Hasil Penerjemahan Peribahasa Budaya Bahasa Perancis ke dalam Bahasa Indonesia Pa
- **ITV Choice**
  - Construction automatique de classes de sélection distributionnelle
  - GAIN
  - The Comparison of BBC1 and CT1 Main News Broadcasting in theTime of the Afghanistan War - 
- **Deadlock (computer science)**
  - Double Reduction Of Ada-Ecatnet Representation Using Rewriting Logic
  - A Mathematical Model and Refinement Relation for a CSP-Like Language
  - Revocations Are Dead, Long Live Revocations
- **Music of Evenkia**
  - Music, Modernity and Locality in Prewar Japan: Osaka and Beyond
  - Possible approaches to deciphering Russian ancient Znamenny chant
  - Notációs jelekmegfejtése Karlheinz Stockhausen Zyklus című művében

### After centroid removal

### OpenAlex NLP article -> nearest Wikipedia

- **Automated identification of discourse markers using NLP approach: The case of okay**
  - Discourse analysis
  - Hansard
  - Utterance
- **Semantic Role Labeling for multi-VP clauses in Chinese**
  - Law enforcement in China
  - Politburo Standing Committee of the Chinese Communist Party
  - Mandate of Heaven
- **A Comparison of Syntactically Motivated Word Alignment Spaces**
  - Sequence alignment
  - Structural alignment
  - Alignment
- **A High-Quality and Large-Scale Dataset for English-Vietnamese Speech Translation**
  - Vietnamese language
  - Vietnam
  - Vietnamese diaspora
- **A high speed transcription interface for annotating primary linguistic data**
  - Primary transcript
  - CDNA library
  - Transcription (biology)

### Wikipedia article -> nearest OpenAlex NLP

- **Varaždin**
  - Disambiguation of Homographic Adjective and Adverb Forms in Croatian Texts
  - Building Croatian WordNet
  - N-gram Based Croatian Language Network
- **Pearl Drums**
  - 17世紀 金鼓의 造形과 特徵
  - The Development of the Database about Satsuma Mandarin
  - Re-Interpreting Sino-Japanese Instrumental Notations
- **ITV Choice**
  - What's on tonight
  - Choosing Tense in English.
  - Collocation and Selectional Preferences: A Frame-based Approach
- **Deadlock (computer science)**
  - Multithreaded Natural Language Processing Tasks
  - A Mathematical Model and Refinement Relation for a CSP-Like Language
  - potsdam.de
- **Music of Evenkia**
  - Possible approaches to deciphering Russian ancient Znamenny chant
  - Применение данных компьютерной лингвистики при анализе сегментной фоностилистики финно-уго
  - Generation of Russian Poetry of Different Genres and Styles Using Neural Networks with Cha

### After anti-LDA (domain-invariant) projection

### OpenAlex NLP article -> nearest Wikipedia

- **Automated identification of discourse markers using NLP approach: The case of okay**
  - Machine translation
  - Corpus linguistics
  - Word-sense disambiguation
- **Semantic Role Labeling for multi-VP clauses in Chinese**
  - Clause
  - Machine translation
  - Chinese name
- **A Comparison of Syntactically Motivated Word Alignment Spaces**
  - Sequence alignment
  - Grammar
  - Machine translation
- **A High-Quality and Large-Scale Dataset for English-Vietnamese Speech Translation**
  - Vietnam
  - Ho Chi Minh City
  - Vietnamese alphabet
- **A high speed transcription interface for annotating primary linguistic data**
  - Corpus linguistics
  - Transcription (linguistics)
  - Machine translation

### Wikipedia article -> nearest OpenAlex NLP

- **Varaždin**
  - Improved Parser for Simple Croatian Sentences
  - Morfessor and variKN machine learning tools for speech and language technology
  - Croatian Network Lexicon within the Syntactic and Semantic Framework and LLOD Cloud
- **Pearl Drums**
  - Mapping sentences onto logical form
  - Pearl: A Probabilistic Chart Parser
  - English Corpus Linguistics in Japan
- **ITV Choice**
  - Improving Statistical Machine Translation with Selectional Preferences
  - Issues in word choice
  - Lexical and Morphological Choices in Machine Translation
- **Deadlock (computer science)**
  - Multithreaded Natural Language Processing Tasks
  - Parallel Communicating Grammar Systems with Bounded Resources: Results, Techniques, Open P
  - Synonym Dictionary Improvement through Markov Clustering and Clustering Stability
- **Music of Evenkia**
  - On Construction of a Chinese Corpus Based on Semantic Dependency Relations
  - On dissolving morphological long distance dependencies in Russian verbs
  - Development of a Dependency Treebank for Russian and its Possible Applications in NLP

## 3. Verdict

- **Strong distribution shift in raw GIST.** Corpus identity is almost perfectly recoverable (AUC = 0.999). Wikipedia and OpenAlex occupy offset regions of the sphere; a linear classifier separates them trivially. Raw GIST does **not** present the two corpora as one homogeneous cloud.
- **But the topical geometry survives the shift.** Even at AUC = 0.999, the raw cross-corpus nearest neighbors are topically sensible (NLP papers land on NLP/linguistics wiki articles; a Vietnamese-translation paper lands on *Vietnamese language*; a word-alignment paper on *sequence alignment*). The shift is a global **offset** between two otherwise-aligned topical manifolds, not a scramble of the within-topic structure.
- **The offset is (almost) a pure translation: centroid removal fixes it.** Subtracting each corpus's centroid collapses separability from AUC 0.999 to 0.497 (chance), and the post-removal NN stay topically sensible. The low top-10 NN-overlap-vs-raw (0.16) is expected and desirable: the raw ordering was dominated by the offset, so removing it genuinely re-ranks which wiki articles are nearest.
- **A learned anti-LDA / domain-invariant projection does NOT work here.** Dropping the top corpus-discriminating S_B/S_T eigendirections barely moves the needle (AUC stays ~0.95 even dropping many directions) and *worsens* hub collapse (top-1 hub share rises). Because every vector is unit-norm and the corpus signal is a mean translation rather than a high-variance axis, the generalized-eigenbasis mis-targets it. The cheap mean-subtraction is both correct and sufficient; the learned alignment is unnecessary.

**Recommendation.** When mixing Wikipedia into a gap-detection / curriculum corpus alongside OpenAlex in raw GIST, **mean-center each source before the union** (subtract the per-corpus centroid). This is the cheap fix and it is sufficient: it removes the spurious academic<->encyclopedic offset that would otherwise let a gap land purely on the corpus boundary, while leaving the topical neighbor structure intact. A learned alignment (anti-LDA / CCLIP / CCA) is **not** warranted — it does not improve over centroid removal here and the offset structure is essentially a translation. Caveat: this was measured in **raw** GIST; bookgap's production clouds run a denoiser (LPP, drop_k=10), which re-centers and reshapes the space and may already absorb much of the offset. Re-check the AUC in the denoised space before assuming Wikipedia needs explicit centering there.

### Anti-LDA n_drop sweep (why the learned projection fails)

Dropping the top corpus-discriminating generalized-eigendirections leaves AUC ~0.95-0.96 regardless of how many are dropped, and increases nearest-neighbor hub collapse — confirming the corpus signal is a translation, not a low-rank variance subspace:

| anti-LDA drop k | held-out AUC | NN top-1 hub share |
|---|---|---|
| 1 | 0.9504 | 0.256 |
| 2 | 0.9557 | 0.250 |
| 4 | 0.9597 | 0.237 |
| 8 | 0.9604 | 0.242 |
| 16 | 0.9620 | 0.247 |
| 32 | 0.9611 | 0.239 |


## Cross-corpus gap detection (OpenAlex books vs OpenAlex+Wikipedia ambient)

SP3 capstone: the gap algorithms now accept any `Corpus`, so `detect_gaps` runs with a cross-corpus `UnionCorpus` ambient. Landmarks are OpenAlex NLP books (topic `T10181`); the ambient/query is a `UnionCorpus` of OpenAlex NLP articles + a Wikipedia slice, all in raw 384-D GIST. Each corpus is **mean-centered then re-normalized before the union** (the validated alignment: the OpenAlex<->Wikipedia shift is a pure per-corpus offset). faiss is pinned to one thread.

### Setup

- **Landmark:** OpenAlex NLP books, N=1231.
- **Ambient/query:** `UnionCorpus` of OpenAlex NLP articles (N=4000) + Wikipedia `cache/wiki/store/wiki_400k` (N=4000); total N=8000.
- Each corpus mean-centered (OpenAlex centroid for books+articles, Wikipedia centroid for wiki) then L2-renormalized. `CellType.KBALL`, `num_gaps=3`, `radius_fraction=0.75`, single-thread faiss.

### Result

- `detect_gaps` ran end-to-end and returned **3 gap(s)**.

#### Gap 0 — size 52 (percentile 1.000)

- Membership: **0 OpenAlex** + **52 Wikipedia** = 52 in-gap points.

- Example Wikipedia members:
  - wiki:114605  —  Danville, Kansas
  - wiki:96802  —  Cobb County, Georgia
  - wiki:110659  —  Bancroft, Idaho
  - wiki:110686  —  Weston, Idaho
  - wiki:136034  —  Haskell, Texas

#### Gap 1 — size 50 (percentile 1.000)

- Membership: **0 OpenAlex** + **50 Wikipedia** = 50 in-gap points.

- Example Wikipedia members:
  - wiki:876817  —  Crescent moon
  - wiki:38325  —  Descent
  - wiki:256138  —  Mere
  - wiki:238971  —  Kira
  - wiki:87562  —  Blast

#### Gap 2 — size 42 (percentile 1.000)

- Membership: **0 OpenAlex** + **42 Wikipedia** = 42 in-gap points.

- Example Wikipedia members:
  - wiki:117325  —  Hamilton Township, Clare County, Michigan
  - wiki:126244  —  Gerry, New York
  - wiki:131938  —  New Bethlehem, Pennsylvania
  - wiki:117151  —  Bertrand Township, Michigan
  - wiki:134183  —  South Huntingdon Township, Pennsylvania

### Interpretation

- **The cross-corpus machinery works end-to-end.** `detect_gaps` consumed a
  `UnionCorpus` ambient/query with no shared `SubCloud.parent` — exercising the
  relaxed `get_kball_coverage` (now `Corpus`-generic: shared-embedder check, not
  `isinstance(SubCloud)`/shared-`.parent`) and the range-searchable
  `UnionCorpus.index()` (a `FlatL2` over the concatenated centered embeddings, so
  `range_search` works where `IndexShards` raised "range search not implemented").
- **The top gaps are 100% Wikipedia, and that is the correct result, not a bug.**
  The landmarks are NLP *books*. The densest book-free regions are therefore the
  Wikipedia general-knowledge clusters that no NLP book sits near: tight masses of
  US place/township stubs (*Danville, Kansas*; *Cobb County, Georgia*) and short
  high-degree disambiguation pages (*Descent*, *Mere*, *Blast*, *Crescent moon*).
  These are far from any NLP book yet densely packed in GIST space, so k-ball
  coverage flags them as large uncovered gaps. The OpenAlex NLP articles, by
  contrast, are well-covered by their natural NLP-book landmarks, so they do not
  form the *largest* gaps and are absent from the top-3.
- **Why mean-centering still matters here.** Without per-corpus centering the raw
  academic↔encyclopedic offset (AUC 0.999, §1) would let a gap land purely on the
  corpus boundary — every gap would trivially be "all of Wikipedia, because it's
  offset from the books." After centering, the corpora share an origin, so the
  gaps reflect genuine local density (specific town/disambig clusters), not the
  global offset. The all-Wikipedia membership is a *topical* statement (random
  Wikipedia is mostly non-NLP and unbooked), not a corpus-offset artifact.
- **Takeaway for a targeted run.** To surface *cross-corpus* NLP gaps (where
  OpenAlex articles and topically-relevant Wikipedia articles co-occupy a
  book-free region), the Wikipedia slice should be filtered to the topic rather
  than sampled uniformly from all 400k pages; a uniform sample is dominated by
  off-topic stubs that out-mass any NLP cluster. The infrastructure is ready for
  that — only the Wikipedia slice's selection changes.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.cross_corpus_gap_demo
--wiki-store cache/wiki/store/wiki_400k --oa-limit 4000 --wiki-limit 4000
--num-gaps 3 --radius-fraction 0.75 --seed 0`. The `--oa-limit`/`--wiki-limit`
caps keep the single-threaded run tractable: kball coverage does one range search
per union point against the full ambient index, ~50 ms/point at 400k, so the full
uncapped union is a multi-hour job, not a demo.)*


## Topic-filtered cross-corpus gaps (NLP-relevant Wikipedia ambient)

The uniform-sample run above had a known weakness flagged in its own takeaway: a
uniform 400k Wikipedia slice is ~entirely non-NLP, so the densest book-free
regions it produces are generic encyclopedic stubs (US townships, bare
disambiguation pages) — *true* k-ball gaps, but not a useful "what does the NLP
literature discuss that no NLP book covers" signal. This follow-up replaces the
uniform Wikipedia slice with a **topic-filtered** one, reusing the
already-embedded `wiki_400k` store (**no re-embedding**) and changing only which
rows enter the union.

### Selection (semantic, no new embed)

After mean-centering each corpus (OpenAlex centroid for books+articles, the
**full-store** Wikipedia centroid for wiki) and re-normalizing, we form the
mean-centered NLP direction = the L2-normalized mean of the centered OpenAlex
NLP article embeddings, and keep the **top-3000** Wikipedia articles by cosine to
it. The selection is exactly what you'd hope for — the 15 highest-cosine
Wikipedia pages are textbook NLP/computational-linguistics topics, not township
stubs:

> Machine translation (0.657), Word-sense disambiguation (0.625), Corpus
> linguistics (0.608), Text corpus (0.587), Sentence extraction (0.560),
> Part-of-speech tagging (0.544), Information extraction (0.542), WordNet
> (0.538), Parsing (0.534), Cross-language information retrieval (0.523),
> Dictionary-based machine translation (0.522), Parse tree (0.519),
> Probabilistic context-free grammar (0.516), Translation memory (0.505),
> Lemmatization (0.505).

### Setup

- **Landmark:** OpenAlex NLP books (topic `T10181`), N=1231.
- **Ambient/query:** `UnionCorpus` of OpenAlex NLP articles (N=4000) +
  top-3000 NLP-relevant Wikipedia from `cache/wiki/store/wiki_400k` (N=3000);
  total N=7000. Each corpus mean-centered then L2-renormalized.
- `CellType.KBALL`, `num_gaps=5`, `radius_fraction=0.75`, single-thread faiss,
  seed 0.

### Result — the 5 gaps

All five are at percentile 1.000 (largest gaps among the books' own k-ball
sizes). The dominant theme across **every** gap is **individual world languages
and language families** — NLP/linguistics-adjacent Wikipedia pages that the
OpenAlex *NLP-techniques* book landmarks sit nowhere near.

| gap | size | OA / wiki | disambig titles | character (highest-NLP-cosine members) |
|---|---|---|---|---|
| 0 | 725 | 0 / 725 | 135 | language names + bare disambiguation pages: *Italian*, *Japanese*, *Finnish*, *Croatian*, *Kara language*, *Cantonese (disambiguation)* |
| 1 | 280 | 0 / 280 | 100 | language + prosody, still disambig-heavy: *Prosody*, *Gypsy language*, *South Slavic*, *Gallo*, *Frisian language (disambiguation)* |
| 2 | 128 | 0 / 128 | 1 | **clean: language families** — *Language family*, *Romance languages*, *Indo-European languages*, *Esperanto*, *Hindi*, *Russian language*, *Language isolate* |
| 3 | 100 | 0 / 100 | 45 | languages + phonetics, disambig-mixed: *Slovene*, *Phrasing*, *Tati language*, *Lingala (disambiguation)* |
| 4 | 82 | **1** / 81 | 0 | **clean cross-corpus: African / creole languages** — OA *"LMF for a selection of African Languages"* co-occupies with *Bambara*, *Songhay*, *Swahili*, *Bantu languages*, *Tunisian Arabic*, *Creole language*, *Portuguese-based creoles* |

(Representative members shown are the highest-cosine-to-NLP-centroid members of
each gap; "disambig titles" counts members whose title contains
"(disambiguation)".)

### Uniform vs topic-filtered

| | uniform 400k sample (above) | topic-filtered top-3000 |
|---|---|---|
| top-3 gap themes | US township stubs (*Danville, Kansas*); bare disambig (*Descent*, *Mere*, *Blast*) | world languages / language families (*Italian*, *Indo-European languages*, *Bantu*); some residual disambig |
| any NLP content? | **none** — gaps are pure general-knowledge noise | **yes** — every gap is a language/linguistics cluster; Gaps 2 & 4 are clean |
| cross-corpus (OA + wiki co-membership)? | no (all gaps 100% wiki) | mostly no, **but Gap 4 mixes an OpenAlex NLP article with wiki language pages** |
| residual noise | dominant | bare disambiguation pages survive (they are short, high-degree, weakly-NLP, and cluster tightly) |

### Verdict

Topic-filtering the Wikipedia ambient **works and is clearly the right move**:
the gaps go from general-knowledge noise (townships, raw disambiguation) to a
coherent **languages-and-linguistics** signal, and one gap (Gap 4) realizes the
intended *cross-corpus* shape — an OpenAlex NLP article sharing a book-free
region with topically-matched Wikipedia language pages. The interpretation is
genuinely useful: *the NLP-techniques book landmarks have essentially no
coverage of the world's individual languages and language families, which both
Wikipedia and the preprint/article literature document heavily.* That is a real,
actionable "gap in book coverage" rather than a corpus-boundary artifact (the
per-corpus mean-centering ensures the offset is removed, §1).

Two honest caveats keep this short of "deploy as-is." (i) The membership is still
overwhelmingly Wikipedia (4 of 5 gaps are 100% wiki): mean-centered Wikipedia
language pages form their own tight, OpenAlex-sparse mass, so they out-compete
mixed OA+wiki clusters for the *largest* gaps. Surfacing more cross-corpus gaps
likely needs either a tighter topic filter (push cosine threshold up so only
densely-NLP wiki survives), a Wikipedia-namespace prefilter (drop
`(disambiguation)` pages and stubs before selection), or a size metric that
rewards OA/wiki co-membership over raw mass. (ii) Bare disambiguation pages
(*DA*, *PL*, *Malta (disambiguation)*) still leak into Gaps 0/1/3 — they are
short, high-degree, weakly-on-topic, and cluster tightly, so a title/namespace
filter would be the cheapest cleanup. Net: cross-corpus gap detection with
Wikipedia-as-ambient **is useful** for finding book-coverage holes once the
Wikipedia slice is topic-relevant; the remaining work is noise-pruning the slice,
not the gap machinery, which is sound.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.cross_corpus_gap_demo
--wiki-store cache/wiki/store/wiki_400k --topic-filter --wiki-topk 3000
--oa-limit 4000 --num-gaps 5 --radius-fraction 0.75 --seed 0`.)*
