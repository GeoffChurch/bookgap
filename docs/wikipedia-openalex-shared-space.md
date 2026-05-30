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


## CSLS + topic-relevant reverse NN

The cross-corpus NN block in §2 had two weaknesses, fixed here. **(A)** Its *reverse* direction sampled **uniform** Wikipedia rows (*Varazdin*, *Pearl Drums*, ...) — off-topic encyclopedic pages with no genuine NLP neighbor, so "do they map to sensible OpenAlex NLP articles?" was unanswerable. Here both directions use **NLP-relevant** samples: the Wikipedia query points are drawn from the **top-3000** Wikipedia by cosine to the mean-centered OpenAlex-NLP-article centroid (the same selection `cross_corpus_gap_demo.py --topic-filter` uses). **(B)** It scored NN with raw cosine, which is hubness-prone (a few generic target points become everyone's nearest neighbor). We add **CSLS** (Cross-domain Similarity Local Scaling) as an alternative and measure whether it reduces hubness.

CSLS(x, y) = 2·cos(x, y) − r_T(x) − r_S(y), where r_T(x) is the mean cosine of the source query x to its k=10 nearest *target*-corpus points and r_S(y) is the mean cosine of the target candidate y to its k=10 nearest *source*-corpus points. A target **hub** is densely surrounded by source points, so it has a high r_S(y) and CSLS subtracts it off — demoting the hub from the top-1 slot.

### Setup

- Same raw 384-D GIST space as §1–2: OpenAlex NLP articles (N=87408, denoiser=NONE drop_k=0) and the full Wikipedia store `cache/wiki/store/wiki_400k` (N=400000, fp `6bcf5b5e02946b7e`).
- Each corpus mean-centered (OpenAlex centroid over the NLP articles, Wikipedia centroid over the **full** store) then re-normalized — identical alignment to the topic-filtered gap demo.
- NLP-relevant Wikipedia subset = top-3000 by cosine to the mean-centered NLP-article centroid. Hub-share over 200 (OA→wiki) / 200 (wiki→OA) sampled queries, seed 0.

### Hub top-1 share — raw cosine vs CSLS (both directions)

Top-1 hub share = fraction of sampled queries whose single nearest target is the *same* most-popular target across the sample. Lower = less hub collapse. (unique NN / queries in parentheses.)

| direction | scorer | NN top-1 hub share | unique NN / queries |
|---|---|---|---|
| OpenAlex → Wikipedia | raw cosine | 0.040 | 156 / 200 |
| OpenAlex → Wikipedia | CSLS | 0.015 | 177 / 200 |
| Wikipedia → OpenAlex | raw cosine | 0.030 | 182 / 200 |
| Wikipedia → OpenAlex | CSLS | 0.010 | 196 / 200 |

### Example NN lists — raw vs CSLS

#### OpenAlex NLP article → nearest NLP-relevant Wikipedia

- **Towards a computational processing of oral dialectal data**
  - raw : Dialect, Greek language, Accent (sociolinguistics)
  - csls: Dialect, Yola dialect, Greek language
- **Où en sommes-nous dans la reconnaissance des entités nommées structurées à partir de la pa**
  - raw : Voice recognition, Transcription (linguistics), Maria Schneider
  - csls: Nas (disambiguation), Sylvester, NNN
- **Fast Computing Grammar-driven Convolution Tree Kernel for Semantic Role Labeling.**
  - raw : Supervised learning, Role and reference grammar, Kernel
  - csls: Supervised learning, Kernel, Role and reference grammar
- **Discontinuity and Non-Projectivity: Using Mildly Context-Sensitive Formalisms for Data-Dri**
  - raw : L-attributed grammar, Dangling else, Recursive descent parser
  - csls: Coco/R, Dangling else, L-attributed grammar
- **Fast computation of maximum entropy / minimum divergence feature gain**
  - raw : Machine learning, Boosting (machine learning), Prediction by partial matching
  - csls: Prediction by partial matching, Machine learning, Boosting (machine learning)

#### NLP-relevant Wikipedia → nearest OpenAlex NLP article

- **IPS**
  - raw : Enriching a Lexicographic Tool with Domain Definitions:, Syntactic Approach Of Pronominal Anaphora Resolution In, Méthode d'accès informatisé aux combinaisons lexicales 
  - csls: Syntactic Approach Of Pronominal Anaphora Resolution In, Enriching a Lexicographic Tool with Domain Definitions:, Semantic classification of noun phrases using web count
- **Azari**
  - raw : Lexical Disambiguation of Polysemeous Adjectives in MT , LEXICAL DISAMBIGUATION OF POLYSEMEOUS ADJECTIVES IN MAC, Arabic word analogies and semantics of simple phrases
  - csls: Arabic word analogies and semantics of simple phrases, LEXICAL DISAMBIGUATION OF POLYSEMEOUS ADJECTIVES IN MAC, Lexical Disambiguation of Polysemeous Adjectives in MT 
- **Kosice (disambiguation)**
  - raw : L'évaluation des résultats de la levée d'ambiguïtés lex, Linking and Harmonizing Different Lexical Resources: a , A Large-Scale Multilingual Disambiguation of Glosses
  - csls: A Large-Scale Multilingual Disambiguation of Glosses, L'évaluation des résultats de la levée d'ambiguïtés lex, Extraction of Verb Synonyms Using Graph-based Clusterin
- **KL-ONE**
  - raw : Semantic interpretation using KL-ONE, Knowledge representation for syntactic/semantic process, Mapping knowledge to language with LOOM
  - csls: Semantic interpretation using KL-ONE, Mapping knowledge to language with LOOM, Zur Wissensrepraesentation in der Maschinellen Ueberset
- **Frankish language (disambiguation)**
  - raw : Interactive disambguation of natural language input: a , Fragmentation and part of speech disambiguation, Traduction automatique pour une paire de langues peu do
  - csls: Interactive disambguation of natural language input: a , Traitement des clitiques dans un environnement multilin, La linguistique textuelle en fenno-scandie


### Verdict

**(A) The topic-relevant reverse NN is now meaningful, and the answer is yes.**
With the reverse queries drawn from the NLP-relevant Wikipedia subset (instead of
the off-topic *Varazdin* / *Pearl Drums* uniform sample of §2), the wiki→OpenAlex
NN are topically sensible: *KL-ONE* lands on *"Semantic interpretation using
KL-ONE"* (an exact topical hit), and the lexical/disambiguation/anaphora wiki
pages land on lexical-disambiguation, anaphora-resolution, and semantic-classification
OpenAlex papers. The residual junk is exactly the noise already flagged in the
topic-filter verdict: short, weakly-NLP **disambiguation stubs** (*IPS*, *Azari*,
*Kosice (disambiguation)*, *Frankish language (disambiguation)*) that survive the
top-3000 cosine cut. Their NN are arbitrary because the *query* is degenerate, not
because the mapping is broken — a Wikipedia-namespace prefilter (drop
`(disambiguation)` pages before selection) would clean these up.

**(B) CSLS reduces hubness in both directions.** Top-1 hub share drops from 0.040
to 0.015 (OA→wiki) and from 0.030 to 0.010 (wiki→OA), and unique-NN diversity
rises (156→177 and 182→196 of 200). So CSLS does what it promises: it demotes the
few generic target points that the raw cosine kept handing back. Two honest
caveats: (i) the absolute hub share is **already low** here (~0.03–0.04) because
this is the *topic-relevant, mean-centered* setting — the §1 raw-space hub share
of 0.139 was the genuinely hub-prone regime, and centering + topic-filtering
already removed most of the collapse before CSLS is even applied; (ii) CSLS is not
free — on the NER paper *"Où en sommes-nous dans la reconnaissance des entités
nommées…"* it replaced sensible raw neighbors (*Voice recognition*,
*Transcription*) with anti-hub junk (*Nas (disambiguation)*, *NNN*). The r_S(y)
penalty over-rewards *under*-connected target points, so when the genuinely-correct
neighbor happens to be moderately popular, CSLS can over-penalize it and surface an
isolated stub instead. Net for the NN check: **CSLS is a worthwhile
hubness diagnostic and a mild improvement, but not a dramatic one in this
already-centered regime, and it has a failure mode of its own (anti-hub promotion).**

### Should CSLS be applied inside the cross-corpus gap k-ball?

The gap algorithm's ambient↔landmark relationship is a **k-ball radius**, not a
top-1 NN: a query point's ball radius is `radius_fraction ×` (distance to its
k-th nearest *landmark* book), and the gap size is how many ambient points fall in
that ball (`kball_algorithm.py` / `get_kball_coverage`). The hubness concern maps
onto this as follows. An **anti-hub** ambient point — one that is far from *every*
book because it sits in a region the landmark corpus does not reach (e.g. a
mean-centered Wikipedia language-family cluster with no nearby NLP *book*) — gets a
**large** nearest-book distance, hence a **large** ball radius, hence sweeps in
many neighbors and is flagged as a large gap. That is precisely the §"Topic-filtered
cross-corpus gaps" result: the top gaps are ~100% Wikipedia language pages, which
are real book-coverage holes but also the most anti-hub-like points relative to the
*book* landmarks.

**Recommendation: do NOT bake CSLS into the cross-corpus gap k-ball, at least not
on this evidence — but keep it on the backlog as an opt-in radius correction.**
Reasoning:

1. **The hubness we measured is in the *cross-corpus NN* score, which the k-ball
   does not use.** The k-ball uses the query→landmark (article/wiki → book)
   distance to set a radius and then a query→ambient range search to count
   members — neither is the OA↔wiki top-1 NN that CSLS rescales. The 0.04→0.015
   hub-share improvement therefore does not directly transfer to "fewer spurious
   gaps."

2. **The all-Wikipedia gaps are a *topical* statement, not a hubness artifact.**
   The shared-space verdict (§1) already established that after mean-centering the
   corpora share an origin and the topical geometry is intact; the language-family
   gaps are genuine "no NLP *book* covers the world's languages" holes, not points
   that are spuriously close to a hub. So the symptom CSLS fixes (collapse onto a
   generic hub) is **not** what is producing those gaps.

3. **Where CSLS *could* help is the radius term, and that is a different, untested
   correction.** If one wanted to suppress anti-hub points from dominating gaps,
   the lever is the *nearest-landmark distance* that sets the radius: an
   anti-hub-corrected radius (shrink the ball for points whose large
   nearest-book distance is an anti-hubness artifact rather than a real
   coverage hole) — i.e. apply a CSLS-style local-scaling term to the
   query→landmark distance, not the cross-corpus NN. This is plausible but
   **unmeasured**, would change gap *sizes* (not just orderings), and risks
   suppressing exactly the real cross-corpus holes we are trying to surface. It
   should be validated on its own before touching the gap algorithm.

This is logged as the backlog item
`docs/backlog/keyword-extraction/soft-hubness-correction-csls.md` (CSLS soft
hubness correction, ported from `mycooc/soft_coocmap.py:_gep_csls`); the
gap-radius variant above is a distinct application of the same idea and is noted
there.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.csls_nn_check
--wiki-store cache/wiki/store/wiki_400k --wiki-topk 3000 --csls-k 10
--n-nn-examples 200 --seed 0`.)*


## Is the shift a pure offset? Residual-structure battery

The §1 result (linear AUC 0.999 -> ~0.497 after per-corpus centering + renormalization) is necessary but not sufficient for "pure offset": a single-direction linear AUC -> chance only shows the *means* coincide after centering and that no single hyperplane separates the corpora. It says nothing about whether the two clouds have the same covariance, shape, or density. This section adds a battery that tests exactly that residual structure, on the **same** centered+renormalized clouds (`_normalize(X - X.mean(0))` per corpus).

- **Sample.** Balanced 50000 OpenAlex NLP articles (topic T10181, raw 384-D GIST) vs 50000 Wikipedia rows from `cache/wiki/store/enwiki_full`, seed 0. Both centered + renormalized.

### 0. Sanity — single-fit logistic AUC reproduces §1

| | held-out logistic AUC |
|---|---|
| raw (uncentered) | 0.9992 |
| centered + renormalized | 0.5446 |

Reproduces the headline collapse (0.999 -> ~0.497 chance) on this larger balanced sample: raw 0.9992, centered 0.5446.

### 1. Multi-direction linear separability (LDA deflation)

Fit an LDA direction on the centered clouds, record its held-out 1-D AUC, then project both corpora orthogonal to it and refit. If linear separability is genuinely rank-1 (mean-only), AUC collapses to ~0.5 at direction 1 and stays; a later direction with AUC>0.5 means the mean-only claim is under-powered (a 2nd/3rd linear axis still separates).

| direction | held-out 1-D LDA AUC |
|---|---|
| 1 | 0.5379 |
| 2 | 0.5391 |
| 3 | 0.5344 |
| 4 | 0.5373 |
| 5 | 0.5379 |

### 2. Covariance / shape comparison

Top-k principal components of each centered corpus: principal angles between the two PC subspaces (0 deg = identical subspace) and relative L1 difference of the top-k eigenvalue (variance) spectra. A pure offset => near-identical covariance: small angles, matching spectra.

| k | mean angle (deg) | median | max | spectrum rel-L1 |
|---|---|---|---|---|
| 10 | 71.9 | 73.8 | 88.9 | 0.470 |
| 20 | 65.3 | 69.2 | 88.7 | 0.380 |
| 50 | 54.9 | 55.4 | 89.8 | 0.274 |

First 10 principal angles at k=50 (deg): 19.0, 20.9, 22.4, 24.2, 25.5, 26.9, 27.7, 29.9, 30.9, 32.5

Top-8 PC variances — OA: 0.0576, 0.0475, 0.0405, 0.0314, 0.0287, 0.0266, 0.0247, 0.0196; wiki: 0.0317, 0.0265, 0.0241, 0.0215, 0.0191, 0.0171, 0.0145, 0.0135

### 3. Nonlinear two-sample test (MMD + energy distance)

On 5000 centered+renorm points per group (RBF bandwidth sigma=1.4177, median heuristic; 200-permutation null). The **within-corpus baseline** (split each corpus in half, measure half-vs-half) is the noise floor for "same distribution." cross ~ within => pure offset; cross >> within => residual higher-order/shape difference.

| statistic | cross-corpus | within-corpus baseline | permutation null (mean +/- std) | p-value |
|---|---|---|---|---|
| MMD^2 (RBF) | 0.001040 | -0.000000 (OA 0.000006, wiki -0.000006) | 0.000002 +/- 0.000015 | 0.0050 |
| energy distance | 0.003144 | 0.001129 (OA 0.001143, wiki 0.001116) | 0.000569 +/- 0.000034 | 0.0050 |

The within-corpus MMD^2 baseline is ~0 (the unbiased estimator's value for two halves of the *same* distribution), so a cross/within ratio is undefined; the cross MMD^2 instead sits **71 permutation-null standard deviations** above its null. The energy distance gives a well-defined **2.8x** the within-corpus baseline. A pure offset would put cross at the within-corpus noise floor (ratio ~1, z ~0); both statistics reject that decisively (p=0.0050), the signature of a residual shape/density difference the linear test cannot see.

### 4. Local kNN same-corpus purity

Pool both centered+renorm samples; for each query point, the fraction of its k=10 nearest neighbors that share its corpus. Baseline = mixing-ratio expectation under random labels (p0^2 + p1^2). mean ~ baseline => locally well-mixed (translates interleave); mean > baseline => local clumping / density-shape difference invisible to the global linear test.

| k | mean purity | baseline | excess | OA-query purity | wiki-query purity |
|---|---|---|---|---|---|
| 10 | 0.9879 | 0.5000 | +0.4879 | 0.9971 | 0.9787 |

### Verdict

**Verdict: MEAN-SHIFT-DOMINANT WITH RESIDUAL.** After per-corpus centering+renormalization the *mean* separation is gone — the single-direction logistic AUC collapses to chance (0.545, §0), and no successive LDA direction recovers it (all five deflation AUCs ~0.54, §1), so there is **no residual *linear* structure** beyond the matched means. But the two clouds are still NOT translates of each other: the residual is **nonlinear — a difference in covariance, shape, and local support** that a hyperplane cannot see. (2) Their top-50 PC subspaces sit at mean principal angle 54.9 deg with mismatched variance spectra (rel-L1 0.27; OA's leading PCs carry markedly more variance than wiki's) — materially different covariance/orientation. (3) The cross-corpus MMD^2 sits 71 sigma above its permutation null and the energy distance is 2.8x the within-corpus (half-vs-half) baseline (both p=0.0050) — far above the same-distribution noise floor, confirming a residual higher-order/shape difference. (4) Local kNN same-corpus purity is 0.988 vs the 0.500 mixing-ratio baseline (excess +0.488) — the points overwhelmingly neighbor their own corpus, so the two centered clouds occupy largely *disjoint local regions* of the sphere despite sharing a mean and admitting no separating hyperplane. Practical bound: a wiki-learned linear projection (e.g. LPP) transfers to OpenAlex only to the extent these covariances align — at 55 deg mean principal angle (k=50) the wiki principal axes are a poor basis for the OA cloud, so expect material degradation, not a clean transfer.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.shift_characterization --wiki-store cache/wiki/store/enwiki_full --n-each 50000 --seed 0`.)*


## Topic-matched re-test

The battery above used a **random** Wikipedia sample drawn from all 7.1M pages. That sample is dominated by non-NLP topics (footballers, villages, chemical compounds), so part of the residual it found — especially the 0.988 kNN same-corpus purity — could be *topic-distribution mismatch* (NLP vs all-of-Wikipedia) rather than a genuine corpus/register shift. The decision-relevant question is the **within-topic** one: take only NLP-relevant Wikipedia and re-run.

- **Wiki side (topic-matched).** Top-40000 Wikipedia pages by cosine to the OpenAlex-NLP centroid — the exact `select_nlp_pages` selection the wiki curriculum uses (cosine 0.145..0.362 to the centered NLP direction).
- **OA side.** SAME OA-NLP slice (topic T10181, raw 384-D GIST) as the prior run, subsampled to 40000 to balance against wiki. Seed 0.

**Centering — two variants, and the distinction matters.** The prior random-sample run subtracted each *sample's own* mean. For a random 50k sample that mean ≈ the full-store mean, so it was implicitly removing the global wiki offset. The NLP-relevant slice is **not** representative of the store, so its own mean sits ~0.25 off the full-store centroid; the two centerings now diverge:

- **Own-mean (primary).** Subtract each sample's own mean, renormalize — the apples-to-apples analog of the prior run. This is the battery used for the side-by-side below and the verdicts: it removes the *same kind* of offset the prior run did, so any surviving residual is a genuine topic-vs-register comparison.
- **Full-store (practical caveat).** Subtract the global full-store wiki centroid (the offset `cross_corpus_gap_demo` removes in the pipeline). Because the slice is non-representative, this leaves a residual wiki mean of norm 0.388 on the sphere, and the clouds stay trivially separable (logistic AUC 1.0000). Reported as the §0 secondary row; NOT run through the full battery, since this leftover *offset* — not a shape residual — would dominate every downstream metric.

Side-by-side with the prior random-sample run (battery identical; only the wiki sample + centering scope changed):

### 0. Sanity — raw vs centered logistic AUC

| centering | random-sample (prior) | topic-matched (new) |
|---|---|---|
| raw (uncentered) | 0.9992 | 0.9955 |
| own-mean centered (primary) | 0.5446 | 0.5344 |
| full-store centered (caveat) | — | 1.0000 |

### 1. Multi-direction linear separability (LDA deflation)

| direction | random-sample 1-D LDA AUC | topic-matched 1-D LDA AUC |
|---|---|---|
| 1 | 0.5379 | 0.5282 |
| 2 | 0.5391 | 0.5285 |
| 3 | 0.5344 | 0.5299 |
| 4 | 0.5373 | 0.5273 |
| 5 | 0.5379 | 0.5310 |

### 2. Covariance / shape (principal angles + spectra)

| k | random-sample mean angle | topic-matched mean angle | random spec rel-L1 | topic-matched spec rel-L1 |
|---|---|---|---|---|
| 10 | 71.9 | 53.8 | 0.470 | 0.333 |
| 20 | — | 49.0 | — | 0.295 |
| 50 | 54.9 | 40.7 | 0.274 | 0.218 |

First 10 topic-matched principal angles at k=50 (deg): 12.6, 14.1, 14.8, 15.4, 16.2, 17.2, 18.0, 18.1, 19.0, 19.8

Top-8 PC variances (topic-matched) — OA: 0.0577, 0.0474, 0.0406, 0.0315, 0.0288, 0.0265, 0.0247, 0.0197; wiki: 0.0645, 0.0361, 0.0249, 0.0208, 0.0193, 0.0158, 0.0150, 0.0135

### 3. Nonlinear two-sample test (MMD + energy distance)

On 4000 centered+renorm points/group (RBF bandwidth sigma=1.4190, median heuristic; 200-permutation null).

| statistic | random-sample (prior) | topic-matched (new) |
|---|---|---|
| cross MMD^2 (sigma above null) | 71σ | 37σ |
| energy distance (× within baseline) | 2.8× | 1.9× |
| permutation p-value | 0.0050 | 0.0050 |

Raw topic-matched two-sample values: cross MMD^2=0.000821 (within baseline 0.000017, null 0.000002±0.000022); cross energy=0.002774 (within baseline 0.001451).

### 4. Local kNN same-corpus purity (k=10)

| | random-sample (prior) | topic-matched (new) |
|---|---|---|
| mean purity | 0.9879 | 0.9234 |
| mixing-ratio baseline | 0.5000 | 0.5000 |
| excess over baseline | +0.4879 | +0.4234 |
| OA-query purity | 0.9971 | 0.9403 |
| wiki-query purity | 0.9787 | 0.9066 |

### Verdict (topic-matched)

**Topic-matched verdict: GENUINE NONLINEAR REGISTER RESIDUAL (residual persists).** Restricting Wikipedia to its NLP-relevant slice changes the picture markedly relative to the random sample.

**(1) How much was topic-distribution vs genuine corpus shift?** Local kNN same-corpus purity drops from 0.988 (random) to 0.923 (topic-matched), excess over the 0.5 baseline collapsing from +0.488 to +0.423 (~13% of the local clumping removed). The covariance mean principal angle (k=50) moves 55° → 41° (Δ-14°), and the cross MMD^2 σ-gap moves 71σ → 37σ (Δ-34σ). So a modest fraction of the prior residual was **topic-distribution mismatch**, not corpus/register.

**(2) Within-topic, after the offset:** the residual is a genuine nonlinear register residual — it persists. After per-corpus (own-mean) centering the means coincide (centered logistic AUC 0.534, no LDA deflation direction recovers it), and the covariance still differs materially (angle 41°), with kNN purity still elevated.

**(3) Revised LPP-transfer bound:** naive LPP-transfer will degrade: the within-topic covariance mismatch persists, so a wiki-learned projection is a poor basis for the OA cloud — covariance alignment (CORAL/CCA) or a joint formulation is needed. At 41° mean principal angle (k=50) and 37σ cross MMD^2, the within-topic covariance mismatch is large. Practical caveat: the pipeline removes the *global* full-store wiki offset, not the slice's own mean — for an NLP-relevant slice that leaves a residual offset (norm 0.39; full-store-centered logistic AUC 1.000), so a wiki→OA projection must mean-match per-slice (or jointly) before any covariance alignment can help.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.shift_characterization_topicmatched --wiki-store cache/wiki/store/enwiki_full --n-select 40000 --seed 0`.)*


## Cross-corpus alignment bake-off: raw vs CSLS dictionary, and CORAL / CCA

Two prior threads left a gap. (i) The §"Topic-matched re-test" battery established that after **per-slice mean-match** the wiki↔OA mean offset is gone but a **genuine nonlinear residual** persists (covariance angle ~41°, MMD² ~37σ, kNN purity ~0.92 — the clouds occupy largely disjoint local support), and explicitly flagged CORAL/CCA as the next test. (ii) The coocmap correspondence re-test found a strong shared-encoder cross-corpus **dictionary**, but its baseline was *CSLS'd*, leaving ambiguous whether the strong dictionary came from RAW cosine or the CSLS correction. This section resolves both, on the SAME per-slice-mean-matched slices.

- **Slices.** NLP-wiki top-3000 (cosine to the OA-NLP centroid) vs OA-NLP 3000 (closest to own centroid), `cache/wiki/store/wiki_400k`, seed 0. **Per-slice mean-match**: subtract each slice's OWN mean, then unit-normalize (NOT full-store centering — the NLP slice is non-representative). Post-mean-match residual mean norms: wiki=0.0046, OA=0.0050 (~0 ✓).

### Part A — dictionary: raw cosine vs CSLS, in isolation

Cross-corpus NN = argmax over the per-slice-mean-matched cosine `sim = Xw @ Xo.T`. CSLS subtracts `0.5·(topk-col-mean + topk-row-mean)`, k=10.

| scorer | mutual-NN rate | top-20 precision (same-concept) |
|---|---|---|
| raw cosine | 0.1853 | ~17/20 = **0.85** |
| CSLS (k=10) | 0.3727 | ~19/20 = **0.95** |

#### Top-20 RAW matches (by confidence)

```
  [+0.715] WIKI: Turkic languages
            ->  OA: A Survey of Turkic Language Modeling: Challenges, Resources and Future
  [+0.702] WIKI: Eurotra
            ->  OA: Eurotra: The Machine Translation Project of the European Communities
  [+0.698] WIKI: Cantonese
            ->  OA: Cantonese natural language processing in the transformers era: a surve
  [+0.682] WIKI: Arabic
            ->  OA: Building Arabic Corpora: Concepts, Methodologies, Tools, and Experimen
  [+0.678] WIKI: Natural language processing
            ->  OA: Trends In Natural Language Processing : Scope And Challenges
  [+0.661] WIKI: Word-sense disambiguation
            ->  OA: An Iterative Approach to Word Sense Disambiguation
  [+0.650] WIKI: Universal Networking Language
            ->  OA: Understanding Natural Language through the UNL Grammar Work- bench
  [+0.645] WIKI: Machine translation
            ->  OA: What's new in statistical machine translation
  [+0.639] WIKI: Natural language understanding
            ->  OA: Natural Language Processing
  [+0.633] WIKI: Probabilistic context-free grammar
            ->  OA: PCFG Induction for Unsupervised Parsing and Language Modelling
  [+0.629] WIKI: Parallel text
            ->  OA: Reviewed by
  [+0.626] WIKI: Head-driven phrase structure grammar
            ->  OA: Head-driven phrase structure grammar: linguistic approach, formal foun
  [+0.614] WIKI: WordNet
            ->  OA: Extensions, Enhancements, and New Perspectives on Wordnet
  [+0.606] WIKI: Bulgarian language
            ->  OA: Adding Linguistic Knowledge to NLP Tasks for Bulgarian: The Verb Parad
  [+0.606] WIKI: Dependency grammar
            ->  OA: Der Einfluss der Dependenzgrammatik auf die Computerlinguistik
  [+0.601] WIKI: Computational linguistics
            ->  OA: A Brief Introduction to Computational Linguistics
  [+0.596] WIKI: Arabic grammar
            ->  OA: Building Arabic Corpora: Concepts, Methodologies, Tools, and Experimen
  [+0.595] WIKI: Information retrieval
            ->  OA: Proceedings of the ACL-2000 workshop on Recent advances in natural lan
  [+0.593] WIKI: Machine learning
            ->  OA: Introduction to the Special Issue on Natural Language and Learning Mac
  [+0.591] WIKI: Document retrieval
            ->  OA: Proceedings of the ACL-2000 workshop on Recent advances in natural lan
```

#### Top-20 CSLS matches (by confidence)

```
  [+0.326] WIKI: Eurotra
            ->  OA: Eurotra: The Machine Translation Project of the European Communities
  [+0.291] WIKI: Universal Networking Language
            ->  OA: Understanding Natural Language through the UNL Grammar Work- bench
  [+0.291] WIKI: Turkic languages
            ->  OA: A Survey of Turkic Language Modeling: Challenges, Resources and Future
  [+0.287] WIKI: Metalanguage
            ->  OA: The Creation of a Corpus of English Metalanguage
  [+0.286] WIKI: Cantonese
            ->  OA: Cantonese natural language processing in the transformers era: a surve
  [+0.266] WIKI: Bulgarian language
            ->  OA: Adding Linguistic Knowledge to NLP Tasks for Bulgarian: The Verb Parad
  [+0.250] WIKI: Code-switching
            ->  OA: Challenges of Computational Processing of Code-Switching
  [+0.247] WIKI: Synthetic language
            ->  OA: morphogen: Translation into Morphologically Rich Languages with Synthe
  [+0.242] WIKI: Greeklish
            ->  OA: NLP for the Greek Language: A Brief Survey
  [+0.239] WIKI: Parallel text
            ->  OA: Reviewed by
  [+0.238] WIKI: Head-driven phrase structure grammar
            ->  OA: Head-driven phrase structure grammar: linguistic approach, formal foun
  [+0.217] WIKI: Decipherment
            ->  OA: UNRAVEL—A Decipherment Toolkit
  [+0.214] WIKI: Urdu
            ->  OA: Automated part-of-speech analysis of Urdu: conceptual and technical is
  [+0.214] WIKI: Latent semantic analysis
            ->  OA: Lexical disambiguation in machine translation with latent semantic ana
  [+0.213] WIKI: Greek language
            ->  OA: NLP for the Greek Language: A Brief Survey
  [+0.211] WIKI: Idiom
            ->  OA: ID10M: Idiom Identification in 10 Languages
  [+0.209] WIKI: Punjabi language
            ->  OA: Developing Punjabi Morphology, Corpus and Lexicon
  [+0.204] WIKI: Kazakh language
            ->  OA: Development and Evaluation of a Small Kazakh Language Corpus to Improv
  [+0.201] WIKI: Constrained writing
            ->  OA: Constraint-based Natural Language Generation: A Survey
  [+0.199] WIKI: Dependency grammar
            ->  OA: Der Einfluss der Dependenzgrammatik auf die Computerlinguistik
```

**Part A verdict — the strong dictionary is RAW cosine; CSLS *cleans* it but does not create it.** Raw per-slice-mean-matched cosine already produces a high-precision top-20 (~0.85): *Turkic languages* → *Turkic Language Modeling*, *Eurotra* → the *Eurotra* MT-project paper, *WordNet* → a *WordNet* paper, *HPSG* → an *HPSG* paper — most are exact-concept hits. The three raw misses are not scramble: one is a degenerate OA target title ("Reviewed by"), and two generic-IR wiki queries (*Information retrieval*, *Document retrieval*) both land on the same broad "Recent advances in NLP" ACL workshop. CSLS, applied in isolation, **improves** the dictionary to ~0.95: it removes exactly those two generic-workshop hub-landings (replacing them with *Code-switching* → *Code-Switching*, *Decipherment* → *Decipherment Toolkit*, *Latent semantic analysis* → *LSA in MT*, all clean), and roughly doubles the mutual-NN rate (0.185 → 0.373) by demoting the few OA hubs that the raw argmax kept handing back. The only surviving CSLS miss is the same "Reviewed by" junk target (a degenerate item, not a scoring failure). So the ~0.90-class dictionary the coocmap re-test reported was **fundamentally a property of raw shared-encoder cosine in the mean-matched space** — CSLS is a worthwhile hubness *correction* on top (it lifts precision ~0.85 → ~0.95 and mutual-NN ~2×), not the source of the signal. This disentangles what the coocmap re-test left ambiguous: the dictionary is real at the raw level, and CSLS is a clean +0.10 precision refinement.

### Part B — CORAL & CCA (anchored on raw mutual-NN pseudo-anchors)

Pseudo-anchors = the 556 raw mutual-NN pairs (the ~strong dictionary). CORAL whitens OA by its anchor covariance and recolors by wiki's (`Cov_OA^{-1/2}` then `Cov_wiki^{1/2}`, ε=0.001); CCA fits on the paired anchors and projects both corpora into the top-d canonical components. **Residual** = principal angle (k=50), cross MMD² σ-gap vs the within-corpus permutation null, and kNN same-corpus purity (k=10). Baseline = raw mean-matched (no alignment).

| alignment | mutual-NN | NN top-20 prec | angle (deg) | MMD² σ-gap | kNN purity (k=10) |
|---|---|---|---|---|---|
| raw mean-matched (baseline) | 0.1853 | ~0.85 | 47.3 | 18.0 | 0.9166 |
| CORAL | 0.1903 | ~0.95 | 32.5 | 81.8 | 0.7728 |
| CCA d=32 | 0.4253 | ~0.20 | 0.0 (k=32) | 3.4 | 0.5983 |
| CCA d=64 | 0.3873 | ~0.20 | 9.0 (k=50) | 2.5 | 0.6500 |
| CCA d=128 | 0.3453 | ~0.20 | 23.8 (k=50) | 3.3 | 0.7078 |

_Interpretation guard:_ CORAL aligns covariance by construction, so its angle dropping below the baseline 47.3° is **expected and not the success signal**. The real questions are whether **kNN purity falls toward 0.5** (baseline 0.917) and whether **NN precision improves**.

#### Top-20 CORAL matches (by confidence)

```
  [+0.837] WIKI: Turkic languages
            ->  OA: A Survey of Turkic Language Modeling: Challenges, Resources and Future
  [+0.806] WIKI: Arabic
            ->  OA: Building Arabic Corpora: Concepts, Methodologies, Tools, and Experimen
  [+0.790] WIKI: Cantonese
            ->  OA: Cantonese natural language processing in the transformers era: a surve
  [+0.764] WIKI: Universal Networking Language
            ->  OA: Understanding Natural Language through the UNL Grammar Work- bench
  [+0.733] WIKI: Eurotra
            ->  OA: Eurotra: The Machine Translation Project of the European Communities
  [+0.713] WIKI: WordNet
            ->  OA: Extensions, Enhancements, and New Perspectives on Wordnet
  [+0.713] WIKI: Greeklish
            ->  OA: NLP for the Greek Language: A Brief Survey
  [+0.710] WIKI: Linguistic typology
            ->  OA: Linguistic typology in natural language processing
  [+0.710] WIKI: Bulgarian language
            ->  OA: Adding Linguistic Knowledge to NLP Tasks for Bulgarian: The Verb Parad
  [+0.706] WIKI: Slavic languages
            ->  OA: Slavic Corpus and Computational Linguistics
  [+0.705] WIKI: Probabilistic context-free grammar
            ->  OA: PCFG Induction for Unsupervised Parsing and Language Modelling
  [+0.695] WIKI: Urdu
            ->  OA: Automated part-of-speech analysis of Urdu: conceptual and technical is
  [+0.694] WIKI: Natural language processing
            ->  OA: Trends In Natural Language Processing : Scope And Challenges
  [+0.692] WIKI: Idiom
            ->  OA: ID10M: Idiom Identification in 10 Languages
  [+0.685] WIKI: Information retrieval
            ->  OA: Proceedings of the ACL-2000 workshop on Recent advances in natural lan
  [+0.681] WIKI: Word-sense disambiguation
            ->  OA: An Iterative Approach to Word Sense Disambiguation
  [+0.681] WIKI: Ambiguity
            ->  OA: Ambiguity Resolution : An Analytical Study
  [+0.677] WIKI: Punjabi language
            ->  OA: Developing Punjabi Morphology, Corpus and Lexicon
  [+0.674] WIKI: Synthetic language
            ->  OA: morphogen: Translation into Morphologically Rich Languages with Synthe
  [+0.674] WIKI: Arabic grammar
            ->  OA: Building Arabic Corpora: Concepts, Methodologies, Tools, and Experimen
```

#### Top-20 CCA (d=128) matches (by confidence)

```
  [+0.606] WIKI: Z39.50
            ->  OA: 50th Annual Meeting of the Association for Computationalc○2012 The Ass
  [+0.587] WIKI: Eval
            ->  OA: Syntactic Features for Evaluation of Machine Translation
  [+0.582] WIKI: Low Franconian
            ->  OA: LOWRECORP: the Low-Resource NLG Corpus Building Challenge
  [+0.574] WIKI: Rebol
            ->  OA: Solution for Word Reordering English-Turkish SMT
  [+0.570] WIKI: Tsat language
            ->  OA: TyPTex : Inductive typological text classification by multivariate sta
  [+0.566] WIKI: Metasyntax
            ->  OA: Microsyntactic Annotation of Corpora and its Use in Computational Ling
  [+0.564] WIKI: Register (music)
            ->  OA: Using register-diversified corpora for general language studies
  [+0.554] WIKI: Dunglish
            ->  OA: COALA-A computational system for interlanguage analysis
  [+0.552] WIKI: Parallel translation
            ->  OA: Parallel texts: using translational equivalents in linguistic typology
  [+0.552] WIKI: Computational Linguistics (journal)
            ->  OA: The Oxford handbook of computational linguistics
  [+0.552] WIKI: Hybrid word
            ->  OA: Latest trends in hybrid machine translation and its applications
  [+0.548] WIKI: Statistical syllogism
            ->  OA: Statistical machine translation gains respect
  [+0.546] WIKI: Trigram
            ->  OA: Design and compilation of a specialized Spanish-German parallel corpus
  [+0.543] WIKI: JavaCC
            ->  OA: Stochastic iterative alignment for machine translation evaluation
  [+0.539] WIKI: Gender neutrality in languages with gendered third-person pronouns
            ->  OA: Comprehension Assistance Meets Machine Translation
  [+0.534] WIKI: Anacoluthon
            ->  OA: A statistical model for parsing and word-sense disambiguation
  [+0.534] WIKI: Czech language
            ->  OA: Prague Dependency Treebank as an exercise book of Czech
  [+0.532] WIKI: Ergodic (disambiguation)
            ->  OA: Finding Translation Examples for Under-Resourced Language Pairs or for
  [+0.532] WIKI: T-schema
            ->  OA: TyPTex : Inductive typological text classification by multivariate sta
  [+0.531] WIKI: Isocolon
            ->  OA: Predicting Pronoun Translation Using Syntactic, Morphological and Cont
```

**Part B verdict — linear second-moment alignment *moves* the local support but cannot win both metrics at once; the nonlinear residual is real and partly survives.**

*CORAL* does what it is built to do and a bit more: it pulls the covariance angle 47.3° → 32.5° (expected, per the guard — not the success signal) and, more interestingly, drops kNN same-corpus purity from the baseline 0.917 to **0.773** — the local support is genuinely *less* disjoint, roughly a third of the excess-over-0.5 removed. NN precision is **preserved** (~0.95, essentially the same concept list as raw with higher confidence values from the recoloring). But CORAL is not free: the cross MMD² σ-gap *rises* 18 → 82. Recoloring the entire OA cloud by a covariance estimated on only 556 anchors matches the 2nd moment while distorting the global density, so the RBF two-sample test sees a *larger* discrepancy even as the local neighborhoods mix better. CORAL helps local mixing without hurting correspondence, but does not "reduce the residual" by the MMD criterion — it relocates it.

*CCA* collapses the residual battery the most — purity 0.598 (d=32), 0.650 (d=64), 0.708 (d=128), and MMD² σ-gap down to ~2.5–3.4 (essentially the same-distribution floor) — and even raises mutual-NN sharply (0.185 → 0.425 at d=32). But this is **the trivial-by-construction collapse the guard warns about**, taken to its conclusion: CCA explicitly maximizes cross-corpus correlation on the anchors, so projecting both corpora into the canonical subspace forces the principal angle to ~0° (d=32) and squeezes the clouds together. The cost shows up where it matters — **NN top-20 correspondence precision craters to ~0.20**: the high-confidence CCA matches are dominated by spurious string/homonym artifacts (*Register (music)* → *register-diversified corpora*, *Hybrid word* → *hybrid machine translation*, *Statistical syllogism* → *statistical machine translation*, *Z39.50* → *50th Annual Meeting*), exactly the overfitting signature of fitting a low-rank canonical map on 556 noisy pseudo-anchors. CCA buys a pretty residual battery by discarding the very correspondence quality that motivated the exercise.

So: linear alignment **can** reduce the disjoint local support (CORAL: purity 0.92 → 0.77; CCA: → 0.60–0.71), confirming the residual is *not* irreducible by linear maps — but it cannot do so while *also* improving (or even preserving, for CCA) the cross-corpus correspondence. The nonlinear residual is real, and the bound from the topic-matched battery holds in its practical form: no linear second-moment alignment beats the raw/CSLS mean-matched dictionary on correspondence precision. CORAL is the only variant that improves local mixing without wrecking the dictionary, and even it inflates the global MMD discrepancy.

### Verdict

**(A) The strong cross-corpus dictionary is RAW shared-encoder cosine, not CSLS.** In the per-slice-mean-matched space, raw cosine already yields ~0.85 top-20 correspondence precision with mostly exact-concept hits; CSLS *refines* it to ~0.95 (and ~doubles mutual-NN, 0.185 → 0.373) by demoting a couple of generic-workshop hub-landings, but it does not create the signal. The coocmap re-test's ~0.90-class dictionary was therefore fundamentally a raw-cosine property; CSLS is a clean +0.10-precision hubness correction layered on top, not the source.

**(B) Neither CORAL nor CCA beats raw/CSLS on correspondence — the disjoint local support is *reduced* but not removed by linear alignment, and reducing it trades against the dictionary.** CORAL halves the kNN-purity excess (0.917 → 0.773, local support genuinely less disjoint) and preserves ~0.95 NN precision, but inflates the global MMD² σ-gap (18 → 82) by recoloring the whole OA cloud from a 556-anchor covariance — it *relocates* the residual rather than reducing it. CCA drives purity to 0.60–0.71 and MMD² to the same-distribution floor, but this is the trivial by-construction collapse (angle → 0°) and it craters NN precision to ~0.20 with string/homonym matches — it overfits the noisy pseudo-anchors. The topic-matched bound holds in practice: the cross-corpus residual is a genuine nonlinear register difference that linear second-moment alignment cannot resolve without sacrificing the very correspondence quality it is meant to improve. Use raw (or CSLS-refined) mean-matched cosine for the dictionary; do not bake CORAL/CCA into the cross-corpus path.

*(Reproduce: `conda run -p .conda python -m artifacts_scripts.wiki.cross_corpus_alignment_bakeoff --wiki-store cache/wiki/store/wiki_400k --n 3000 --csls-k 10 --seed 0`.)*
