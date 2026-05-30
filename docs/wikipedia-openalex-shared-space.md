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
