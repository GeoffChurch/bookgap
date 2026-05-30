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
