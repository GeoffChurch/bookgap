# Eval option: show a representative title alongside keywords (D1)

In the ToC-reconstruction pairwise eval we render chapters blind, keywords-only (`--n-titles 0`),
because sample article titles were noisy. But this means a **content-correct, label-corrupted**
chapter gets tanked on a cosmetic artifact: e.g. Neuromorphic ch3's dominant work is a clean VCSEL
paper, yet its top-30 keywords are padded with junk hapaxes (`tesla, rechenfehlern, crayfish`), and
gemma penalized specificity on the *label*, not the content. See
[[curriculum-specificity-semantic-reweight]] for the full debris diagnosis (degenerate cluster +
fixed display-k).

**Idea:** render one **mass-top representative title** per chapter alongside the keywords (the
single highest-mass work is reliable, unlike a sample of N noisy ones). A judge seeing "VCSEL
dynamics for neuromorphic photonic systems" wouldn't call that chapter off-topic regardless of the
keyword tail.

**Open question (why this is backlogged, not done):** unclear whether we *want* this to influence
the eval. Showing titles measures "is the chapter's best work on-topic" rather than "is the chapter
label clean" — arguably fairer to curriculum *content*, but it also leaks more signal and changes
what the eval rewards. Decide deliberately: is the eval scoring curriculum *quality* (favor titles)
or the *rendered artifact a user reads* (keep keywords-only, and fix the labels upstream via dynamic
top-k / no-degenerate-clusters)? Don't wire it in until that's settled.
