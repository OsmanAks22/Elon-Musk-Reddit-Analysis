# Reddit Sentiment & Topic Analysis: Elon Musk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OsmanAks22/Elon-Musk-Reddit-Analysis/blob/main/Elon_Musk_Reddit_Analysis.ipynb)

An NLP analysis of how five Reddit communities talk about Elon Musk. Posts are collected with the Reddit API (PRAW) and analysed with VADER sentiment, bigram frequencies and LDA topic modelling. The findings are written up as advice for a PR team.

> Individual coursework for **7BUIS025W Web & Social Media Analytics**, MSc Data Science & Analytics, University of Westminster (April 2026).

**Tech stack:** Python · PRAW (Reddit API) · pandas · NLTK · VADER · scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`) · matplotlib · seaborn · wordcloud

## Problem

A communications team wants to know how Reddit sees its client. Which communities are friendly or hostile, what people actually talk about, and how the conversation has changed over time.

## Data

- **Source:** the Reddit API through PRAW. The query "Elon Musk" was run in r/elonmusk, r/spacex, r/tesla, r/technology and r/worldnews, with up to 150 posts per subreddit, newest first.
- **Collected:** 604 raw posts. After removing duplicate IDs and posts whose cleaned text is 10 characters or shorter, **603 posts** remain, spanning 14 May 2014 to 23 April 2026.
  - r/tesla returned only 4 posts, because its moderators restrict outside search; the other four subreddits each hit the 150-post cap.
  - 498 of the 603 posts have no body text (link or image posts).
- **Not included in this repository.** The original April 2026 sample was not saved, so the numbers below can be checked against the figures but not recomputed. The notebook now caches what it collects to `data/reddit_posts.csv` (git-ignored, no usernames), so a new run can be repeated offline. A fresh sample will differ from the original, because Reddit search favours recent posts.

## Method

1. **Cleaning:** lower-casing, removal of URLs and non-letters, NLTK English stop-words, and dropping tokens of 2 characters or fewer. Title and body are combined into one `full_text` field.
2. **Exploration:** posts per subreddit, yearly volume, score distribution by subreddit, score against comments, word frequencies and a word cloud.
3. **Sentiment:** VADER compound score on post titles, labelled positive if ≥ 0.05, negative if ≤ −0.05, and neutral otherwise.
4. **Phrases:** the top 20 bigrams from `CountVectorizer(ngram_range=(2, 2))`.
5. **Topics:** LDA with 5 topics on a 1,000-term vocabulary (`min_df=5`, `max_df=0.7`). Four and six topics were also tried.

## Results

All numbers come from the original April 2026 run, as reported in the coursework report and shown in its figures.

- **Sentiment is evenly split:** 170 posts positive (28.2%), 151 negative (25.0%) and 282 neutral (46.8%). The mean compound score is 0.013.
- **The communities differ sharply.**
  - r/spacex is the friendliest, with 12.6% of posts negative.
  - r/elonmusk is the most positive, with 34.5% of posts positive.
  - r/worldnews and r/technology are the most hostile, each with 32.7% of posts negative.
- **Volume jumped in 2025:** 289 posts, against 43 in 2024 (6.7×) and 94 in 2022, the previous peak (3.1×). Because Reddit search favours recent posts, this is a trend in what search returns, not a measured change in total discussion.
- **Two separate conversations:** reading the five LDA topics, they fall into two groups. One is engineering (SpaceX launches, rocket hardware, Starbase). The other is political and platform news (political disputes, X, Starlink, Tesla). This grouping is an interpretation of the topics, not a separate clustering step.
- **Engagement is very skewed:** the mean post score is 3,291.94 against a median of 139, and the maximum is 78,015.

<p align="center">
  <img src="figures/fig7_sentiment_by_subreddit.png" width="80%" alt="Sentiment composition per subreddit">
</p>
<p align="center">
  <img src="figures/fig2_yearly_trend.png" width="80%" alt="Yearly volume of posts">
</p>
<p align="center">
  <img src="figures/fig10_lda_topics.png" width="100%" alt="Top 10 words for each of the five LDA topics">
</p>

All 11 figures from the original run are in [`figures/`](figures/).

## Limitations

- Reddit search returns recent and popular posts, not a random sample. The yearly trend therefore shows what search returns, not the true volume of discussion.
- r/tesla contributed only 4 posts, so the analysis says little about Tesla-specific sentiment.
- VADER scores headline-style titles; sarcasm and domain terms can mislead it.
- The original raw sample was not kept, so the published numbers cannot be recomputed from this repository.

## How to run

You need a Reddit "script" app (created at <https://www.reddit.com/prefs/apps>). The notebook reads its credentials from environment variables. With credentials, it collects posts and caches them to `data/reddit_posts.csv`. Without credentials, it loads that cache (or the file named in `REDDIT_POSTS_CSV`), and stops with a clear message if neither is available.

**Locally (Python 3.12):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export REDDIT_CLIENT_ID=...        # from your Reddit app
export REDDIT_CLIENT_SECRET=...
jupyter notebook Elon_Musk_Reddit_Analysis.ipynb
```

**Colab:** add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` as Colab secrets (key icon in the sidebar). Then run this before the notebook's second cell:

```python
import os
from google.colab import userdata
os.environ["REDDIT_CLIENT_ID"] = userdata.get("REDDIT_CLIENT_ID")
os.environ["REDDIT_CLIENT_SECRET"] = userdata.get("REDDIT_CLIENT_SECRET")
```

## Repository structure

```
Elon_Musk_Reddit_Analysis.ipynb   collection, cleaning, EDA, sentiment, bigrams, LDA
figures/                          the 11 figures from the original run
requirements.txt                  package versions used for the October 2026 check
data/                             created at run time; git-ignored
```

## Changes since submission

- The notebook file was renamed to fix a typo (`Anlysis` → `Analysis`).
- Credentials are now read from environment variables instead of empty strings in the code.
- `plt.boxplot(labels=...)` was changed to `tick_labels=...`. Matplotlib renamed the argument in 3.9 and removed the old name in 3.11, so the Figure 3 cell crashed on current versions.
- Collected posts are cached to CSV so the analysis can be re-run offline. The `author` field is no longer collected: it was never used, and it is personal data.
- `datetime.utcfromtimestamp` (deprecated since Python 3.12) was replaced with a timezone-aware equivalent that gives the same UTC timestamps.
- The unused `pyLDAvis` and `gensim` installs and the unused NLTK `punkt` download were removed.
- The coursework report links to `github.com/OsmanAks22/Elon_Musk_Reddit_Analysis` (underscores). The correct address is `github.com/OsmanAks22/Elon-Musk-Reddit-Analysis`.
