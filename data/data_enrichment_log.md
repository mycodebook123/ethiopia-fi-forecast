# Data Enrichment Log — Task 1

Author: Meklit Workineh Daba
Date: 2026-07-21

## Schema Deviation Found

The starter dataset (`data/raw/ethiopia_fi_unified_data.csv`) as distributed contains **30
observations, 10 events, 3 targets, and 0 impact_link records**, despite the challenge brief
stating the dataset should include 14 impact_links. There is also **no `parent_id` column**
in the schema as distributed, despite the brief's repeated instruction that impact_links
"link via parent_id."

Interpretation: Task 1's own instructions ask "Additional impact_links: Can you identify other
relationships between events and indicators that should be modeled?" — read together with the
missing column and zero pre-populated rows, this indicates that constructing the impact_link
records is an intended enrichment exercise, not a data omission. We treat it as such below,
re-adding the `parent_id` column and populating it only for new impact_link rows (values are
the `record_id` of the referenced event, e.g. `EVT_0001`).

## New Observations Added (2)

| record_id | indicator | value | source | confidence |
|---|---|---|---|---|
| REC_0034 | Smartphone Ownership Gender Gap | 34% | GSMA Mobile Gender Gap Report 2026 | high |
| REC_0035 | Internet Penetration Rate | 21.7% (Dec 2025) | DataReportal Digital 2026: Ethiopia | high |

**REC_0034** — Ethiopia has the widest smartphone ownership gender gap (34%) among GSMA's
surveyed African countries, alongside a 36% mobile internet adoption gap and 22% overall mobile
ownership gap. Source: https://www.connectingafrica.com/digital-divide/sub-saharan-africa-s-mobile-gender-gap-narrows-further
Relevant per Enrichment Guide Sheet C (indirect/enabler correlation — gender gap in device
ownership constrains Access growth).

**REC_0035** — Ethiopia had 29.5 million internet users as of October 2025 (21.7% penetration).
Source: https://datareportal.com/reports/digital-2026-ethiopia (citing ITU/GSMA Intelligence).
Relevant per Enrichment Guide Sheet C — mobile internet access is a precondition for app-based
digital payment usage, making it a leading indicator candidate for the Usage pillar.

## New impact_link Records Added (14)

Built to fill the confirmed gap described above. All 10 existing events now have at least one
modeled effect on an indicator. Where Ethiopia-specific pre/post evidence exists (e.g. Telebirr's
direct effect on mobile money account rates, M-Pesa's effect on its own registered users), the
link is marked `evidence_basis: empirical`. Where Ethiopian data is insufficient and we relied on
comparable-country evidence (per the brief's Task 3 guidance), the link is marked `literature` or
`theoretical`, with `comparable_country` populated (mainly Kenya, drawing on Suri & Jack (2016)
"The Long-Run Poverty and Gender Impacts of Mobile Money"; also India and Brazil for digital
ID / instant payment rail comparisons).

Full table with direction, magnitude, lag, and reasoning is in
`data/processed/ethiopia_fi_unified_data_enriched.csv` (record_type == 'impact_link') and is
explored in `notebooks/01_data_exploration.ipynb`.

Summary of events and how many indicators each was linked to:

| event | category | # links |
|---|---|---|
| EVT_0001 Telebirr Launch | product_launch | 3 |
| EVT_0002 Safaricom Commercial Launch | market_entry | 1 |
| EVT_0003 M-Pesa Ethiopia Launch | product_launch | 2 |
| EVT_0004 Fayda Digital ID Rollout | infrastructure | 2 |
| EVT_0005 FX Liberalization | policy | 1 |
| EVT_0006 P2P/ATM Crossover | milestone | 1 |
| EVT_0007 M-Pesa EthSwitch Integration | partnership | 1 |
| EVT_0008 EthioPay Launch | infrastructure | 1 |
| EVT_0009 NFIS-II Strategy Launch | policy | 1 |
| EVT_0010 Safaricom Price Increase | pricing | 1 |

## Known Limitations

- Impact magnitudes and lags for `theoretical`/`literature`-basis links are estimates informed
  by comparable-country evidence and domain reasoning, not fitted from Ethiopian data (there are
  too few Findex survey points to fit event-response models empirically). These should be treated
  as priors to be refined in Task 3, not ground truth.
- The two new observations added are national aggregate figures; deeper enrichment (Findex
  microdata gender/urban-rural splits beyond what's already present, agent density, POS terminal
  counts) was not pursued further due to time constraints ahead of the final submission deadline.
