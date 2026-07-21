import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Task 1: Data Exploration and Enrichment
## Ethiopia Financial Inclusion Forecasting

This notebook explores the starter dataset (`ethiopia_fi_unified_data.csv`) and
`reference_codes.csv`, understands the unified schema, and documents a schema
deviation found during exploration before moving into enrichment.
"""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

df = pd.read_csv('../data/raw/ethiopia_fi_unified_data.csv')
ref = pd.read_csv('../data/raw/reference_codes.csv')

print(f"Main dataset shape: {df.shape}")
print(f"Reference codes shape: {ref.shape}")
df.head()
"""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Schema Overview"))

cells.append(nbf.v4.new_code_cell(
"""print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""df.dtypes
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 2. Record Counts by record_type

The brief states the dataset should contain 30 observations, 10 events, 14 impact_links,
and 3 targets. Let's verify.
"""
))

cells.append(nbf.v4.new_code_cell(
"""record_type_counts = df['record_type'].value_counts()
print(record_type_counts)
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""### Schema Deviation Found

The dataset as provided contains **0 `impact_link` records**, not the 14 described in the
brief. There is also **no `parent_id` column** in the schema at all - the brief's instructions
reference linking impact_links to events "via parent_id", but this field does not exist.

**Interpretation:** Task 1's instruction to "Enrich the Dataset... Additional impact_links:
Can you identify other relationships between events and indicators that should be modeled?"
suggests that building the impact_link records is an intentional enrichment exercise for the
trainee, not a data entry omission. This is documented in `data/data_enrichment_log.md` and
addressed in the Enrichment section below, where we construct impact_link rows referencing
events by their `indicator_code`-style event code (e.g. `EVT_TELEBIRR`) via a new
`related_event_code` field, since no `parent_id` column exists to reuse.
"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Counts by pillar, source_type, and confidence"))

cells.append(nbf.v4.new_code_cell(
"""print("By pillar:")
print(df['pillar'].value_counts(dropna=False))
print()
print("By source_type:")
print(df['source_type'].value_counts(dropna=False))
print()
print("By confidence:")
print(df['confidence'].value_counts(dropna=False))
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""Note: `event` records have a blank `pillar` by design (per the brief: "Events are categorized
by type... but are NOT pre-assigned to pillars. Their effects on specific indicators are
captured through impact_link records. This keeps the data unbiased.")
"""
))

cells.append(nbf.v4.new_markdown_cell("## 4. Temporal Range of Observations"))

cells.append(nbf.v4.new_code_cell(
"""obs = df[df['record_type'] == 'observation'].copy()
obs['observation_date'] = pd.to_datetime(obs['observation_date'])

print(f"Earliest observation: {obs['observation_date'].min()}")
print(f"Latest observation: {obs['observation_date'].max()}")
print()
print("Observations per year:")
print(obs['observation_date'].dt.year.value_counts().sort_index())
"""
))

cells.append(nbf.v4.new_markdown_cell("## 5. Unique Indicators and Coverage"))

cells.append(nbf.v4.new_code_cell(
"""indicator_coverage = obs.groupby(['indicator_code', 'indicator']).agg(
    n_observations=('value_numeric', 'count'),
    years=('observation_date', lambda x: sorted(x.dt.year.unique().tolist())),
    pillar=('pillar', 'first')
).reset_index()

indicator_coverage
"""
))

cells.append(nbf.v4.new_markdown_cell("## 6. Events Catalogued"))

cells.append(nbf.v4.new_code_cell(
"""events = df[df['record_type'] == 'event'].copy()
events[['record_id', 'indicator_code', 'indicator', 'category', 'observation_date', 'source_name']]
"""
))

cells.append(nbf.v4.new_markdown_cell("## 7. Targets"))

cells.append(nbf.v4.new_code_cell(
"""targets = df[df['record_type'] == 'target'].copy()
targets[['record_id', 'indicator', 'indicator_code', 'value_numeric', 'observation_date', 'notes']]
"""
))

cells.append(nbf.v4.new_markdown_cell("## 8. Reference Codes Overview"))

cells.append(nbf.v4.new_code_cell(
"""ref.groupby('field')['code'].apply(list)
"""
))

nb['cells'] = cells

with open('notebooks/01_data_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/01_data_exploration.ipynb")
