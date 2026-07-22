import pandas as pd

df = pd.read_csv('data/processed/ethiopia_fi_unified_data_enriched.csv')

new_row = {
    'record_id': 'REC_0036', 'record_type': 'observation', 'pillar': 'USAGE',
    'indicator': 'Digital Payment Adoption Rate', 'indicator_code': 'USG_DIGITAL_PAYMENT',
    'indicator_direction': 'higher_better', 'value_numeric': 35, 'value_type': 'percentage', 'unit': '%',
    'observation_date': '2024-12-31', 'fiscal_year': 2024, 'gender': 'all', 'location': 'national',
    'source_name': 'Global Findex 2024', 'source_type': 'survey',
    'source_url': 'https://www.worldbank.org/en/publication/globalfindex',
    'confidence': 'medium', 'collected_by': 'Meklit Workineh Daba', 'collection_date': '2026-07-22',
    'original_text': 'Made or received digital payment: ~35% (2024), as cited in the Week 11 challenge brief overview.',
    'notes': 'Added during Task 4 when it became apparent the dataset lacked a direct indicator matching the Findex-defined Usage headline metric (Digital Payment Adoption Rate) despite the brief explicitly citing this figure. Single-point observation (no trend); confidence set to medium given secondary sourcing (brief text, not directly verified against original Findex microdata release).'
}

df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df.to_csv('data/processed/ethiopia_fi_unified_data_enriched.csv', index=False)
print(f"Added REC_0036. New total rows: {len(df)}")
print(df[df['record_type']=='observation']['indicator_code'].value_counts().get('USG_DIGITAL_PAYMENT', 0), "USG_DIGITAL_PAYMENT rows now present")
