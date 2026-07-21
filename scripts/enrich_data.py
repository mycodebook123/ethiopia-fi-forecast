import pandas as pd

df = pd.read_csv('data/raw/ethiopia_fi_unified_data.csv')

if 'parent_id' not in df.columns:
    df['parent_id'] = pd.NA

new_rows = []

new_rows.append({
    'record_id': 'REC_0034', 'record_type': 'observation', 'pillar': 'GENDER',
    'indicator': 'Smartphone Ownership Gender Gap', 'indicator_code': 'GEN_GAP_SMARTPHONE',
    'indicator_direction': 'lower_better', 'value_numeric': 34, 'value_type': 'percentage', 'unit': '%',
    'observation_date': '2026-06-11', 'fiscal_year': 2026, 'gender': 'all', 'location': 'national',
    'source_name': 'GSMA Mobile Gender Gap Report 2026', 'source_type': 'research',
    'source_url': 'https://www.connectingafrica.com/digital-divide/sub-saharan-africa-s-mobile-gender-gap-narrows-further',
    'confidence': 'high', 'collected_by': 'Meklit Workineh Daba', 'collection_date': '2026-07-21',
    'original_text': 'Across the surveyed African countries, Ethiopia had the highest gender gaps in mobile Internet adoption (36%), smartphone ownership (34%) and overall mobile ownership (22%).',
    'notes': 'Ethiopia has the widest smartphone ownership gender gap among GSMA-surveyed African countries; relevant enabler/proxy for Access forecasting per Enrichment Guide Sheet C.'
})

new_rows.append({
    'record_id': 'REC_0035', 'record_type': 'observation', 'pillar': 'ACCESS',
    'indicator': 'Internet Penetration Rate', 'indicator_code': 'ACC_INTERNET_PEN',
    'indicator_direction': 'higher_better', 'value_numeric': 21.7, 'value_type': 'percentage', 'unit': '%',
    'observation_date': '2025-12-31', 'fiscal_year': 2025, 'gender': 'all', 'location': 'national',
    'source_name': 'DataReportal Digital 2026: Ethiopia', 'source_type': 'research',
    'source_url': 'https://datareportal.com/reports/digital-2026-ethiopia',
    'confidence': 'high', 'collected_by': 'Meklit Workineh Daba', 'collection_date': '2026-07-21',
    'original_text': "Ethiopia's internet penetration rate stood at 21.7 percent of the total population at the end of the year, with 29.5 million internet users as of October 2025.",
    'notes': 'Indirect/enabler indicator for digital payment usage per Enrichment Guide Sheet C (mobile internet access is a precondition for app-based digital payments).'
})

links = [
    dict(event='EVT_0001', indicator='Account Ownership Rate', code='ACC_OWNERSHIP', pillar='ACCESS',
         direction='increase', magnitude='medium', lag=12, basis='literature', country='Kenya',
         notes='Mobile money launches associated with medium-term account ownership gains per Suri & Jack (2016) Kenya M-Pesa evidence.'),
    dict(event='EVT_0001', indicator='Mobile Money Account Rate', code='ACC_MM_ACCOUNT', pillar='ACCESS',
         direction='increase', magnitude='high', lag=3, basis='empirical', country=None,
         notes='Direct: Telebirr is the primary driver of Ethiopia mobile money account growth (4.7% in 2021 to 9.45% in 2024).'),
    dict(event='EVT_0001', indicator='P2P Transaction Count', code='USG_P2P_COUNT', pillar='USAGE',
         direction='increase', magnitude='high', lag=6, basis='empirical', country=None,
         notes='Telebirr P2P transfers are the dominant use case per Enrichment Guide Sheet D (P2P dominance).'),
    dict(event='EVT_0002', indicator='Mobile Subscription Penetration', code='ACC_MOBILE_PEN', pillar='ACCESS',
         direction='increase', magnitude='medium', lag=12, basis='theoretical', country='Kenya',
         notes='New MNO market entry historically increases mobile subscription penetration via competition/pricing effects.'),
    dict(event='EVT_0003', indicator='Mobile Money Account Rate', code='ACC_MM_ACCOUNT', pillar='ACCESS',
         direction='increase', magnitude='high', lag=6, basis='empirical', country=None,
         notes='M-Pesa entry directly contributed to mobile money account growth alongside Telebirr.'),
    dict(event='EVT_0003', indicator='M-Pesa Registered Users', code='USG_MPESA_USERS', pillar='USAGE',
         direction='increase', magnitude='high', lag=3, basis='empirical', country=None,
         notes='Direct causal link: registered users measured from this launch.'),
    dict(event='EVT_0004', indicator='Fayda Digital ID Enrollment', code='ACC_FAYDA', pillar='ACCESS',
         direction='increase', magnitude='high', lag=0, basis='empirical', country=None,
         notes='Direct: enrollment is a direct output of program rollout.'),
    dict(event='EVT_0004', indicator='Account Ownership Rate', code='ACC_OWNERSHIP', pillar='ACCESS',
         direction='increase', magnitude='low', lag=18, basis='theoretical', country='India',
         notes='Digital ID reduces KYC friction for account opening; long lag reflects gradual integration into financial institution onboarding, per India Aadhaar literature.'),
    dict(event='EVT_0005', indicator='Account Ownership Rate', code='ACC_OWNERSHIP', pillar='ACCESS',
         direction='mixed', magnitude='low', lag=6, basis='theoretical', country=None,
         notes='FX liberalization is a macro shock; near-term effect on account ownership is ambiguous (inflation may deter saving, but banking system stabilization may encourage formal account use).'),
    dict(event='EVT_0006', indicator='ATM Transaction Count', code='USG_ATM_COUNT', pillar='USAGE',
         direction='decrease', magnitude='medium', lag=0, basis='empirical', country=None,
         notes='Milestone reflects concurrent substitution away from ATM cash withdrawal toward P2P digital transfer, not a forward-looking driver.'),
    dict(event='EVT_0007', indicator='P2P Transaction Count', code='USG_P2P_COUNT', pillar='USAGE',
         direction='increase', magnitude='medium', lag=3, basis='theoretical', country=None,
         notes='Interoperability between M-Pesa and EthSwitch network expected to increase cross-platform P2P transaction volume.'),
    dict(event='EVT_0008', indicator='Account Ownership Rate', code='ACC_OWNERSHIP', pillar='ACCESS',
         direction='increase', magnitude='medium', lag=12, basis='theoretical', country='Brazil',
         notes='National instant payment rails (cf. Brazil Pix, India UPI) are associated with subsequent account ownership growth as merchants/consumers open accounts to access the rail.'),
    dict(event='EVT_0009', indicator='Account Ownership Rate', code='ACC_OWNERSHIP', pillar='ACCESS',
         direction='increase', magnitude='medium', lag=24, basis='theoretical', country=None,
         notes='NFIS-II sets the 70% account ownership target and coordinates the policy environment (agent licensing, interoperability mandates) that enables other events in this list; long lag reflects multi-year strategy implementation.'),
    dict(event='EVT_0010', indicator='Mobile Subscription Penetration', code='ACC_MOBILE_PEN', pillar='ACCESS',
         direction='decrease', magnitude='low', lag=3, basis='theoretical', country=None,
         notes='Price increases are expected to modestly dampen new subscription growth due to price elasticity of demand.'),
]

valid_event_ids = set(df[df['record_type'] == 'event']['record_id'])

for i, link in enumerate(links, start=1):
    assert link['event'] in valid_event_ids, f"Unknown event record_id: {link['event']}"
    parent_record_id = link['event']
    new_rows.append({
        'record_id': f'IL_{i:04d}',
        'record_type': 'impact_link',
        'parent_id': parent_record_id,
        'pillar': link['pillar'],
        'related_indicator': link['code'],
        'relationship_type': 'direct' if link['basis'] == 'empirical' else 'indirect',
        'impact_direction': link['direction'],
        'impact_magnitude': link['magnitude'],
        'lag_months': link['lag'],
        'evidence_basis': link['basis'],
        'comparable_country': link['country'],
        'collected_by': 'Meklit Workineh Daba',
        'collection_date': '2026-07-21',
        'notes': link['notes'],
    })

new_df = pd.DataFrame(new_rows)
enriched = pd.concat([df, new_df], ignore_index=True)

enriched.to_csv('data/processed/ethiopia_fi_unified_data_enriched.csv', index=False)

print(f"Original rows: {len(df)}")
print(f"New rows added: {len(new_rows)}")
print(f"Enriched total: {len(enriched)}")
print()
print("New record_type counts:")
print(enriched['record_type'].value_counts())
print()
print("Impact links with unresolved parent_id (should be 0):")
print(enriched[(enriched['record_type']=='impact_link') & (enriched['parent_id'].isna())])
