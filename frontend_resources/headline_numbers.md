| Stage | Rows | Note |
|---|---:|---|
| Raw MSK-IMPACT | 54,331 | 45 columns |
| With survival labels | 50,044 | OS time + status non-null |
| Time ∈ (0, 600) mo | 49,627 | drop zero / impossible times |
| Patient-level | 43,673 | 1 sample / patient (Primary preferred) |
| Train (80%) | 34,938 | stratified by status, event rate ≈ 38.6% |
| Test (20%) | 8,735 | stratified by status, event rate ≈ 38.6% |
| Final modelling features | 17 | after dropping IDs / HLA / >40% missing; covers 16 cancer-type buckets (top-15 + Other) |
