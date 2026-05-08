# Dataset - IEEE-CIS Fraud Detection

## Source

Kaggle competition: [ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

Hosted by IEEE Computational Intelligence Society in partnership with Vesta Corporation, October 2019.

## Why not auto-downloaded

- Competition T&C requires explicit acceptance via the Kaggle web UI before the API will release the files. Programmatic acceptance is not supported.
- Total payload is ~700 MB compressed, ~1.7 GB uncompressed. Per Liora Phase 1 rules (>500 MB and behind competition acceptance) this is documented rather than auto-fetched.

## Download command

After accepting the competition rules at the Kaggle URL above:

```bash
cd /root/AI/liora_projects/14_ieee_fraud/data
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
rm ieee-fraud-detection.zip
```

Authentication is read from `~/.kaggle/kaggle.json` (chmod 600).

## Files after extraction

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `train_transaction.csv` | 590,540 | 394 | Training transactions, target `isFraud` |
| `train_identity.csv` | 144,233 | 41 | Identity attributes for ~24% of training transactions |
| `test_transaction.csv` | 506,691 | 393 | Test transactions, target hidden |
| `test_identity.csv` | 141,907 | 41 | Identity attributes for ~28% of test transactions |
| `sample_submission.csv` | 506,691 | 2 | `TransactionID, isFraud` template |

Identity tables join to transaction tables on `TransactionID`. Left join is the right pattern: most transactions have no identity row.

## Schema notes

- `TransactionDT`: integer seconds elapsed from a reference timestamp (not a calendar timestamp). To split chronologically, sort on this column. Roughly 6 months of activity in the train set, ~6 months more in the test set.
- `TransactionAmt`: float, in USD.
- `ProductCD`: 5 levels, anonymised.
- `card1`..`card6`: anonymised card-level identifiers (issuer bank, country, type, etc.).
- `addr1, addr2`: anonymised billing region / country.
- `dist1, dist2`: anonymised distances (e.g., billing-to-shipping).
- `P_emaildomain, R_emaildomain`: purchaser / recipient email domain.
- `C1..C14`: counting features (e.g., number of addresses associated with the card).
- `D1..D15`: time-delta features.
- `M1..M9`: match flags.
- `V1..V339`: rich Vesta-engineered features, mostly anonymised aggregations.
- `id_01..id_38, DeviceType, DeviceInfo`: identity / device context.

## Reference materials

- Competition page: https://www.kaggle.com/competitions/ieee-fraud-detection
- Vesta Corporation: https://trustvesta.com/
- Top public solution writeups (1st through 10th place) document feature engineering tricks for `card1` collisions, UID construction, and time-delta features. See `reports/references.md` and the discussion forum for context.

## Disk-space check before download

```bash
df -h /root/AI
# need ~3 GB free for download + extraction + intermediate caches
```
