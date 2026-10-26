# Data coverage by catchment

This is a snapshot of the files in `data/` generated on 2026-10-25 by `python -m processing.describe_data`.
Record counts and date ranges describe the files on disk; they are not a guarantee that observations are valid
or continuous between the endpoints. Catchments are named after their groundwater borehole.

## HDE observations

Each cell gives the first and last observation date and the number of CSV rows. `—` means that no matching
file was present locally.

| Catchment | Groundwater level | Rainfall | River flow |
|---|---|---|---|
| Ashridge Wood | 2004-10-22–2026-10-24; 193,208 | 2007-07-31–2026-10-24; 674,460 | 1992-10-27–2026-10-15; 1,190,962 |
| Barton Horkstow Road | 2005-10-04–2026-10-24; 670,186 | 2000-04-12–2026-10-24; 930,363 | 1988-05-12–2026-10-13; 1,347,152 |
| Chilgrove House | 1999-06-01–2026-10-19; 896,726 | 1999-10-02–2026-10-22; 948,696 | 1981-05-01–2026-10-15; 1,543,927 |
| Compton | 2002-02-16–2026-10-19; 216,270 | 1991-02-21–2026-10-24; 1,250,842 | 1981-05-01–2026-10-15; 1,593,926 |
| Dickley Down | 1957-10-01–2026-10-24; 2,422,209 | 1995-04-22–2026-10-24; 1,104,833 | 1992-01-01–2026-10-24; 1,220,690 |
| Farringdon Station | 2002-04-08–2026-10-24; 860,623 | 2011-05-31–2026-10-24; 540,029 | 1991-10-18–2026-10-15; 1,226,978 |
| Folly Farm | 2015-07-13–2026-10-24; 99,240 | 1999-08-26–2026-10-24; 952,439 | 1989-01-09–2026-10-15; 1,324,105 |
| Houndean Bottom | 1999-07-13–2026-10-24; 769,372 | 1999-10-02–2026-10-24; 948,926 | 1981-05-01–2026-10-15; 1,493,889 |
| Hursley | 2003-01-27–2026-10-19; 207,973 | 1985-01-24–2026-10-24; 1,463,824 | 1981-08-01–2021-11-12; 1,412,479 |
| Lanham Lane | 2014-04-01–2026-10-19; 109,334 | 1988-01-20–2026-10-24; 1,359,105 | 1981-07-28–2026-10-15; 1,585,479 |
| Lilley Bottom | 2005-06-21–2026-10-24; 187,686 | 1989-11-01–2026-10-24; 1,246,636 | 1987-02-02–2026-10-15; 1,391,977 |
| Little Bucket Farm | 1993-10-13–2026-10-19; 289,426 | 2000-03-22–2026-10-24; 932,360 | 1968-03-08–2026-10-15; 2,054,918 |
| Newlands Corner | 2010-03-12–2026-10-24; 220,008 | 1990-11-01–2025-02-25; 1,203,326 | 1987-07-01–2026-10-15; 1,377,699 |
| Northfield Farm | 2001-11-22–2026-10-24; 259,049 | 1979-09-14–2026-10-24; 1,651,901 | 2003-03-14–2026-10-15; 827,116 |
| Purleybury | 2001-12-11–2024-07-18; 615,385 | 1991-09-11–2026-10-24; 1,231,417 | 1981-01-01–2026-10-15; 1,605,410 |
| Rockley | 1998-11-25–2026-10-24; 260,719 | 1985-05-01–2023-02-20; 1,325,666 | 1987-07-01–2026-10-15; 1,377,699 |
| Sweeps Lane | 2005-02-04–2026-10-24; 787,495 | 1990-01-01–2024-08-16; 1,214,017 | 1990-04-10–2026-10-15; 1,280,390 |
| Wolverton | 1995-05-19–2026-10-24; 576,001 | 1998-03-11–2026-10-24; 1,003,591 | 1981-06-01–2026-10-15; 1,590,951 |

Stations used (nearest 15-minute stations; see `data/matching/chalk_matches.csv`):

| Catchment | Rainfall station | River flow station |
|---|---|---|
| Ashridge Wood | West Ilsley | Frilsham |
| Barton Horkstow Road | Goxhill | Laceby Beck |
| Chilgrove House | Chilgrove House | Graylingwell |
| Compton | Walderton | Westbourne |
| Dickley Down | Evershot | Sydling St Nicholas |
| Farringdon Station | East Worldham | Caker Stream |
| Folly Farm | Broughton | Chilbolton Total |
| Houndean Bottom | Plumpton | Lewes Winterbourne |
| Hursley | Otterbourne | Allbrook |
| Lanham Lane | Bishops Sutton | Drove Lane Main |
| Lilley Bottom | Whitwell | Whitwell |
| Little Bucket Farm | Kingsmill Down | Horton |
| Newlands Corner | Guildford | Shalford |
| Northfield Farm | Chieveley | Welford |
| Purleybury | Purley Oaks | Carshalton |
| Rockley | Marlborough | Poulton Farm |
| Sweeps Lane | Orpington | Crayford |
| Wolverton | Lower Standen | Crabble Mill |

HDE files are named for nominal 15-minute data, but actual sampling is not uniformly 15-minute: groundwater
series in particular include hourly stretches and gaps. The counts above are raw rows, not unique timestamps
or valid-value counts. Inspect and align each series before treating its date span as continuous coverage.

## Forecast data

GEFS counts are distinct initialization dates in the local files. GeoSmart member coverage counts issue dates
that have at least one ensemble member (`TimeSeriesID` other than 0) after the last history time; the
denominator is the issue dates present in the consolidated file (`data/geosmart/<prefix>/<Prefix>_forecasts.csv.gz`).
Issue dates with history only are not usable ensemble forecasts. Days that S3 does not hold are listed in
`<Prefix>_missing_days.txt` and are not counted.

| Catchment | GEFS initialization dates | GeoSmart member forecast dates |
|---|---|---|
| Ashridge Wood | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 999; file dates 2023-11-10–2026-10-24 |
| Barton Horkstow Road | 2020-10-01–2026-10-24; 2,215 (daily) | 345 / 1,694; file dates 2020-01-01–2026-10-24 |
| Chilgrove House | 2020-10-01–2026-10-23; 2,214 (daily) | 2,095 / 2,323; file dates 2020-01-01–2026-10-24 |
| Compton | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 1,405; file dates 2020-04-06–2026-10-24 |
| Dickley Down | 2020-10-01–2026-10-24; 2,215 (daily) | 295 / 1,663; file dates 2020-01-01–2026-10-24 |
| Farringdon Station | 2020-10-01–2026-10-24; 2,215 (daily) | 1,430 / 1,655; file dates 2020-01-01–2026-10-24 |
| Folly Farm | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 1,593; file dates 2020-01-01–2026-10-24 |
| Houndean Bottom | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 1,592; file dates 2020-01-01–2026-10-24 |
| Hursley | 2020-10-01–2026-10-24; 2,215 (daily) | 1,376 / 1,678; file dates 2020-01-01–2026-10-24 |
| Lanham Lane | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 1,590; file dates 2020-01-01–2026-10-24 |
| Lilley Bottom | 2020-10-01–2026-10-24; 2,215 (daily) | 1,428 / 1,653; file dates 2020-01-01–2026-10-24 |
| Little Bucket Farm | 2020-10-01–2026-10-24; 2,215 (daily) | 1,429 / 1,654; file dates 2020-01-01–2026-10-24 |
| Newlands Corner | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 1,589; file dates 2020-01-01–2026-10-24 |
| Northfield Farm | 2020-10-01–2026-10-24; 2,215 (daily) | 1,426 / 1,650; file dates 2020-01-01–2026-10-24 |
| Purleybury | 2020-10-01–2026-10-24; 2,215 (daily) | 639 / 639; file dates 2020-01-01–2021-12-09 |
| Rockley | 2020-10-01–2026-10-24; 2,215 (daily) | 596 / 1,585; file dates 2020-01-01–2026-10-24 |
| Sweeps Lane | 2020-10-01–2026-10-24; 2,215 (daily) | 1,007 / 1,602; file dates 2020-01-01–2026-10-24 |
| Wolverton | 2020-10-01–2026-10-24; 2,215 (daily) | 0 / 35; file dates 2020-01-01–2020-02-04 |

## Catchments usable for joint training

`python main.py all-chalk` needs, for every catchment, all three HDE series, GEFS, GeoSmart ensemble members and enough
origins where GEFS and GeoSmart both exist (complete encoder and target windows) for each purged
train/validation/test split, and a gauge that is not dry. Samples are daily forecast origins; the split
sizes below are the numbers of origins after the purge gaps.

| Catchment | Usable | Train | Validation | Test | Reason if excluded |
|---|---|---|---|---|---|
| Ashridge Wood | no | | | | no usable GeoSmart ensemble member forecasts |
| Barton Horkstow Road | no | | | | only 337 aligned samples, split sizes {'train': 276, 'val': 0, 'test': 61} (need 5 per split) |
| Chilgrove House | yes | 1104 | 244 | 274 | |
| Compton | no | | | | no usable GeoSmart ensemble member forecasts |
| Dickley Down | yes | 158 | 30 | 17 | |
| Farringdon Station | yes | 495 | 241 | 260 | |
| Folly Farm | no | | | | no usable GeoSmart ensemble member forecasts |
| Houndean Bottom | no | | | | no usable GeoSmart ensemble member forecasts |
| Hursley | no | | | | flow is ~zero in 100.0% of training targets (ephemeral/dry gauge; baselines score zero loss) |
| Lanham Lane | no | | | | no usable GeoSmart ensemble member forecasts |
| Lilley Bottom | yes | 342 | 157 | 204 | |
| Little Bucket Farm | yes | 561 | 242 | 259 | |
| Newlands Corner | no | | | | no usable GeoSmart ensemble member forecasts |
| Northfield Farm | yes | 321 | 238 | 232 | |
| Purleybury | no | | | | only 252 aligned samples, split sizes {'train': 194, 'val': 0, 'test': 24} (need 5 per split) |
| Rockley | yes | 74 | 35 | 24 | |
| Sweeps Lane | yes | 337 | 40 | 192 | |
| Wolverton | no | | | | no usable GeoSmart ensemble member forecasts |

8 of 18 catchments are usable.

## Completeness notes

- No catchment is missing any of the three HDE inputs.
- Several HDE records end well before the present (stations stopped reporting), which shortens the overlap with GeoSmart and GEFS; check the end dates in the HDE table.
- GeoSmart files with no ensemble members at all (history only): Ashridge Wood, Compton, Folly Farm, Houndean Bottom, Lanham Lane, Newlands Corner, Wolverton. These catchments have no usable GeoSmart forecasts, so they can only be modelled without GeoSmart.
- These counts describe the local snapshot. They do not establish common usable sample dates across input series; the pipeline must align timestamps and require complete target windows before training or evaluation.
