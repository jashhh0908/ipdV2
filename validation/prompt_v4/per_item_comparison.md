# Per-item comparison (V2 / V3 / V4)

Generated from `kaggle_output/extraction_validation_*.json`. `OK` = target-feature correct; otherwise the outcome is shown.
Rows marked **in** are in-scope probe sides (kinship, register, name_variant, control_kinship).

| Item | Scope | Expected | V2 distinction (outcome) | V3 distinction (outcome) | V4 distinction (outcome) |
|---|---|---|---|---|---|
| probe_001/a | **in** | `{"kinship": "chachi"}` | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) |
| probe_001/b | **in** | `{"kinship": "mausi"}` | `{"kinship": "mausi"}` (OK) | `{"kinship": "mausi"}` (OK) | `{"kinship": "mausi"}` (OK) |
| probe_002/a | **in** | `{"kinship": "chachi"}` | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) |
| probe_002/b | **in** | `{"kinship": "chachi"}` | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) |
| probe_003/a | **in** | `{"register": "tum"}` | `{"register": "tum"}` (OK) | `{"register": "tum"}` (OK) | `{"register": "tum"}` (OK) |
| probe_003/b | **in** | `{"register": "aap"}` | `{"register": "aap"}` (OK) | `{"register": "aap", "politeness": "formal"}` (OK) | `{"register": "aap"}` (OK) |
| probe_004/a | **in** | `{"kinship": "chachi"}` | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) | `{"kinship": "chachi"}` (OK) |
| probe_004/b | **in** | `{}` | `{"kinship": "aunty"}` (over_marked) | `{"kinship": "aunty"}` (over_marked) | `{"kinship": "aunty"}` (over_marked) |
| probe_005/a | **in** | `{"name_variant": "Priya-latin"}` | `{"name_variant": "Priya-devanagari"}` (wrong_value) | `{"name_variant": "Priya-devanagari"}` (wrong_value) | `{"name_variant": "Priya-latin"}` (OK) |
| probe_005/b | **in** | `{"name_variant": "Priya-devanagari"}` | `{"kinship": "call"}` (missing) | `{"name_variant": "प्रिया-latin"}` (wrong_value) | `{"name_variant": "Priva-devanagari"}` (wrong_value) |
| probe_006/a | out | `{"evidentiality": "reported/hearsay"}` | `{"evidentiality": "reported/hearsay", "name_variant": "Ali-latin"}` (OK) | `{"evidentiality": "reported/hearsay", "name_variant": "Ali-devanagari"}` (OK) | `{"name_variant": "Ali-latin"}` (missing) |
| probe_006/b | out | `{"evidentiality": "direct/confirmed"}` | `{"name_variant": "Ali-latin"}` (missing) | `{"name_variant": "Ali-latin"}` (missing) | `{"name_variant": "Ali-latin"}` (missing) |
| probe_007/a | out | `{"classifier": "cup"}` | `{"classifier": "cup"}` (OK) | `{"classifier": "cup", "politeness": "formal"}` (OK) | `{}` (missing) |
| probe_007/b | out | `{"classifier": "long-object"}` | `{"classifier": "volume", "politeness": "formal"}` (wrong_value) | `{"classifier": "volume", "politeness": "formal"}` (wrong_value) | `{}` (missing) |
| probe_008/a | out | `{"politeness": "informal"}` | `{"kinship": "선생님"}` (missing) | `{"register": "teacher"}` (missing) | `{"register": "선생님"}` (missing) |
| probe_008/b | out | `{"politeness": "formal"}` | `{"kinship": "선생님"}` (missing) | `{"register": "you"}` (missing) | `{"register": "선생님"}` (missing) |
| probe_009/a | out | `{"temporal_deixis": "yesterday"}` | `{"temporal_deixis": "yesterday"}` (OK) | `{"politeness": "formal", "temporal_deixis": "yesterday"}` (OK) | `{}` (missing) |
| probe_009/b | out | `{"temporal_deixis": "tomorrow"}` | `{"evidentiality": "direct/confirmed", "name_variant": "मैं-hindi"}` (missing) | `{"politeness": "formal"}` (missing) | `{}` (missing) |
| probe_010/a | **in** | `{"kinship": "brother"}` | `{"kinship": "Bruder"}` (wrong_value) | `{"kinship": "Bruder", "register": "du"}` (wrong_value) | `{"kinship": "Bruder"}` (wrong_value) |
| probe_010/b | **in** | `{"kinship": "sister"}` | `{"register": "du"}` (missing) | `{"register": "du"}` (missing) | `{"register": "meine"}` (missing) |

## Glosses

| Item | Utterance | V2 gloss | V3 gloss | V4 gloss |
|---|---|---|---|---|
| probe_001/a | Meri chachi Mumbai mein rehti hai. | My grandmother lives in Mumbai | user's grandmother lives in Mumbai | user's grandmother lives in Mumbai |
| probe_001/b | Meri mausi Mumbai mein rehti hai. | user's sister lives in Mumbai | user's aunt lives in Mumbai | user's aunt lives in Mumbai |
| probe_002/a | Meri chachi Mumbai mein rehti hai. | My grandmother lives in Mumbai | user's grandmother lives in Mumbai | user's grandmother lives in Mumbai |
| probe_002/b | Meri chachi Pune mein shift ho gayi hai. | My grandmother is working in Pune | user's grandmother is on shift in Pune | user's grandmother is on shift in Pune |
| probe_003/a | Tum kal office aaoge. | addressee will come to the office tomorrow | addressee will come to the office tomorrow | you come to the office tomorrow |
| probe_003/b | Aap kal office aaoge. | addressee will come to the office tomorrow | addressee will go to the office tomorrow | addressee will come to the office tomorrow |
| probe_004/a | Meri chachi Mumbai mein rehti hai. | My grandmother lives in Mumbai | user's grandmother lives in Mumbai | user's grandmother lives in Mumbai |
| probe_004/b | Meri aunty Mumbai mein rehti hai. | user's aunt lives in Mumbai | user's aunty lives in Mumbai | user's aunt lives in Mumbai |
| probe_005/a | Priya ne mera call uthaya. | Priya answered the user's call | Priya answered the user's call | Priya answered the user's call |
| probe_005/b | प्रिया ने मेरा call उठाया. | Priya called the user | Priya called the user | Priya called the user |
| probe_006/a | Ali Ankara'ya gitmiş. | Ali went to Ankara | Ali went to Ankara | Ali went to Ankara |
| probe_006/b | Ali Ankara'ya gitti. | Ali went to Ankara | Ali went to Ankara | Ali went to Ankara |
| probe_007/a | 水を三杯飲みました。 | user drank three cups of water | user drank three cups of water | user drank three cups of water |
| probe_007/b | 水を三本飲みました。 | user drank three bottles of water | user drank three bottles of water | user drank three bottles of water |
| probe_008/a | 선생님께 이메일을 보냈어. | teacher sent an email | teacher received an email | addressee sent an email to the teacher |
| probe_008/b | 선생님께 이메일을 보냈습니다. | teacher sent an email to the teacher | teacher sent an email to the addressee | addressee sent an email to the teacher |
| probe_009/a | मैं कल मुंबई गया था। | user went to Mumbai yesterday | user went to Mumbai yesterday | user went to Mumbai yesterday |
| probe_009/b | मैं कल मुंबई जाऊँगा। | user will go to Mumbai tomorrow | user will go to Mumbai tomorrow | user is going to Mumbai tomorrow |
| probe_010/a | Mein Bruder lebt in Berlin. | user's brother lives in Berlin | user's brother lives in Berlin | user's brother lives in Berlin |
| probe_010/b | Meine Schwester studiert in Hamburg. | user's sister studies in Hamburg | user's sister studies in Hamburg | user's sister studies in Hamburg |
