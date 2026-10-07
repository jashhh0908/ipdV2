| pair | side | utterance | current B fact | candidate B fact | candidate label | retention |
|---|---|---|---|---|---|---|
| e1dc | a | Rohan ek botal doodh laaya | Rohan bought a carton of milk | Rohan gave one carton of milk. | translated_to_english | 0.2 |
| e1dc | b | Rohan ek gilaas doodh laaya | (none) | Rohan gave some strange milk. | translated_to_english | 0.2 |
| ce85 | a | Dadi ji ne pooja karwayi | (none) | Dadi ji ne pooja karwayi | source_language | 1.0 |
| ce85 | b | Dadi ne pooja karwayi | (none) | Dadi ne pooja karwayi | source_language | 1.0 |
| 4b3d | a | kal Suresh Pune gaya tha | (none) | kal Suresh Pune gaya tha | source_language | 1.0 |
| 4b3d | b | kal Suresh Pune jayega | (none) | kal Suresh Pune jayega | source_language | 1.0 |
| aef7 | a | Vikram ek gilaas paani piya | (none) | Vikram ek gilaas paani piya. | source_language | 1.0 |
| aef7 | b | Vikram ek botal paani piya | (none) | Vikram paī botal paani piya. | source_language | 0.8 |
| d19d | a | Neha party mein aayi thi | Neha was at a party | Neha party mein aayi thi | source_language | 1.0 |
| d19d | b | नेहा party mein aayi thi | Neha was at a party | नेहा party mein aayi thi | source_language | 1.0 |
| 3886 | a | Neha office se aa rahi hai | Neha is at the office | Neha office se aa rahi hai | source_language | 1.0 |
| 3886 | b | Neha college se aa rahi hai | Neha is at college | Neha college se aa rahi hai | source_language | 1.0 |
| f85a | a | Vikram ne do jode kapde diye | (none) | Vikram ne do jode kapde diye. | source_language | 1.0 |
| f85a | b | Vikram ne do tukde kapda diya | (none) | Vikram ne do tukde kapda diya. | source_language | 1.0 |
| 679e | a | Manoj ek botal doodh laaya | Manoj bought milk | Manoj dadaa ek botal doodh laaya. | source_language | 1.0 |
| 679e | b | Manoj ek gilaas doodh laaya | (none) | Manoj gave some strange milk. | translated_to_english | 0.2 |
| 43a3 | a | kal Manoj Pune gaya tha | (none) | kal Manoj Pune gaya tha | source_language | 1.0 |
| 43a3 | b | kal Manoj Pune jayega | (none) | Kal Manoj Pune gareebega. | source_language | 0.75 |
| 4a2c | a | kehte hain Arjun bimar hai | Arjun is sick | kehte hain Arjun bimar hai | source_language | 1.0 |
| 4a2c | b | maine khud dekha Arjun bimar hai | (none) | maine khud dekha Arjun bimar hai | source_language | 1.0 |
| 0b25 | a | Markus, wie war dein Urlaub in Italien? | (none) | Markus, wie war dein Urlaub in Italien? | source_language | 1.0 |
| 0b25 | b | Markus, wie war Ihr Urlaub in Italien? | (none) | Markus, wie war Ihr Urlaub in Italien? | source_language | 1.0 |
| 91d5 | a | yeh laptop mera hai | (none) | yeh laptop mera hai | source_language | 1.0 |
| 91d5 | b | woh laptop mera hai | Laptop is mine | laptop mera hai | source_language | 0.6667 |
| 181e | a | Suresh ne do jode kapde diye | (none) | Suresh ne do jode kapde diye. | source_language | 1.0 |
| 181e | b | Suresh ne do tukde kapda diya | (none) | Suresh gave two clothes to him. | translated_to_english | 0.1667 |
| c64d | a | Verma uncle kal ghar aaye the | Verma uncle is coming to the house | Verma uncle kal ghar aaye the | source_language | 1.0 |
| c64d | b | Verma bhaiya kal ghar aaye the | (none) | Verma bhaiya kal ghar aaye the | source_language | 1.0 |
| 8e23 | a | Anjali aa rahi hai | (none) | Anjali is coming. | mixed | 0.25 |
| 8e23 | b | अंजलि aa rahi hai | अंजलि is awake | अंजलि aa rahi hai | source_language | 1.0 |
| d82e | a | Nani ji ne achaar bheja | (none) | Nani ji ne achaar bheja | source_language | 1.0 |
| d82e | b | Nani ne achaar bheja | (none) | Nani ne achaar bheja | source_language | 1.0 |
| 2535 | a | Neha school se aa rahi hai | Neha is at school | Neha school se aa rahi hai | source_language | 1.0 |
| 2535 | b | Neha gym se aa rahi hai | Neha is at the gym | Neha gym se aa rahi hai | source_language | 1.0 |
| b7ac | a | Neha aa rahi hai | (none) | Neha está llegando. | mixed | 0.25 |
| b7ac | b | नेहा aa rahi hai | Neha is coming | नेहा aa rahi hai | source_language | 1.0 |
| d1db | a | meri mami Jaipur aa rahi hai | Mami is in Jaipur | meri mami Jaipur aa rahi hai | source_language | 1.0 |
| d1db | b | meri mausi Jaipur aa rahi hai | (none) | meri mausi Jaipur aa rahi hai | source_language | 1.0 |
| 61e5 | a | Ali geldi | (none) | Ali geldi | source_language | 1.0 |
| 61e5 | b | Ali gelmiş | (none) | Ali gelmiş | source_language | 1.0 |

| pair | candidate entry A | candidate entry B | judge | gate (lexicon) |
|---|---|---|---|---|
| e1dc | Rohan gave one carton of milk. | Rohan gave some strange milk. | keep_both | keep_both |
| ce85 | Dadi ji ne pooja karwayi | Dadi ne pooja karwayi | merge | merge |
| 4b3d | kal Suresh Pune gaya tha | kal Suresh Pune jayega | merge | merge |
| aef7 | Vikram ek gilaas paani piya. | Vikram paī botal paani piya. | keep_both | keep_both |
| d19d | Neha party mein aayi thi | नेहा party mein aayi thi | merge | merge |
| 3886 | Neha office se aa rahi hai | Neha college se aa rahi hai | keep_both | keep_both |
| f85a | Vikram ne do jode kapde diye. | Vikram ne do tukde kapda diya. | merge | merge |
| 679e | Manoj dadaa ek botal doodh laaya. | Manoj gave some strange milk. | keep_both | keep_both |
| 43a3 | kal Manoj Pune gaya tha | Kal Manoj Pune gareebega. | keep_both | keep_both |
| 4a2c | kehte hain Arjun bimar hai | maine khud dekha Arjun bimar hai | merge | merge |
| 0b25 | Markus, wie war dein Urlaub in Italien? | Markus, wie war Ihr Urlaub in Italien? | merge | merge |
| 91d5 | yeh laptop mera hai | laptop mera hai | merge | merge |
| 181e | Suresh ne do jode kapde diye. | Suresh gave two clothes to him. | keep_both | keep_both |
| c64d | Verma uncle kal ghar aaye the | Verma bhaiya kal ghar aaye the | merge | merge |
| 8e23 | Anjali is coming. | अंजलि aa rahi hai | merge | merge |
| d82e | Nani ji ne achaar bheja | Nani ne achaar bheja | merge | merge |
| 2535 | Neha school se aa rahi hai | Neha gym se aa rahi hai | keep_both | keep_both |
| b7ac | Neha está llegando. | नेहा aa rahi hai | keep_both | keep_both |
| d1db | meri mami Jaipur aa rahi hai | meri mausi Jaipur aa rahi hai | keep_both | keep_both |
| 61e5 | Ali geldi | Ali gelmiş | merge | merge |
