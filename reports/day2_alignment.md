# Day 2 — Char-span -> BIO Alignment Verification

Each example is aligned to BIO tokens and round-tripped back to char spans. PASS requires every gold span to be recovered by a same-type span on overlap, with no spurious spans.


**Overall: ALL PASS ✓**


### Example 1 — PASS
> Patient John Reyes, DOB 03/14/1981, MRN A55213, seen by Dr. Smith.

| token | char range | label |
|---|---|---|
| `[CLS]` | 0-0 `` | -100 (special) |
| `▁Patient` | 0-7 `Patient` | O |
| `▁John` | 7-12 ` John` | B-NAME |
| `▁Reyes` | 12-18 ` Reyes` | I-NAME |
| `,` | 18-19 `,` | O |
| `▁DOB` | 19-23 ` DOB` | O |
| `▁03` | 23-26 ` 03` | B-DATE |
| `/` | 26-27 `/` | I-DATE |
| `14` | 27-29 `14` | I-DATE |
| `/` | 29-30 `/` | I-DATE |
| `1981` | 30-34 `1981` | I-DATE |
| `,` | 34-35 `,` | O |
| `▁M` | 35-37 ` M` | O |
| `RN` | 37-39 `RN` | O |
| `▁A` | 39-41 ` A` | B-MRN |
| `552` | 41-44 `552` | I-MRN |
| `13` | 44-46 `13` | I-MRN |
| `,` | 46-47 `,` | O |
| `▁seen` | 47-52 ` seen` | O |
| `▁by` | 52-55 ` by` | O |
| `▁Dr` | 55-58 ` Dr` | O |
| `.` | 58-59 `.` | O |
| `▁Smith` | 59-65 ` Smith` | O |
| `.` | 65-66 `.` | O |
| `[SEP]` | 0-0 `` | -100 (special) |

- gold `John Reyes` (NAME) -> recovered: [('NAME', ' John Reyes')]
- gold `03/14/1981` (DATE) -> recovered: [('DATE', ' 03/14/1981')]
- gold `A55213` (MRN) -> recovered: [('MRN', ' A55213')]

### Example 2 — PASS
> Member SSN 402-11-9837; plan ID UHC9921047733 on the policy.

| token | char range | label |
|---|---|---|
| `[CLS]` | 0-0 `` | -100 (special) |
| `▁Member` | 0-6 `Member` | O |
| `▁SSN` | 6-10 ` SSN` | O |
| `▁402` | 10-14 ` 402` | B-SSN |
| `-` | 14-15 `-` | I-SSN |
| `11` | 15-17 `11` | I-SSN |
| `-` | 17-18 `-` | I-SSN |
| `98` | 18-20 `98` | I-SSN |
| `37` | 20-22 `37` | I-SSN |
| `;` | 22-23 `;` | O |
| `▁plan` | 23-28 ` plan` | O |
| `▁ID` | 28-31 ` ID` | O |
| `▁U` | 31-33 ` U` | B-PLAN_ID |
| `HC` | 33-35 `HC` | I-PLAN_ID |
| `99` | 35-37 `99` | I-PLAN_ID |
| `210` | 37-40 `210` | I-PLAN_ID |
| `477` | 40-43 `477` | I-PLAN_ID |
| `33` | 43-45 `33` | I-PLAN_ID |
| `▁on` | 45-48 ` on` | O |
| `▁the` | 48-52 ` the` | O |
| `▁policy` | 52-59 ` policy` | O |
| `.` | 59-60 `.` | O |
| `[SEP]` | 0-0 `` | -100 (special) |

- gold `402-11-9837` (SSN) -> recovered: [('SSN', ' 402-11-9837')]
- gold `UHC9921047733` (PLAN_ID) -> recovered: [('PLAN_ID', ' UHC9921047733')]

### Example 3 — PASS
> Call the patient at (216) 555-0148 or email jane.doe@gmail.com.

| token | char range | label |
|---|---|---|
| `[CLS]` | 0-0 `` | -100 (special) |
| `▁Call` | 0-4 `Call` | O |
| `▁the` | 4-8 ` the` | O |
| `▁patient` | 8-16 ` patient` | O |
| `▁at` | 16-19 ` at` | O |
| `▁(` | 19-21 ` (` | B-PHONE |
| `216` | 21-24 `216` | I-PHONE |
| `)` | 24-25 `)` | I-PHONE |
| `▁555` | 25-29 ` 555` | I-PHONE |
| `-` | 29-30 `-` | I-PHONE |
| `01` | 30-32 `01` | I-PHONE |
| `48` | 32-34 `48` | I-PHONE |
| `▁or` | 34-37 ` or` | O |
| `▁email` | 37-43 ` email` | O |
| `▁jane` | 43-48 ` jane` | B-EMAIL |
| `.` | 48-49 `.` | I-EMAIL |
| `do` | 49-51 `do` | I-EMAIL |
| `e` | 51-52 `e` | I-EMAIL |
| `@` | 52-53 `@` | I-EMAIL |
| `gmail` | 53-58 `gmail` | I-EMAIL |
| `.` | 58-59 `.` | I-EMAIL |
| `com` | 59-62 `com` | I-EMAIL |
| `.` | 62-63 `.` | O |
| `[SEP]` | 0-0 `` | -100 (special) |

- gold `(216) 555-0148` (PHONE) -> recovered: [('PHONE', ' (216) 555-0148')]
- gold `jane.doe@gmail.co` (EMAIL) -> recovered: [('EMAIL', ' jane.doe@gmail.com')]

### Example 4 — PASS
> Home address on file: 728 Oak Street, Akron, OH 44312.

| token | char range | label |
|---|---|---|
| `[CLS]` | 0-0 `` | -100 (special) |
| `▁Home` | 0-4 `Home` | O |
| `▁address` | 4-12 ` address` | O |
| `▁on` | 12-15 ` on` | O |
| `▁file` | 15-20 ` file` | O |
| `:` | 20-21 `:` | O |
| `▁728` | 21-25 ` 728` | B-ADDRESS |
| `▁Oak` | 25-29 ` Oak` | I-ADDRESS |
| `▁Street` | 29-36 ` Street` | I-ADDRESS |
| `,` | 36-37 `,` | I-ADDRESS |
| `▁Akron` | 37-43 ` Akron` | I-ADDRESS |
| `,` | 43-44 `,` | I-ADDRESS |
| `▁OH` | 44-47 ` OH` | I-ADDRESS |
| `▁443` | 47-51 ` 443` | I-ADDRESS |
| `12` | 51-53 `12` | I-ADDRESS |
| `.` | 53-54 `.` | O |
| `[SEP]` | 0-0 `` | -100 (special) |

- gold `728 Oak Street, Akron, OH 44312` (ADDRESS) -> recovered: [('ADDRESS', ' 728 Oak Street, Akron, OH 44312')]

### Example 5 — PASS
> Patient is 92 years old; session from 73.118.42.9 logged.

| token | char range | label |
|---|---|---|
| `[CLS]` | 0-0 `` | -100 (special) |
| `▁Patient` | 0-7 `Patient` | O |
| `▁is` | 7-10 ` is` | O |
| `▁92` | 10-13 ` 92` | B-AGE90 |
| `▁years` | 13-19 ` years` | O |
| `▁old` | 19-23 ` old` | O |
| `;` | 23-24 `;` | O |
| `▁session` | 24-32 ` session` | O |
| `▁from` | 32-37 ` from` | O |
| `▁73` | 37-40 ` 73` | B-IP |
| `.` | 40-41 `.` | I-IP |
| `118` | 41-44 `118` | I-IP |
| `.` | 44-45 `.` | I-IP |
| `42` | 45-47 `42` | I-IP |
| `.` | 47-48 `.` | I-IP |
| `9` | 48-49 `9` | I-IP |
| `▁logged` | 49-56 ` logged` | O |
| `.` | 56-57 `.` | O |
| `[SEP]` | 0-0 `` | -100 (special) |

- gold `92` (AGE90) -> recovered: [('AGE90', ' 92')]
- gold `73.118.42.9` (IP) -> recovered: [('IP', ' 73.118.42.9')]
