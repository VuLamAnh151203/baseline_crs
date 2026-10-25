# Additional TopDial development confirmation membership

The immutable resource
`data/manifests/topdial-development-confirmation64.json` freezes 64 additional
development scenarios before inspecting outcome metrics from the new
numerical/reference-pool debugging benchmarks.
It records IDs/order, normalized full scenario content hashes and the processed
development population hash. Membership is ranks 80:144 after sorting
`digest([17,id])`, excluding calibration ranks 0:16 and discovery ranks 16:80.
No outcomes are read during membership freezing and no new inference is launched.

This resource supports later confirmation of a development-selected repair,
using fixed PPDPP, root seed 17, both user protocols and unchanged shared model,
simulator, judge, decoding and refusal contracts. PPDPP must be evaluated on these
same new scenarios; records from the discovery subset cannot be reused across IDs.
Freeze any candidate decision and thresholds before opening confirmation outcomes.
Do not tune on this resource after seeing those outcomes. Report paired uncertainty
and single-seed limitations, including failures to improve.

Status: membership frozen and checksum/count/disjointness verified; candidate
decision, runtime gates and confirmation dialogues are pending. This is additional
development evidence in post-test research, not independent official-test
confirmation. It changes no official test IDs, completed results, training inputs
or global selection lock. The two already frozen discovery benchmarks remain
unchanged.
