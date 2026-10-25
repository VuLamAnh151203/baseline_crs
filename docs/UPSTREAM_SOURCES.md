# Pinned upstream sources

The adapters execute local implementations under `tdrl/src/tdrl/baselines`; they do not import
arbitrary code from neighboring checkouts at runtime. These upstream revisions define the audited
reference implementations or released resources:

| Method/resource | Repository | Revision |
| --- | --- | --- |
| DPDP | <https://github.com/cs-holder/DPDP> | `4d1e22e35db452d51f0f39388456d310ec1c971b` |
| T-EPL | <https://github.com/huyquangdao/EPL> | `364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda` |
| EnPL/ConvPlan | <https://github.com/pandazzh2020/ConvPlan> | `a76d5a7d6ea1d9c7a07a9574f4901ff79ed440bd` |
| ProCoT | <https://github.com/dengyang17/LLM-Proactive> | `77ff4ed7dfda11afd719db3bd40886a1a51edf4e` |
| SCOPE | <https://github.com/chenzhiliang94/convo-plan-SCOPE> | `b35dbc64808afce25e8f8e5d6838ecf271ec4d0b` |
| CARP evaluator reference | <https://github.com/kuduck192/carp_www> | `86f56b07e4efd4e645a51e389423bea9d0d185be` |
| TRIP | <https://github.com/iwangjian/TRIP> | `ee845a5306902ae673e92083711b283b9f33bb9f` |
| DiffTOD | <https://github.com/ninglab/DiffTOD> | `fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1` |

File-level hashes and adaptation contracts live in the corresponding `*-reference.lock.json`
files and `tdrl/configs/sources.lock.json`. Upstream repositories have independent licenses; this
repository links to them instead of vendoring unknown-license source or paper PDFs.
