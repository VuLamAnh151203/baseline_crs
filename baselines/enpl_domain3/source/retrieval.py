"""Domain-adapted EnPL retrieval with exact entity-set cardinality overlap."""

from .normalize import extract_domain_entities
from .schema import RetrievedPlan


class DomainEntityOverlapRetriever:
    def __init__(self, entries, k=3, distinct_source_conversations=True):
        if k != 3:
            raise ValueError("the selected EnPL paper setting requires retrieval k=3")
        self.entries = tuple(entries)
        self.k = k
        self.distinct = distinct_source_conversations

    def select(self, profile, target_items, excluded_groups=()):
        domain, query_entities = extract_domain_entities(profile, target_items)
        query = set(query_entities)
        excluded = set(excluded_groups)
        ranked = []
        for entry in self.entries:
            if (
                entry.source_split != "train"
                or entry.duplicate_group_id in excluded
                or entry.retrieval_domain != domain
            ):
                continue
            overlap = len(query & set(entry.retrieval_entities))
            ranked.append(
                (
                    -overlap,
                    -entry.consistency_score,
                    entry.source_dataset,
                    entry.source_conversation_id,
                    entry.candidate_index,
                    entry,
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        selected = []
        used_groups = set()
        for item in ranked:
            entry = item[-1]
            if self.distinct and entry.duplicate_group_id in used_groups:
                continue
            overlap = -item[0]
            selected.append(RetrievedPlan(entry, overlap, overlap == 0))
            used_groups.add(entry.duplicate_group_id)
            if len(selected) == self.k:
                break
        if not selected:
            raise ValueError(f"no eligible EnPL training demonstrations for domain {domain}")
        if len(selected) != self.k:
            raise ValueError(f"fewer than {self.k} distinct EnPL demonstrations for {domain}")
        return tuple(selected)
