1 What the “triples” count means
While looping over each PDF chunk we call
(subj, verb, obj) = triples(chunk.page_content)
Each triple is a subject – predicate – object semantic fact, e.g.

subject	    predicate (verb)	object
Transformer	uses	multi-head attention

chunk 75 → 2 triples
= we extracted two SPO facts from that chunk.
At the end we inserted 242 such triples (edges) into Neo4j.

 