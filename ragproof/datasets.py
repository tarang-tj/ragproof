"""Baked-in demo corpus and labeled query set.

Deterministic and fully offline so the demo/tests never depend on network
access or external data. The corpus spans 8 topic clusters (5 docs each) and
the query set mixes exact-keyword phrasing with genuine paraphrases, so
sparse (lexical) and dense (latent-semantic) retrieval each win on a
different subset of queries -- see README for why this makes hybrid fusion
a legitimate, non-fabricated win rather than a cherry-picked number.
"""

from __future__ import annotations

TypedDoc = dict[str, str]

CORPUS: TypedDoc = {
    "pp1": "Python functions are defined with the def keyword and can return values to the caller.",
    "pp2": "A for loop in python iterates over a list, tuple, or other iterable object one item at a time.",
    "pp3": "Python dictionaries store key value pairs and provide fast constant time lookup by key.",
    "pp4": "Exception handling in python uses try except blocks to catch and recover from runtime errors.",
    "pp5": "List comprehensions in python offer a concise syntax for building a new list from an existing iterable.",
    "cb1": "Pour over coffee brewing uses a slow steady pour of hot water over ground coffee beans in a filter.",
    "cb2": "Espresso extraction forces hot pressurized water through finely ground coffee to produce a concentrated shot.",
    "cb3": "Cold brew coffee steeps coarse coffee grounds in cold water for many hours to reduce bitterness and acidity.",
    "cb4": "The roast level of coffee beans, from light to dark, changes the flavor profile and caffeine content.",
    "cb5": "A burr grinder produces a more consistent coffee grind size than a blade grinder, improving extraction.",
    "se1": "A rocket launch uses staged engines to escape Earth's gravity and reach orbital velocity.",
    "se2": "The International Space Station orbits Earth roughly every ninety minutes carrying a rotating crew of astronauts.",
    "se3": "Mars rovers use solar panels or radioisotope generators to power instruments that analyze soil and rock samples.",
    "se4": "Reusable rocket boosters land vertically after launch, cutting the cost of reaching orbit.",
    "se5": "A satellite in geostationary orbit stays above the same point on Earth, useful for communications and weather.",
    "gd1": "Tomato plants need at least six hours of direct sunlight and consistent watering to produce a good yield.",
    "gd2": "Composting kitchen scraps and yard waste creates nutrient rich soil amendment for garden beds.",
    "gd3": "Pruning fruit trees in late winter encourages healthy new growth and better fruit production in spring.",
    "gd4": "Raised garden beds improve drainage and soil quality compared to planting directly in native clay soil.",
    "gd5": "Companion planting pairs marigolds with vegetables to repel pests without chemical pesticides.",
    "pf1": "A monthly budget tracks income against fixed expenses, variable spending, and savings contributions.",
    "pf2": "An emergency fund covering three to six months of expenses protects against job loss or unexpected bills.",
    "pf3": "Compound interest grows an investment faster the longer money stays invested, rewarding early saving.",
    "pf4": "Paying off high interest credit card debt before investing usually produces a better guaranteed return.",
    "pf5": "Diversifying a portfolio across stocks, bonds, and index funds reduces risk from any single investment.",
    "ce1": "Running at a conversational pace for thirty minutes builds aerobic base fitness over several weeks.",
    "ce2": "Interval training alternates short bursts of high intensity effort with recovery periods to boost endurance.",
    "ce3": "A resting heart rate that trends lower over months is a common sign of improving cardiovascular fitness.",
    "ce4": "Cycling is a low impact cardio workout that builds leg strength while easing stress on the knees.",
    "ce5": "Swimming laps works the whole body and burns significant calories with minimal joint impact.",
    "pc1": "Salting pasta water generously before boiling seasons the noodles from the inside as they cook.",
    "pc2": "Al dente pasta is cooked until firm to the bite, usually one to two minutes less than the package suggests.",
    "pc3": "Reserving a cup of starchy pasta water helps emulsify sauce so it clings to the noodles.",
    "pc4": "A simple tomato sauce starts with garlic and olive oil, then simmers crushed tomatoes until thickened.",
    "pc5": "Fresh pasta made with eggs and flour cooks much faster than dried pasta from a box.",
    "cc1": "Auto scaling adds or removes server instances automatically based on real time traffic load.",
    "cc2": "A load balancer distributes incoming requests across multiple backend servers to avoid overload.",
    "cc3": "Serverless functions run application code on demand without provisioning or managing dedicated servers.",
    "cc4": "Object storage services keep large files like backups and media durable across multiple data centers.",
    "cc5": "A content delivery network caches static assets closer to users to reduce page load latency.",
}

DEMO_QUERIES: list[dict] = [
    {"query": "how do I catch errors in python with try except", "relevant_doc_ids": ["pp4"]},
    {"query": "shorter syntax to build a new list from an iterable in python", "relevant_doc_ids": ["pp5"]},
    {"query": "what data structure gives fast lookup by key in python", "relevant_doc_ids": ["pp3"]},
    {"query": "letting grounds sit in cool liquid overnight instead of using heat", "relevant_doc_ids": ["cb3"]},
    {"query": "why does bean grind uniformity change how well flavor pulls out", "relevant_doc_ids": ["cb5"]},
    {"query": "does darkening the beans longer shift taste and stimulant content", "relevant_doc_ids": ["cb4"]},
    {"query": "landing boosters upright after liftoff to reuse hardware and save money", "relevant_doc_ids": ["se4"]},
    {"query": "why a craft stays fixed over one location as earth turns beneath it", "relevant_doc_ids": ["se5"]},
    {"query": "generating electricity on a distant planet to run science instruments", "relevant_doc_ids": ["se3"]},
    {"query": "how much light exposure and moisture do vine vegetables need to thrive", "relevant_doc_ids": ["gd1"]},
    {"query": "using flowers alongside crops to keep bugs away without spraying chemicals", "relevant_doc_ids": ["gd5"]},
    {"query": "trimming fruit tree branches in the coldest part of the year to boost growth next spring", "relevant_doc_ids": ["gd3"]},
    {"query": "how many months of expenses should a cash cushion or emergency reserve cover before a layoff", "relevant_doc_ids": ["pf2"]},
    {"query": "clearing expensive revolving balances before putting money into the market", "relevant_doc_ids": ["pf4"]},
    {"query": "why does starting to save early let compound interest multiply investment returns over decades", "relevant_doc_ids": ["pf3"]},
    {"query": "a slowing pulse at rest over time signals better heart condition", "relevant_doc_ids": ["ce3"]},
    {"query": "which low strain workout builds legs without hurting the joints", "relevant_doc_ids": ["ce4"]},
    {"query": "alternating hard effort with easy stretches to raise stamina over time", "relevant_doc_ids": ["ce2"]},
    {"query": "keeping the cloudy starchy liquid to help sauce stick to noodles", "relevant_doc_ids": ["pc3"]},
    {
        "query": "cooking time so pasta stays firm to the bite rather than mushy, a bit less than package instructions",
        "relevant_doc_ids": ["pc2"],
    },
    {"query": "does handmade egg pasta cook faster than dried pasta from a box", "relevant_doc_ids": ["pc5"]},
    {"query": "spreading incoming traffic across machines so none gets overwhelmed", "relevant_doc_ids": ["cc2"]},
    {"query": "executing app code on demand with nothing to provision or maintain", "relevant_doc_ids": ["cc3"]},
    {"query": "caching static files near visitors to cut down page wait times", "relevant_doc_ids": ["cc5"]},
]


def load_demo_dataset() -> tuple[dict[str, str], list[dict]]:
    """Return (corpus, labeled_queries) for the baked-in offline demo.

    corpus: doc_id -> document text.
    labeled_queries: list of {"query": str, "relevant_doc_ids": list[str]}.
    """
    return CORPUS, DEMO_QUERIES


def load_beir(name: str) -> tuple[dict[str, str], list[dict]]:
    """Load a real BEIR benchmark dataset (e.g. "scifact", "nfcorpus").

    NOT IMPLEMENTED. This is a stub marking the real seam for production
    benchmarking: BEIR (https://github.com/beir-cellar/beir) ships
    standardized IR datasets with corpus.jsonl, queries.jsonl, and qrels
    files. TODO: add a `beir` dependency, download/cache the requested
    dataset, and adapt its qrels into the same
    {"query": str, "relevant_doc_ids": list[str]} shape used by
    load_demo_dataset() so eval_runner works unchanged against real data.
    """
    raise NotImplementedError(
        f"load_beir({name!r}) is not implemented. This is a stub for wiring "
        "in real BEIR benchmark datasets (see docstring). Use "
        "load_demo_dataset() for the offline demo."
    )
