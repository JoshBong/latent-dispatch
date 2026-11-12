"""Synthetic two-shard QA with an exact crossing dial (Paper A toy_gen2.py, v2 de-leaked generator).

Shard A = people facts (favorite city, birthplace of each mayor), shard B = city facts (mayor, river). The question
chains across them:
    hop0  answer inside one shard
    hop1  A -> B              (person's favorite city -> its mayor)
    hop2  A -> B -> A         (-> that mayor's birthplace)
Every fact segment is shuffled and the target chain sits at a random index, so "take the first fact of that type"
answers nothing. Distractor shards use disjoint name pools.

Dataset name is toy_d<k> for k distractor shards (toy_d0 = the Paper A setting). Placement is over the two
shards: "AB" = people shard on worker A, cities shard on worker B (crossings = hop),
"BA" = reversed, "AA"/"BB" = both on one worker (crossings = 0; BB is the real control, see data.py). With
n_distract = 0 the item sequence is identical to the 2026-09-30 v2 run.
"""
import random

from .data import place

PEOPLE = ["Ines Farrow", "Tomas Reddy", "Priya Castellan", "Owen Mbeki", "Lena Okonkwo", "Marco Duval", "Sana Ilori",
          "Felix Brandt", "Nora Halvorsen", "Dev Anand", "Yuki Sato", "Ada Lindqvist"]
CITIES = ["Velmora", "Kastrel", "Orinth", "Bellhaven", "Sundmark", "Quillford", "Tarnova", "Ashgrove", "Lorwick",
          "Peldane", "Marisol", "Ny Havn"]
MAYORS = ["Bram Holt", "Celia Vance", "Dario Okafor", "Elin Ruud", "Farah Nasser", "Gideon Pike", "Hana Kimura",
          "Ivo Petrov", "Jora Lind", "Kip Alvarez", "Lior Adler", "Mira Solano"]
RIVERS = ["the Ossory", "the Teviot", "the Marne", "the Kallis", "the Brenn", "the Ulla", "the Sorne", "the Duven",
          "the Pell", "the Arno", "the Vasse", "the Nyx"]
X_PEOPLE = ["Aria Venn", "Bo Castile", "Cyra Hale", "Dmitri Osei", "Esme Varga", "Florin Iyer", "Greta Moss",
            "Hugo Tanaka", "Isla Brenner", "Jonas Quade", "Kaia Fenn", "Luca Moreau"]
X_CITIES = ["Dunmere", "Halcrest", "Irvale", "Jessamy", "Korrin", "Luthany", "Mordale", "Nessby", "Oakhurst", "Pryne",
            "Rellow", "Saltwick"]
X_MAYORS = ["Anya Stroud", "Basil Crane", "Corin Albescu", "Dana Whitlow", "Ezra Kolb", "Fern Ashby", "Gil Navarro",
            "Hedda Lund", "Ian Mercado", "Juno Park", "Kofi Ansah", "Lena Brook"]
X_RIVERS = ["the Calder", "the Wensum", "the Tamar", "the Ribble", "the Kennet", "the Fowey", "the Lune", "the Dove",
            "the Esk", "the Tees", "the Wear", "the Aire"]


def _seg(rng, pairs, fmt):
    pairs = list(pairs)
    rng.shuffle(pairs)
    return " ".join(fmt(a, b) for a, b in pairs)


def _people_doc(rng, people, fav, mayors, home):
    return ("People: " + _seg(rng, zip(people, fav), lambda p, c: f"{p}'s favorite city is {c}.") + " "
            + _seg(rng, zip(mayors, home), lambda m, h: f"{m} was born in {h}."))


def _cities_doc(rng, fav, mayors, home, rivers):
    return ("Cities: " + _seg(rng, zip(fav, mayors), lambda c, m: f"The mayor of {c} is {m}.") + " "
            + _seg(rng, zip(home, rivers), lambda h, r: f"The river {r} flows through {h}."))


def _distractor(rng, kind):
    people, cities = rng.sample(X_PEOPLE, 3), rng.sample(X_CITIES, 6)
    mayors, rivers = rng.sample(X_MAYORS, 3), rng.sample(X_RIVERS, 3)
    if kind == 0:
        return _people_doc(rng, people, cities[:3], mayors, cities[3:])
    return _cities_doc(rng, cities[:3], mayors, cities[3:], rivers)


def items(hop: int, n: int, seed: int = 0, n_distract: int = 0) -> list[dict]:
    """Raw items: question, gold, the two shards, distractor shards, chain. Same rng sequence as toy_gen2."""
    if hop not in (0, 1, 2):
        raise ValueError(hop)
    rng = random.Random(seed * 100 + hop)
    out = []
    for i in range(n):
        people, cities = rng.sample(PEOPLE, 3), rng.sample(CITIES, 6)
        mayors, rivers = rng.sample(MAYORS, 3), rng.sample(RIVERS, 3)
        fav, home = cities[:3], cities[3:6]
        a = _people_doc(rng, people, fav, mayors, home)
        b = _cities_doc(rng, fav, mayors, home, rivers)
        t = rng.randrange(3)
        p, c, m, h, r = people[t], fav[t], mayors[t], home[t], rivers[t]
        if hop == 0:
            if i % 2 == 0:
                q, gold, answer_shard = f"What is {p}'s favorite city?", c, 0
            else:
                q, gold, answer_shard = f"Who is the mayor of {c}?", m, 1
        elif hop == 1:
            q, gold, answer_shard = f"Who is the mayor of {p}'s favorite city?", m, None
        else:
            q, gold, answer_shard = f"In which city was the mayor of {p}'s favorite city born?", h, None
        distract = [_distractor(rng, k % 2) for k in range(n_distract)]
        out.append(dict(i=i, question=q, gold=gold, shards=[a, b], distract=distract, answer_shard=answer_shard,
                        chain=[p, c, m, h, r][:hop + 2]))
    return out


def questions(hop: int, n: int, seed: int, patterns: list[str], n_distract: int = 0, n_workers: int = 2) -> list[dict]:
    out = []
    for it in items(hop, n, seed, n_distract):
        source_id = f"toy-d{n_distract}-h{hop}-s{seed}-i{it['i']}"
        for pat in patterns:
            if len(pat) != 2:
                raise ValueError("toy patterns place the two shards: AB, BA, AA or BB")
            rng = random.Random(f"{seed}:{source_id}:{n_distract}")    # per question, not per pattern (data.place)
            workers, gold = place(it["shards"], it["distract"], pat, n_workers, rng)
            if it["answer_shard"] is not None:          # hop0: only the shard holding the answer is gold
                gold = [g for g in gold if g["hop"] == it["answer_shard"]]
            out.append(dict(
                id=f"{source_id}:{pat}", dataset=f"toy_d{n_distract}", source_id=source_id, question=it["question"],
                answers=[it["gold"]], qtype=f"hop{hop}", hops=len(gold), placement=pat,
                crossings=0 if (hop == 0 or pat[0] == pat[1]) else hop, seed=seed, workers=workers, gold=gold,
                bridges=it["chain"][1:-1], decomposition=[]))
    return out
