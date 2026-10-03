"""Builds mock Yelp-style (JSONL) and Amazon-style (CSV) files with known dirty rows."""
import csv, json, random
rng = random.Random(7)

POS = ["The food was really good and the portions were generous.", "Everything arrived quickly and it works exactly as described.",
       "I would happily recommend this to anyone looking for a solid option.", "Great value for the price and the quality is excellent.",
       "It was clean, fast and the whole experience felt effortless."]
NEG = ["It stopped working after a week and I am very disappointed.", "The order was wrong and nobody seemed to care about fixing it.",
       "Way too expensive for what you get and the quality was poor.", "I waited far too long and the food arrived cold.",
       "The product feels cheap and broke the first time I used it."]
MIX = ["It is okay but nothing special and I expected a bit more.", "Some parts were good while others were honestly quite disappointing.",
       "The price is fair, although the quality could definitely be better.", "It does the job, but I am not sure I would buy it again.",
       "Decent overall, with a few things that really let it down."]
FILL = ["We came here after work and it was a pretty normal visit.", "I have been using this for a couple of weeks now.",
        "Overall the experience was about what I expected from the description.", "There were a few small things that stood out to me.",
        "I compared it with a couple of similar options before deciding."]

def clean(stars):
    pool = NEG if stars <= 2 else (MIX if stars == 3 else POS)
    return " ".join(rng.sample(pool, 3) + rng.sample(FILL, 2))

DIRTY = {
    "name":   lambda s: clean(s) + " Our server Maria was very kind to us.",
    "name2":  lambda s: "Priya helped me at the counter and it went fine. " + clean(s),
    "gender": lambda s: clean(s) + " My husband agreed with me about all of this.",
    "ethnic": lambda s: clean(s) + " It felt like a proper Indian place honestly.",
    "loc":    lambda s: clean(s) + " Best one I have found in Philadelphia so far.",
    "url":    lambda s: clean(s) + " More photos at www.example.com today.",
    "short":  lambda s: "Not good at all.",
    "long":   lambda s: " ".join(clean(s) for _ in range(12)),
    "nonen":  lambda s: "La comida estuvo muy buena y el servicio fue excelente, volveremos pronto sin duda alguna gracias por todo " * 2,
}
CLEAN_START = ["Love this place, would come back.", "Just ordered again and it was good.", "Price is fair for what you get.", "Will visit again soon."]

def make_rows(n, prefix):
    rows, dup_pool = [], []
    for i in range(n):
        stars = rng.choice([1, 2, 3, 3, 4, 5])
        r = rng.random()
        if r < 0.45:
            text = clean(stars)
            if rng.random() < 0.15: text = rng.choice(CLEAN_START) + " " + text  # allowlisted starters must survive
            kind = "clean"
        elif r < 0.55 and dup_pool:
            text, stars = rng.choice(dup_pool); kind = "dup"
        else:
            kind = rng.choice(list(DIRTY)); text = DIRTY[kind](stars)
        if kind == "clean": dup_pool.append((text, stars))
        rows.append({"id": f"{prefix}{i}", "stars": stars, "text": text, "kind": kind})
    return rows

yelp = make_rows(9000, "y")
with open("mock_yelp.json", "w") as f:
    for r in yelp:
        f.write(json.dumps({"review_id": r["id"], "user_id": "u", "business_id": "b", "stars": r["stars"],
                            "useful": 0, "funny": 0, "cool": 0, "text": r["text"], "date": "2019-01-01", "_kind": r["kind"]}) + "\n")
amz = make_rows(7000, "a")
with open("mock_amazon.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["rating", "title", "text", "helpful_vote", "_kind"]); w.writeheader()
    for r in amz:
        w.writerow({"rating": float(r["stars"]), "title": "t", "text": r["text"], "helpful_vote": 0, "_kind": r["kind"]})
print("mock files written")
