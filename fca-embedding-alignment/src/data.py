import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from sklearn.model_selection import train_test_split

SEED = 42
QUALITY_THRESHOLD = 0.3

GENDER_CANDIDATES = [
    ("he", "she"), ("him", "her"), ("his", "hers"), ("himself", "herself"),
    ("man", "woman"), ("boy", "girl"), ("male", "female"), ("guy", "gal"),
    ("lad", "lass"), ("gentleman", "lady"), ("father", "mother"), ("son", "daughter"),
    ("brother", "sister"), ("uncle", "aunt"), ("nephew", "niece"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"),
    ("grandpa", "grandma"), ("dad", "mom"), ("husband", "wife"),
    ("boyfriend", "girlfriend"), ("groom", "bride"), ("widower", "widow"),
    ("godfather", "godmother"), ("stepfather", "stepmother"),
    ("stepson", "stepdaughter"), ("fiance", "fiancee"), ("patriarch", "matriarch"),
    ("papa", "mama"), ("daddy", "mummy"), ("beau", "belle"),
    ("mr", "mrs"), ("sir", "madam"), ("lord", "lady"), ("king", "queen"),
    ("prince", "princess"), ("duke", "duchess"), ("baron", "baroness"),
    ("count", "countess"), ("emperor", "empress"), ("actor", "actress"),
    ("waiter", "waitress"), ("host", "hostess"), ("steward", "stewardess"),
    ("hero", "heroine"), ("monk", "nun"), ("priest", "priestess"),
    ("businessman", "businesswoman"), ("spokesman", "spokeswoman"),
    ("chairman", "chairwoman"), ("congressman", "congresswoman"),
    ("councilman", "councilwoman"), ("policeman", "policewoman"),
    ("salesman", "saleswoman"), ("headmaster", "headmistress"),
    ("landlord", "landlady"), ("schoolboy", "schoolgirl"),
    ("postmaster", "postmistress"), ("shepherd", "shepherdess"),
    ("masseur", "masseuse"), ("heir", "heiress"), ("god", "goddess"),
    ("wizard", "witch"), ("sorcerer", "sorceress"), ("abbot", "abbess"),
    ("bachelor", "spinster"), ("hunter", "huntress"),
    ("fatherhood", "motherhood"), ("brotherhood", "sisterhood"),
    ("fraternity", "sorority"), ("masculinity", "femininity"),
    ("paternity", "maternity"), ("manhood", "womanhood"),
    ("boyhood", "girlhood"), ("paternal", "maternal"),
    ("testosterone", "estrogen"), ("monastery", "convent"),
    ("stallion", "mare"), ("colt", "filly"), ("rooster", "hen"),
    ("bull", "cow"), ("lion", "lioness"), ("tiger", "tigress"),
    ("boar", "sow"), ("baritone", "soprano"),
    ("men", "women"), ("boys", "girls"), ("fathers", "mothers"),
    ("sons", "daughters"), ("brothers", "sisters"), ("uncles", "aunts"),
    ("nephews", "nieces"), ("husbands", "wives"), ("grandsons", "granddaughters"),
    ("kings", "queens"), ("princes", "princesses"), ("actors", "actresses"),
    ("monks", "nuns"), ("heroes", "heroines"), ("gods", "goddesses"),
]

ROYALTY_CANDIDATES = [
    ("emperor", "peasant"), ("king", "commoner"), ("prince", "pauper"),
    ("monarch", "subject"), ("sovereign", "subject"), ("pharaoh", "slave"),
    ("tsar", "serf"), ("sultan", "peasant"), ("regent", "commoner"),
    ("throne", "hovel"), ("lord", "serf"), ("noble", "villager"),
    ("duke", "farmer"), ("baron", "laborer"), ("earl", "worker"),
    ("aristocrat", "commoner"), ("nobleman", "peasant"), ("knight", "squire"),
    ("elite", "plebeian"), ("patrician", "plebeian"), ("overlord", "vassal"),
    ("master", "servant"), ("ruler", "subject"), ("tyrant", "slave"),
    ("general", "private"), ("admiral", "sailor"), ("commander", "soldier"),
    ("officer", "recruit"), ("colonel", "corporal"), ("captain", "sailor"),
    ("marshal", "trooper"), ("lieutenant", "private"), ("warlord", "footman"),
    ("executive", "clerk"), ("director", "assistant"), ("manager", "intern"),
    ("president", "employee"), ("chairman", "secretary"), ("boss", "subordinate"),
    ("employer", "employee"), ("supervisor", "trainee"), ("founder", "worker"),
    ("magnate", "clerk"), ("mogul", "laborer"), ("tycoon", "worker"),
    ("pope", "monk"), ("bishop", "priest"), ("cardinal", "friar"),
    ("archbishop", "deacon"), ("abbot", "novice"), ("professor", "student"),
    ("dean", "freshman"), ("scholar", "pupil"), ("chancellor", "undergraduate"),
    ("headmaster", "pupil"), ("leader", "follower"), ("conqueror", "conquered"),
    ("victor", "defeated"), ("champion", "loser"), ("chief", "subordinate"),
    ("governor", "citizen"), ("dignitary", "commoner"), ("celebrity", "nobody"),
    ("hero", "peasant"), ("patriarch", "servant"), ("millionaire", "beggar"),
    ("tycoon", "pauper"), ("mogul", "vagrant"), ("aristocrat", "peasant"),
    ("bourgeois", "proletarian"), ("palace", "shack"), ("mansion", "cottage"),
    ("castle", "hut"), ("estate", "slum"), ("cathedral", "chapel"),
    ("powerful", "powerless"), ("dominant", "submissive"), ("strong", "weak"),
    ("superior", "inferior"), ("mighty", "feeble"), ("commanding", "obedient"),
    ("authoritative", "subservient"), ("prestigious", "humble"),
    ("eminent", "obscure"), ("distinguished", "ordinary"), ("exalted", "lowly"),
    ("majestic", "modest"), ("imperial", "common"), ("regal", "plain"),
    ("noble", "ignoble"), ("grand", "meager"), ("lofty", "lowly"),
    ("illustrious", "unknown"), ("prominent",
                                 "marginal"), ("renowned", "anonymous"),
    ("influential", "insignificant"), ("elevated", "degraded"),
    ("glorious", "wretched"), ("supreme", "negligible"), ("preeminent", "mediocre"),
    ("rich", "poor"), ("wealthy", "impoverished"), ("affluent", "destitute"),
    ("privileged", "underprivileged"), ("prosperous",
                                        "needy"), ("opulent", "indigent"),
]


class AnchorPairDataset(Dataset):
    def __init__(self, pairs, embeddings, target_axis):
        self.vec_a = [torch.tensor(
            embeddings[w1], dtype=torch.float32) for w1, _ in pairs]
        self.vec_b = [torch.tensor(
            embeddings[w2], dtype=torch.float32) for _, w2 in pairs]
        self.target = torch.tensor(target_axis, dtype=torch.float32)

    def __len__(self):
        return len(self.vec_a)

    def __getitem__(self, idx):
        return self.vec_a[idx], self.vec_b[idx], self.target


def filter_pairs(candidates, embeddings, pair_name=""):
    valid, missing = [], []
    for w1, w2 in candidates:
        w1_in = w1 in embeddings.key_to_index
        w2_in = w2 in embeddings.key_to_index
        if w1_in and w2_in:
            valid.append((w1, w2))
        else:
            missing.append((w1, w2))
    print(f"\n{'-'*50}\n  {pair_name}\n{'-'*50}")
    print(
        f"  candidates: {len(candidates)} | valid ones: {len(valid)} | discarded: {len(missing)}")
    return valid


def analyze_pair_quality(pairs, embeddings, axis_name="", quality_threshold=QUALITY_THRESHOLD):
    diffs = []
    for w1, w2 in pairs:
        diff = embeddings[w1] - embeddings[w2]
        norm = np.linalg.norm(diff)
        diffs.append(diff / norm if norm > 1e-10 else diff)
    diffs = np.array(diffs)
    mean_dir = diffs.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
    cos_scores = diffs @ mean_dir
    good = [(p, s)
            for p, s in zip(pairs, cos_scores) if s >= quality_threshold]
    print(f"\n  analysis: {axis_name}")
    print(f"  cos mean={cos_scores.mean():.4f} std={cos_scores.std():.4f} | threshold {quality_threshold}: {len(good)}/{len(pairs)}")
    return [p for p, _ in good], mean_dir, list(zip(pairs, cos_scores))


def load_data(embeddings, test_size=0.2):

    gender_pairs = filter_pairs(GENDER_CANDIDATES, embeddings, "GENDER PAIRS")
    royalty_pairs = filter_pairs(
        ROYALTY_CANDIDATES, embeddings, "ROYALTY PAIRS")

    gender_clean, gender_dir, _ = analyze_pair_quality(
        gender_pairs, embeddings, "GENDER")
    royalty_clean, royalty_dir, _ = analyze_pair_quality(
        royalty_pairs, embeddings, "STATUS")

    cos_axes = np.dot(gender_dir, royalty_dir)
    print(f"\n  Cosine between axes: {cos_axes:.4f}")

    g_train, g_val = train_test_split(
        gender_clean, test_size=test_size, random_state=SEED)
    r_train, r_val = train_test_split(
        royalty_clean, test_size=test_size, random_state=SEED)

    train_ds = ConcatDataset([
        AnchorPairDataset(g_train, embeddings, [1, 0, 0]),
        AnchorPairDataset(r_train, embeddings, [0, 1, 0]),
    ])
    val_ds = ConcatDataset([
        AnchorPairDataset(g_val, embeddings, [1, 0, 0]),
        AnchorPairDataset(r_val, embeddings, [0, 1, 0]),
    ])
    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32,
                            shuffle=False, num_workers=0)
    print(f"\n  Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader, gender_clean, royalty_clean
