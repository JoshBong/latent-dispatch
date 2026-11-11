from latent_dispatch.score import contains_answer, extract_answer, normalize, score


def test_extract_prefers_boxed_after_thinking():
    assert extract_answer("<think>maybe \\boxed{Wrong}</think>\nSo \\boxed{Bram Holt}.") == "Bram Holt"
    assert extract_answer("reasoning about \\boxed{no}\n</think>\n\n\\boxed{Velmora}") == "Velmora"   # prefilled opener


def test_extract_fallbacks():
    assert extract_answer("The answer is Kastrel.") == "Kastrel"
    assert extract_answer("Velmora. It is the city.") == "Velmora"
    assert extract_answer("The answer is St. Louis.") == "St. Louis"
    assert extract_answer("The answer is the U.S. Army. It won.") == "the U.S. Army"
    assert extract_answer("") == ""


def test_number_words_match_digits():
    assert normalize("Five") == normalize("5") == "5"
    assert score("\\boxed{five}", ["5"])[1]


def test_aliases_and_f1():
    pred, em, f1 = score("\\boxed{NYC}", ["New York City", "NYC"])
    assert pred == "NYC" and em and f1 == 1.0
    _, em, f1 = score("\\boxed{the city of Velmora}", ["Velmora"])
    assert not em and 0 < f1 < 1


def test_articles_and_punctuation():
    assert score("\\boxed{The Beatles!}", ["Beatles"])[1]


def test_contains_answer_is_whole_word():
    assert contains_answer("the mayor is Bram Holt.", ["Bram Holt"])
    assert not contains_answer("nobody knows", ["no"])


def test_boxed_is_brace_matched_and_latex_free():
    assert extract_answer("\\boxed{\\text{Bram Holt}}") == "Bram Holt"
    assert extract_answer("so \\boxed{\\textbf{Denver}} it is") == "Denver"
    assert score("\\boxed{1{,}000}", ["1,000"])[1]
    assert extract_answer("\\boxed{Velmora") == "Velmora"          # cut off by the token cap
