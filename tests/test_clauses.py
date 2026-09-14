from app.pipeline.clauses import question_clauses


def test_single_question_is_left_whole():
    message = "Como uma reclamacao de cliente deve ser tratada?"
    assert question_clauses(message) == [message]


def test_two_questions_joined_by_e_are_split():
    parts = question_clauses(
        "Qual foi a receita do mes de marco e qual norma regula a retencao desses registros?"
    )
    assert parts == ["Qual foi a receita do mes de marco", "qual norma regula a retencao desses registros"]


def test_two_sentences_are_split():
    parts = question_clauses("Qual foi a receita do Sudeste? Quem aprova uma compra de trinta mil?")
    assert len(parts) == 2


def test_english_compound_is_split():
    parts = question_clauses("What was the Sudeste revenue and who approves a purchase of that size?")
    assert len(parts) == 2


def test_e_as_the_verb_does_not_split():
    """"Uma exportacao ... e reportada" — the "e" is the verb *é*, written unaccented.

    One of the two false positives the interrogative test was added to remove: the
    first half asks nothing, so the message is one question that happens to contain
    the letter.
    """
    message = "Uma exportacao sem anonimizacao e reportada para quem?"
    assert question_clauses(message) == [message]


def test_e_joining_two_noun_phrases_does_not_split():
    """The other false positive: "e" coordinates nouns, not questions."""
    message = "Qual a diferenca entre um atraso comunicado e um atraso descoberto?"
    assert question_clauses(message) == [message]


def test_a_fragment_too_short_to_be_a_question_does_not_split():
    message = "Quais produtos e quais regioes?"
    assert question_clauses(message) == [message]
