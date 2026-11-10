from ragpipeline import quote_is_verified

SOURCE = """Manufacturers shall ensure that their product is accompanied by clear
instructions and safety information in a language which can be easily
understood by consumers."""


def test_exact_quote_is_verified():
    assert quote_is_verified("accompanied by clear instructions and safety information", SOURCE)


def test_whitespace_and_line_breaks_are_ignored():
    assert quote_is_verified("clear  instructions and\nsafety information", SOURCE)


def test_ellipsis_allowed_when_fragments_appear_in_order():
    assert quote_is_verified("Manufacturers shall ensure ... easily understood by consumers", SOURCE)


def test_ellipsis_rejected_when_fragments_out_of_order():
    assert not quote_is_verified("easily understood by consumers ... Manufacturers shall ensure", SOURCE)


def test_reworded_quote_is_rejected():
    assert not quote_is_verified("Manufacturers must give simple instructions", SOURCE)


def test_empty_quote_is_rejected():
    assert not quote_is_verified("  ...  ", SOURCE)
