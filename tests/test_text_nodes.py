import pytest


def run(plugin, text, delimiter=",", start_index=0, skip_every=0, max_count=10, trim=False, regex=False):
    return plugin.NODE_CLASS_MAPPINGS["TankTextSplit"]().split(text, delimiter, start_index, skip_every, max_count, trim, regex)


def test_matches_mixlab_selection_rules(plugin):
    assert run(plugin, "a,b,,c, ,d,e") == (["a", "b", "c", "d", "e"], 5)
    assert run(plugin, "a,b,c,d,e,f", start_index=1, skip_every=1, max_count=2) == (["b", "d"], 2)
    assert run(plugin, " a , b ") == ([" a ", " b "], 2)
    assert run(plugin, "  whole text \n", delimiter="") == (["whole text"], 1)


def test_escaped_and_chinese_delimiters(plugin):
    assert run(plugin, "第一行\n\n第二行", delimiter="\\n") == (["第一行", "第二行"], 2)
    assert run(plugin, "a\tb", delimiter="\\t") == (["a", "b"], 2)
    assert run(plugin, "红色，蓝色，绿色", delimiter="，") == (["红色", "蓝色", "绿色"], 3)
    assert run(plugin, "甲|乙", delimiter="|") == (["甲", "乙"], 2)
    assert run(plugin, r"a\db", delimiter=r"\d") == (["a", "b"], 2)


def test_trim_and_empty_result(plugin):
    assert run(plugin, " a ,\n b ", trim=True) == (["a", "b"], 2)
    with pytest.raises(ValueError, match="no items"):
        run(plugin, "a,b", start_index=5)


def test_regex_mode(plugin):
    assert run(plugin, "红,蓝，绿", delimiter="[,，]", regex=True) == (["红", "蓝", "绿"], 3)
    assert run(plugin, "a\n\n\nb", delimiter="\\n+", regex=True) == (["a", "b"], 2)
    assert run(plugin, "a1b2c", delimiter="(\\d)", regex=True) == (["a", "b", "c"], 3)
    with pytest.raises(ValueError, match="regular expression"):
        run(plugin, "a", delimiter="[", regex=True)
