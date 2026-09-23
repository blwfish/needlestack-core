from needlestack_core.constants import model_names_from_tags_response


def test_model_names_from_tags_response_happy_path():
    data = {"models": [{"name": "qwen2.5vl:7b"}, {"name": "minicpm-v:latest"}]}
    assert model_names_from_tags_response(data) == ["qwen2.5vl:7b", "minicpm-v:latest"]


def test_model_names_from_tags_response_missing_models_key():
    assert model_names_from_tags_response({}) == []


def test_model_names_from_tags_response_models_not_a_list():
    assert model_names_from_tags_response({"models": "not a list"}) == []


def test_model_names_from_tags_response_entry_missing_name():
    data = {"models": [{"size": 123}, {"name": "ok:model"}]}
    assert model_names_from_tags_response(data) == ["ok:model"]


def test_model_names_from_tags_response_non_dict_entry():
    data = {"models": [None, "garbage", {"name": "ok:model"}]}
    assert model_names_from_tags_response(data) == ["ok:model"]
