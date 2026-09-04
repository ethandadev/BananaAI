"""Configuration: defaults, file loading, environment overrides, validation."""

from __future__ import annotations

from pathlib import Path

from tests.harness import Suite, temp_dir

from core.settings import (
    CONFIG_NAME, DEFAULTS, ConfigError, Settings, dumps, find, from_env, load,
    write_default,
)

suite = Suite("settings")
test = suite.test


@test
def defaults_load_without_a_file():
    with temp_dir() as tmp:
        s = load(tmp / "does-not-exist.toml", use_env=False)
    assert s.data["model"]["preset"] == "auto"
    assert s.data["hardware"]["max_hours"] == 72.0
    assert not s.validate(), s.validate()
    return "built-in defaults are valid on their own"


@test
def a_config_file_overrides_only_what_it_sets():
    with temp_dir() as tmp:
        path = tmp / CONFIG_NAME
        path.write_text('[model]\npreset = "small"\n', encoding="utf-8")
        s = load(path, use_env=False)
    assert s.data["model"]["preset"] == "small", "override not applied"
    assert s.data["model"]["vocab_size"] == DEFAULTS["model"]["vocab_size"], \
        "an unset key was clobbered instead of inherited"
    assert s.data["hardware"]["max_hours"] == 72.0, "an untouched section changed"
    return "one key overridden, the rest inherited"


@test
def a_written_config_round_trips():
    with temp_dir() as tmp:
        path = tmp / CONFIG_NAME
        write_default(path)
        reloaded = load(path, use_env=False)
    assert reloaded.data == DEFAULTS, "round trip changed the configuration"
    return f"{len(DEFAULTS)} sections survive write and reload"


@test
def every_default_value_is_serialisable():
    text = dumps(DEFAULTS)
    for section in DEFAULTS:
        assert f"[{section}]" in text, f"{section} missing from the output"
    # Both boolean spellings, checked on the serialiser rather than on the
    # defaults -- which today happen to be all True.
    both = dumps({"x": {"yes": True, "no": False}})
    assert "yes = true" in both and "no = false" in both, both
    return f"{sum(len(v) for v in DEFAULTS.values())} keys serialise"


@test
def unserialisable_values_are_refused():
    try:
        dumps({"model": {"weird": {1, 2, 3}}})
    except ConfigError as e:
        assert "serialise" in str(e)
        return "a set raises rather than emitting broken TOML"
    raise AssertionError("silently serialised an unsupported type")


@test
def strings_with_quotes_survive():
    text = dumps({"project": {"name": 'a "quoted" \\ name'}})
    import tomllib
    assert tomllib.loads(text)["project"]["name"] == 'a "quoted" \\ name'
    return "quotes and backslashes escaped correctly"


@test
def unknown_sections_are_rejected():
    with temp_dir() as tmp:
        path = tmp / CONFIG_NAME
        path.write_text('[nonsense]\nx = 1\n', encoding="utf-8")
        try:
            load(path, use_env=False)
        except ConfigError as e:
            assert "nonsense" in str(e) and "valid sections" in str(e)
            return "a typo'd section name fails loudly with the valid list"
    raise AssertionError("accepted an unknown section")


@test
def unknown_keys_are_rejected():
    """A silently ignored key is a setting the user thinks is applied."""
    with temp_dir() as tmp:
        path = tmp / CONFIG_NAME
        path.write_text('[model]\npreset = "small"\nprest = "base"\n', encoding="utf-8")
        try:
            load(path, use_env=False)
        except ConfigError as e:
            assert "prest" in str(e)
            return "a misspelled key is reported, not ignored"
    raise AssertionError("accepted an unknown key")


@test
def malformed_toml_names_the_file():
    with temp_dir() as tmp:
        path = tmp / CONFIG_NAME
        path.write_text("[model\npreset = broken", encoding="utf-8")
        try:
            load(path, use_env=False)
        except ConfigError as e:
            assert CONFIG_NAME in str(e), str(e)
            return "parse errors carry the file path"
    raise AssertionError("accepted malformed TOML")


@test
def environment_overrides_apply_with_the_right_types():
    config = {k: dict(v) for k, v in DEFAULTS.items()}
    out = from_env(config, {
        "BANANAAI_MODEL_PRESET": "large",
        "BANANAAI_HARDWARE_MAX_HOURS": "12.5",
        "BANANAAI_MODEL_VOCAB_SIZE": "16384",
        "BANANAAI_HARDWARE_COMPILE": "false",
        "PATH": "/should/be/ignored",
    })
    assert out["model"]["preset"] == "large"
    assert out["hardware"]["max_hours"] == 12.5 and isinstance(out["hardware"]["max_hours"], float)
    assert out["model"]["vocab_size"] == 16384 and isinstance(out["model"]["vocab_size"], int)
    assert out["hardware"]["compile"] is False, "boolean not parsed"
    return "str/int/float/bool all coerced from the environment"


@test
def unrelated_environment_variables_are_left_alone():
    config = {k: dict(v) for k, v in DEFAULTS.items()}
    out = from_env(config, {"BANANAAI_NOSUCH_KEY": "x", "HOME": "/home/someone"})
    assert out == config, "an unknown BANANAAI_ variable changed the config"
    return "unknown variables ignored rather than rejected"


@test
def config_is_found_by_searching_upward():
    with temp_dir() as tmp:
        (tmp / CONFIG_NAME).write_text("[model]\n", encoding="utf-8")
        nested = tmp / "a" / "b" / "c"
        nested.mkdir(parents=True)
        found = find(nested)
    assert found is not None and found.name == CONFIG_NAME, found
    return "found from three directories deep"


@test
def validation_catches_a_vocabulary_too_big_for_uint16():
    s = Settings({**{k: dict(v) for k, v in DEFAULTS.items()}})
    s.data["model"]["vocab_size"] = 70000
    problems = s.validate()
    assert any("uint16" in p for p in problems), problems
    return "70000 tokens flagged against the uint16 bin format"


@test
def validation_catches_contradictory_data_settings():
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()})
    s.data["data"]["custom_share"] = 0.5
    s.data["data"]["custom_dir"] = ""
    problems = s.validate()
    assert any("custom_dir" in p for p in problems), problems
    return "a custom share with no custom folder is reported"


@test
def validation_catches_bad_enums():
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()})
    s.data["hardware"]["device"] = "quantum"
    s.data["model"]["preset"] = "enormous"
    s.data["export"]["dtype"] = "float8"
    problems = s.validate()
    assert len(problems) >= 3, problems
    assert any("quantum" in p for p in problems)
    assert any("enormous" in p for p in problems)
    return "device, preset and dtype all validated against their options"


@test
def auto_values_become_none_for_the_planner():
    s = load(use_env=False)
    assert s.device_preference() is None, "'auto' device should mean 'let detect decide'"
    assert s.preset_preference() is None, "'auto' preset should mean 'let recommend decide'"
    s.data["hardware"]["device"] = "cpu"
    s.data["model"]["preset"] = "nano"
    assert s.device_preference() == "cpu" and s.preset_preference() == "nano"
    return "'auto' maps to None, explicit values pass through"


@test
def settings_produce_a_usable_plan():
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()})
    s.data["hardware"]["device"] = "cpu"
    s.data["hardware"]["max_hours"] = 24.0
    plan = s.plan()
    assert plan.model.vocab_size == DEFAULTS["model"]["vocab_size"]
    assert plan.estimated_hours <= 24.0
    return f"config -> plan: {plan.preset}, {plan.estimated_hours:.0f}h"




@test
def an_explicit_context_length_overrides_the_preset():
    """The batch schedule is derived from it, so it must be applied in plan()."""
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()})
    s.data["hardware"]["device"] = "cpu"
    s.data["model"]["preset"] = "nano"
    s.data["model"]["context_len"] = 128

    plan = s.plan()
    assert plan.model.context_len == 128, \
        f"context_len {plan.model.context_len} -- the override was ignored"
    per_micro = plan.train.micro_batch * 128
    assert plan.train.tokens_per_step % per_micro == 0, \
        "batch schedule no longer divides evenly after the override"
    plan.train.grad_accum_steps(plan.model.context_len)      # raises if inconsistent
    assert any("context length overridden" in w for w in plan.warnings)
    return f"128 applied, {plan.train.tokens_per_step:,} tokens/step re-derived"


@test
def zero_context_length_keeps_the_preset_default():
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()})
    s.data["hardware"]["device"] = "cpu"
    s.data["model"]["preset"] = "small"
    s.data["model"]["context_len"] = 0
    plan = s.plan()
    from core.hardware import LADDER
    assert plan.model.context_len == dict(LADDER)["small"]["context_len"]
    return f"0 means inherit ({plan.model.context_len})"


@test
def the_export_directory_does_not_shadow_the_package():
    """A default of "export" would write artifacts over the export/ package."""
    assert DEFAULTS["export"]["out_dir"] != "export", \
        "export.out_dir shadows the Python package that does the exporting"
    return f"artifacts go to {DEFAULTS['export']['out_dir']}/"


@test
def every_stage_directory_is_configurable():
    for section in ("train", "sft", "dpo", "export", "data"):
        assert "out_dir" in DEFAULTS[section], f"[{section}] has no out_dir"
    assert "data" in DEFAULTS["sft"] and "data" in DEFAULTS["dpo"]
    return "train, sft, dpo, export and data all have configurable outputs"


if __name__ == "__main__":
    p, n = suite.run()
    raise SystemExit(0 if p == n else 1)
