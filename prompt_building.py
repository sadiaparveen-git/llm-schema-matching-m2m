"""Prompt construction

build_prompts() renders the 1:1 prompt templates (oneToN, nToOne) following
a per-attribute iteration approach: one Prompt per source attribute for
oneToN mode, one per target attribute for nToOne mode.

build_m2m_prompts() renders the manyToMany template — one prompt per relation
pair, presenting only residual (unresolved) attributes, injecting max_group_size.

build_relatedness_prompts() renders the relationRelatedness template — one
prompt per relation pair.

System-role messages are passed through natively (not flattened to user role).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
from typing import Dict, List, Tuple, Union

from jinja2 import Environment

from config import config
from models import (
    Attribute,
    Parameters,
    Prompt,
    PromptAttributePair,
    PromptDesign,
)


# ---------------------------------------------------------------------------
# Template loading (cached per process)
# ---------------------------------------------------------------------------

@functools.cache
def _load_template(template_name: str) -> List[Dict[str, str]]:
    """Load and parse a Jinja2 JSON prompt template from TEMPLATE_DIR."""
    path = os.path.join(config["TEMPLATE_DIR"], f"{template_name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Template iteration
# ---------------------------------------------------------------------------

def _template_iterator(
    template: List[Dict[str, str]],
    sources: List[Attribute],
    targets: List[Attribute],
):
    """Yield (part, source_attr, target_attr) triples.

    Parts that reference {{source_attribute}} are expanded once per source.
    Parts that reference {{target_attribute}} are expanded once per target.
    All other parts are yielded once.
    """
    for part in template:
        content = part["content"]
        if "{{source_attribute.name}}" in content:
            for s in sources:
                yield part, s, targets[0]
        elif "{{target_attribute.name}}" in content:
            for t in targets:
                yield part, sources[0], t
        else:
            yield part, sources[0], targets[0]


# ---------------------------------------------------------------------------
# Single-prompt rendering
# ---------------------------------------------------------------------------

def _render_messages(
    sources: List[Attribute],
    targets: List[Attribute],
    parameters: Parameters,
    template: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Render template into a list of chat messages.

    System-role messages are passed through unchanged — no flattening to
    user role.
    """
    env = Environment()
    messages = []
    for part, source, target in _template_iterator(template, sources, targets):
        rendered_content = env.from_string(part["content"]).render(
            source_relation=parameters.source_relation,
            source_attribute=source,
            target_relation=parameters.target_relation,
            target_attribute=target,
            feedback=parameters.feedback,
        )
        if rendered_content:
            messages.append({"role": part["role"], "content": rendered_content})
    return messages


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_model(parameters: Parameters) -> str:
    if parameters.llm_model:
        return parameters.llm_model
    if config["LLM_PROVIDER"] == "anthropic":
        return config["ANTHROPIC_MODEL"]
    return config["OPENAI_MODEL"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_prompts(
    parameters: Parameters,
    modes: List[PromptDesign],
) -> List[Prompt]:
    """Build 1:1 prompts for the given modes (oneToN, nToOne).

    For oneToN: one Prompt per included source attribute (vs all target attrs).
    For nToOne: one Prompt per included target attribute (vs all source attrs).
    System-role messages in templates are passed through natively.

    Returns a flat list of Prompt objects.
    """
    model = _resolve_model(parameters)
    prompts: List[Prompt] = []

    for mode in modes:
        template_name = _mode_to_template_name(mode)
        template = _load_template(template_name)

        source_card, target_card = mode.value.split("-")

        included_sources = [a for a in parameters.source_relation.attributes if a.included]
        included_targets = [a for a in parameters.target_relation.attributes if a.included]

        # Build the Cartesian product of source/target iterators per mode.
        # oneToN ("1-n"): iterate over each source, bundle all targets together.
        # nToOne ("n-1"): bundle all sources together, iterate over each target.
        if source_card == "n":
            sources_iter: List[List[Attribute]] = [included_sources]
        else:
            sources_iter = [[a] for a in included_sources]

        if target_card == "n":
            targets_iter: List[List[Attribute]] = [included_targets]
        else:
            targets_iter = [[a] for a in included_targets]

        for src_list in sources_iter:
            for tgt_list in targets_iter:
                messages = _render_messages(src_list, tgt_list, parameters, template)
                prompt_dict: Dict = {
                    "model": model,
                    "temperature": config["OPENAI_TEMPERATURE"],
                    "messages": messages,
                    "n": config["OPENAI_N"],
                }
                prompts.append(
                    Prompt(
                        parameters=parameters,
                        attributes=PromptAttributePair(
                            sources=src_list,
                            targets=tgt_list,
                        ),
                        prompt=prompt_dict,
                    )
                )

    return prompts


def _mode_to_template_name(mode: PromptDesign) -> str:
    """Map a PromptDesign enum value to the JSON template filename stem."""
    mapping = {
        PromptDesign.oneToN: "oneToN",
        PromptDesign.nToOne: "nToOne",
        PromptDesign.manyToMany: "manyToMany",
        PromptDesign.relationRelatedness: "relationRelatedness",
    }
    if mode not in mapping:
        raise ValueError(f"No template registered for mode {mode!r}")
    return mapping[mode]


def _render_relation_prompt(
    parameters: Parameters,
    template: List[Dict[str, str]],
    extra_vars: Dict,
) -> List[Dict[str, str]]:
    """Render a relation-level template (not per-attribute-pair).

    All messages rendered with the full parameters context plus any
    extra_vars passed in (e.g., max_group_size).
    """
    env = Environment()
    messages = []
    for part in template:
        rendered = env.from_string(part["content"]).render(
            source_relation=parameters.source_relation,
            target_relation=parameters.target_relation,
            feedback=parameters.feedback,
            **extra_vars,
        )
        if rendered:
            messages.append({"role": part["role"], "content": rendered})
    return messages


def build_m2m_prompts(
    parameters: Parameters,
    residual_sources: List[Attribute],
    residual_targets: List[Attribute],
    max_group_size: int,
) -> List[Prompt]:
    """Build the many-to-many prompt for residual attributes.

    One prompt per relation pair. Sets attr.included=True only for residual
    attributes when rendering — all other attributes appear with included=False
    so the Jinja {% if attr.included %} guard hides them.

    Returns a list with exactly one Prompt.
    """
    template = _load_template("manyToMany")
    model = _resolve_model(parameters)

    residual_src_names = {a.name for a in residual_sources}
    residual_tgt_names = {a.name for a in residual_targets}

    # Build shallow-copy relations where only residual attrs are included.
    src_attrs = [
        dataclasses.replace(a, included=(a.name in residual_src_names))
        for a in parameters.source_relation.attributes
    ]
    tgt_attrs = [
        dataclasses.replace(a, included=(a.name in residual_tgt_names))
        for a in parameters.target_relation.attributes
    ]

    patched_src = dataclasses.replace(parameters.source_relation, attributes=src_attrs)
    patched_tgt = dataclasses.replace(parameters.target_relation, attributes=tgt_attrs)
    patched_params = dataclasses.replace(
        parameters,
        source_relation=patched_src,
        target_relation=patched_tgt,
    )

    messages = _render_relation_prompt(
        patched_params,
        template,
        extra_vars={"max_group_size": max_group_size},
    )

    prompt_dict: Dict = {
        "model": model,
        "temperature": config["OPENAI_TEMPERATURE"],
        "messages": messages,
        "n": config["OPENAI_N"],
    }

    return [
        Prompt(
            parameters=patched_params,
            attributes=PromptAttributePair(
                sources=residual_sources,
                targets=residual_targets,
            ),
            prompt=prompt_dict,
        )
    ]


def build_relatedness_prompts(parameters: Parameters) -> List[Prompt]:
    """Build the relation-relatedness prompt.

    One prompt per relation pair. All attributes are included as-is.
    Returns a list with exactly one Prompt.
    """
    template = _load_template("relationRelatedness")
    model = _resolve_model(parameters)

    messages = _render_relation_prompt(parameters, template, extra_vars={})

    prompt_dict: Dict = {
        "model": model,
        "temperature": config["OPENAI_TEMPERATURE"],
        "messages": messages,
        "n": config["OPENAI_N"],
    }

    included_sources = [a for a in parameters.source_relation.attributes if a.included]
    included_targets = [a for a in parameters.target_relation.attributes if a.included]

    return [
        Prompt(
            parameters=parameters,
            attributes=PromptAttributePair(
                sources=included_sources,
                targets=included_targets,
            ),
            prompt=prompt_dict,
        )
    ]
