"""Data model for llm-schema-matching-m2m.

Defines all dataclasses used throughout the pipeline: Attribute, Relation,
Parameters, Prompt, Answer, Decision, ResultPair, AttributeGroupPair,
ResultGroupPair, Result, RelationRelatednessResult, and PromptDesign.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional

if TYPE_CHECKING:
    # `from __future__ import annotations` keeps this lazy at runtime, so we
    # don't need openai installed to import this module — only to type-check
    # against it.
    from openai.types.completion_create_params import CompletionCreateParams


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Vote(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class Side(StrEnum):
    SOURCE = "source"
    TARGET = "target"


# ---------------------------------------------------------------------------
# Attribute / Relation
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Attribute:
    name: str
    description: Optional[str] = None
    included: bool = field(default=True, compare=False)

    def digest(self) -> str:
        return hashlib.blake2s(
            (self.name + str(self.description) + str(self.included)).encode()
        ).hexdigest()

    def __hash__(self) -> int:
        return hash(self.digest())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Attribute) and self.digest() == other.digest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Attribute":
        return Attribute(**data)


@dataclass
class Relation:
    name: str
    side: Side
    attributes: List[Attribute] = field(default_factory=list)
    description: Optional[str] = None

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                self.name
                + self.side.value
                + "".join([a.digest() for a in sorted(self.attributes)])
                + str(self.description)
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Relation":
        return Relation(
            name=data["name"],
            side=Side(data["side"]),
            attributes=[Attribute.from_dict(a) for a in data["attributes"]],
            description=data.get("description", None),
        )


# ---------------------------------------------------------------------------
# AttributePair
# ---------------------------------------------------------------------------

@dataclass(order=True)
class AttributePair:
    source: Attribute
    target: Attribute

    def digest(self) -> str:
        return hashlib.blake2s(
            (self.source.digest() + self.target.digest()).encode()
        ).hexdigest()

    def __str__(self) -> str:
        return f"{self.source.name}->{self.target.name}"

    def __hash__(self) -> int:
        return hash(self.digest())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AttributePair) and self.digest() == other.digest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AttributePair":
        return AttributePair(
            source=Attribute.from_dict(data["source"]),
            target=Attribute.from_dict(data["target"]),
        )


# ---------------------------------------------------------------------------
# Feedback (required by Parameters)
# ---------------------------------------------------------------------------

@dataclass
class Feedback:
    general: Optional[str] = None
    per_attribute: Dict[Attribute, str] = field(default_factory=dict)
    per_attribute_pair: Dict[AttributePair, str] = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                str(self.general)
                + "".join(
                    [
                        a.digest() + self.per_attribute[a]
                        for a in sorted(self.per_attribute)
                    ]
                )
                + "".join(
                    [
                        ap.digest() + self.per_attribute_pair[ap]
                        for ap in sorted(self.per_attribute_pair)
                    ]
                )
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Feedback":
        return Feedback(
            general=data.get("general", None),
            per_attribute={
                Attribute.from_dict(k): v
                for k, v in data.get("per_attribute", {}).items()
            },
            per_attribute_pair={
                AttributePair.from_dict(k): v
                for k, v in data.get("per_attribute_pair", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Parameters:
    source_relation: Relation
    target_relation: Relation
    llm_model: str
    feedback: Feedback = field(default_factory=Feedback)
    meta: Dict[str, str] = field(default_factory=dict)
    max_group_size: Optional[int] = None

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                self.source_relation.digest()
                + self.target_relation.digest()
                + self.feedback.digest()
                + self.llm_model
                + str(self.max_group_size or "")
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Parameters":
        return Parameters(
            source_relation=Relation.from_dict(data["source_relation"]),
            target_relation=Relation.from_dict(data["target_relation"]),
            llm_model=data.get("llm_model", ""),
            feedback=Feedback.from_dict(data.get("feedback", {})),
            meta=data.get("meta", {}),
            max_group_size=data.get("max_group_size", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def about_the_same(self, other: "Parameters") -> bool:
        return (
            (self.source_relation.digest() == other.source_relation.digest())
            and (self.target_relation.digest() == other.target_relation.digest())
        )


# ---------------------------------------------------------------------------
# PromptAttributePair (uses blake2s for consistency with all other digest methods in this module)
# ---------------------------------------------------------------------------

@dataclass
class PromptAttributePair:
    sources: List[Attribute] = field(default_factory=list)
    targets: List[Attribute] = field(default_factory=list)

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                "".join([a.digest() for a in sorted(self.sources)])
                + "".join([a.digest() for a in sorted(self.targets)])
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PromptAttributePair":
        return PromptAttributePair(
            sources=[Attribute.from_dict(a) for a in data["sources"]],
            targets=[Attribute.from_dict(a) for a in data["targets"]],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    parameters: Parameters
    attributes: PromptAttributePair
    prompt: CompletionCreateParams
    meta: Dict[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        prompt_digest = hashlib.blake2s(
            (
                self.prompt.get("model", "")
                + str(self.prompt.get("temperature", 1))
                + "".join([m["role"] + m["content"] for m in self.prompt["messages"]])
                + str(self.prompt.get("n", 1))
                + str(self.prompt.get("timeout", 60))
            ).encode()
        ).hexdigest()

        return hashlib.blake2s(
            (
                self.parameters.digest() + self.attributes.digest() + prompt_digest
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Prompt":
        return Prompt(
            parameters=Parameters.from_dict(data["parameters"]),
            attributes=PromptAttributePair.from_dict(data["attributes"]),
            # CompletionCreateParams is a pydantic model, not a dataclass.
            prompt=data["prompt"],
            meta=data.get("meta", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

@dataclass
class Answer:
    attributes: PromptAttributePair
    answer: str
    index: int = 0
    valid: bool = False
    meta: Dict[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                self.attributes.digest()
                + str(self.index)
                + self.answer
                + str(self.valid)
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Answer":
        return Answer(
            attributes=PromptAttributePair.from_dict(data["attributes"]),
            answer=data["answer"],
            index=data["index"],
            valid=data.get("valid", False),
            meta=data.get("meta", {}),
        )

    def __lt__(self, other: "Answer") -> bool:
        return self.index < other.index

    def __le__(self, other: "Answer") -> bool:
        return self.index <= other.index

    def __gt__(self, other: "Answer") -> bool:
        return self.index > other.index

    def __ge__(self, other: "Answer") -> bool:
        return self.index >= other.index

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Decision:
    vote: Vote
    explanation: str
    answer: Optional[Answer] = None

    def digest(self) -> str:
        return hashlib.blake2s(
            (self.vote.value + self.explanation).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Decision":
        raw_answer = data.get("answer")
        answer = Answer.from_dict(raw_answer) if raw_answer else None
        return Decision(
            vote=Vote(data["vote"]),
            explanation=data["explanation"],
            answer=answer,
        )


# ---------------------------------------------------------------------------
# ResultPair
# ---------------------------------------------------------------------------

@dataclass
class ResultPair:
    attributes: AttributePair
    votes: List[Decision] = field(default_factory=list)
    score: float = 0.0

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                self.attributes.digest()
                + "".join([d.digest() for d in sorted(self.votes)])
                + str(self.score)
            ).encode()
        ).hexdigest()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ResultPair":
        return ResultPair(
            attributes=AttributePair.from_dict(data["attributes"]),
            votes=[Decision.from_dict(v) for v in data["votes"]],
            score=data.get("score", 0.0),
        )


# ---------------------------------------------------------------------------
# AttributeGroupPair
# ---------------------------------------------------------------------------

@dataclass
class AttributeGroupPair:
    """A group of source attributes paired with a group of target attributes.

    `frozenset` is used so that equality and hashing are insertion-order
    independent.
    """
    sources: FrozenSet[Attribute]
    targets: FrozenSet[Attribute]

    def digest(self) -> str:
        s = "".join(sorted(a.digest() for a in self.sources))
        t = "".join(sorted(a.digest() for a in self.targets))
        return hashlib.blake2s((s + t).encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.digest())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AttributeGroupPair)
            and self.digest() == other.digest()
        )

    @property
    def is_one_to_one(self) -> bool:
        return len(self.sources) == 1 and len(self.targets) == 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [asdict(a) for a in sorted(self.sources)],
            "targets": [asdict(a) for a in sorted(self.targets)],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AttributeGroupPair":
        return AttributeGroupPair(
            sources=frozenset(Attribute.from_dict(a) for a in data["sources"]),
            targets=frozenset(Attribute.from_dict(a) for a in data["targets"]),
        )


# ---------------------------------------------------------------------------
# ResultGroupPair
# ---------------------------------------------------------------------------

@dataclass
class ResultGroupPair:
    """Accumulates votes for a group match. Score is YES votes / total votes."""
    attributes: AttributeGroupPair
    votes: List[Decision] = field(default_factory=list)
    score: float = 0.0

    def digest(self) -> str:
        v = "".join(sorted(d.digest() for d in self.votes))
        return hashlib.blake2s(
            (self.attributes.digest() + v).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attributes": self.attributes.to_dict(),
            "votes": [asdict(d) for d in self.votes],
            "score": self.score,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ResultGroupPair":
        return ResultGroupPair(
            attributes=AttributeGroupPair.from_dict(data["attributes"]),
            votes=[Decision.from_dict(v) for v in data["votes"]],
            score=data.get("score", 0.0),
        )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    parameters: Parameters
    name: Optional[str] = None
    pairs: Dict[AttributePair, ResultPair] = field(default_factory=dict)
    group_pairs: Dict[AttributeGroupPair, ResultGroupPair] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)

    def digest(self) -> str:
        pairs_part = "".join(
            [p.digest() + self.pairs[p].digest() for p in sorted(self.pairs)]
        )
        group_keys = sorted(self.group_pairs, key=lambda k: k.digest())
        groups_part = "".join(
            [k.digest() + self.group_pairs[k].digest() for k in group_keys]
        )
        return hashlib.blake2s(
            (self.parameters.digest() + pairs_part + groups_part).encode()
        ).hexdigest()

    def to_json(self) -> str:
        dct = {
            "parameters": self.parameters.to_dict(),
            "name": self.name,
            "pairs": {
                k.digest(): {"key": asdict(k), "value": asdict(v)}
                for k, v in self.pairs.items()
            },
            "group_pairs": {
                k.digest(): {"key": k.to_dict(), "value": v.to_dict()}
                for k, v in self.group_pairs.items()
            },
            "meta": self.meta,
        }
        return json.dumps(dct)

    @staticmethod
    def from_json(raw: str) -> "Result":
        return Result.from_dict(json.loads(raw))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Result":
        pairs: Dict[AttributePair, ResultPair] = {}
        for v in data.get("pairs", {}).values():
            ap = AttributePair.from_dict(v["key"])
            rp = ResultPair.from_dict(v["value"])
            pairs[ap] = rp

        group_pairs: Dict[AttributeGroupPair, ResultGroupPair] = {}
        for v in data.get("group_pairs", {}).values():
            agp = AttributeGroupPair.from_dict(v["key"])
            rgp = ResultGroupPair.from_dict(v["value"])
            group_pairs[agp] = rgp

        return Result(
            parameters=Parameters.from_dict(data["parameters"]),
            name=data.get("name", None),
            pairs=pairs,
            group_pairs=group_pairs,
            meta=data.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# RelationRelatednessResult
# ---------------------------------------------------------------------------

@dataclass
class RelationRelatednessResult:
    source_relation_name: str
    target_relation_name: str
    related: bool
    confidence: str  # "high" | "medium" | "low"
    reasoning: str

    def digest(self) -> str:
        return hashlib.blake2s(
            (
                self.source_relation_name
                + self.target_relation_name
                + str(self.related)
                + self.confidence
            ).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RelationRelatednessResult":
        return RelationRelatednessResult(**data)


# ---------------------------------------------------------------------------
# PromptDesign — Prompt template modes supported by the pipeline.
# ---------------------------------------------------------------------------

class PromptDesign(StrEnum):
    oneToN = "1-n"
    nToOne = "n-1"
    manyToMany = "m-m"
    relationRelatedness = "rel"
