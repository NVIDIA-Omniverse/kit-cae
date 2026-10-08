# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Disposable committed-evaluation evidence, never authoritative product state."""

from __future__ import annotations

from dataclasses import dataclass

from ._stage_io import _path_query, _release_path_query


@dataclass
class Evaluation:
    """Update-local metadata comparison state for one binding.

    Inputs and recipe describe the current evaluation; signature identifies the
    implementation and device. Previous is a Record that Session has already checked
    against committed Stage state and that signature. Publication completion can
    turn this evidence into a candidate, not directly into a successful baseline.
    recipe_was_changed is the native range result computed before product writes.
    No scientific intermediates or previous recipe values are stored here.
    """

    inputs: object
    recipe: object
    signature: tuple
    previous: object = None
    recipe_was_changed: bool = True

    def recipe_changed(self):
        """Return the precomputed native comparison, also guarding dependency changes.

        Session queries native history before any product writes. A changed read set
        or unknown schema cannot establish equivalence even if old properties held.
        """
        return (
            self.previous is None
            or self.recipe is None
            or self.previous.recipe is None
            or self.recipe.dependencies != self.previous.recipe.dependencies
            or self.recipe_was_changed
        )

    def input_changed(self, role, components):
        """Compare selected effective input identities against the validated baseline.

        Components names topology/geometry entries or field:<output-role> entries in the
        plan. Reject an empty or unavailable selection instead of silently declaring it
        unchanged. A missing prior role/component or unknown current identity requires
        recomputation, even if both evaluations lack evidence.
        """
        current = self.inputs.identities[role]
        if not components or not components <= current.keys():
            raise ValueError(f"comparison must select available components of input {role!r}")
        previous = self.previous.inputs.identities.get(role, {}) if self.previous else {}
        return any(current[name] is None or current[name] != previous.get(name) for name in components)

    def unchanged(self):
        """Require matching recipe and all requested scientific identities before skipping.

        Product validity is a precondition supplied through previous by Session; this
        method compares only evaluation evidence and performs no Stage access.
        """
        return not self.recipe_changed() and all(
            not self.input_changed(role, set(evidence)) for role, evidence in self.inputs.identities.items()
        )


@dataclass
class Record:
    """Metadata-only evidence of a completed product publication.

    Ordinal and attributes identify the original product publication on Stage.
    product_checked_ordinal is a rolling validation checkpoint, not a new write.
    Inputs, recipe, and signature describe what produced it. A Record may be an
    uncommitted candidate or a validated baseline; History owns that distinction.
    It holds no Visualization Product or authoritative copy of Stage state.
    """

    inputs: object
    recipe: object
    signature: tuple
    ordinal: int
    attributes: frozenset[str]
    product_checked_ordinal: int | None = None


@dataclass
class History:
    """Bounded candidate and baseline records for one Session/operator binding.

    A candidate represents completed writes awaiting observed commitment. A baseline
    has passed native product validation, but must be checked again before reuse.
    Session drops the history on binding invalidation or failure and on close.
    Methods are used by the Session evaluation worker, not concurrent writers.
    """

    baseline: Record | None = None
    candidate: Record | None = None
    recipe_invalidated: bool = False

    def committed(self, stage, path, floor, *, oldest=None):
        """Promote a verifiable candidate and return the still-valid committed baseline.

        Use the observed write floor and native product evidence. Once the floor covers
        a candidate, discard that candidate slot whether validation succeeds or fails;
        an older baseline survives only if its product remains valid. Uncommitted
        candidates never replace the baseline. Call before any Update product writes,
        which can make reads of shared native attribute storage unavailable.
        """
        promoted = False
        if self.candidate is not None and self.candidate.ordinal <= floor:
            if product_intact(stage, path, self.candidate, floor, oldest=oldest):
                self.baseline = self.candidate
                self.baseline.product_checked_ordinal = floor
                self.recipe_invalidated = False
                promoted = True
            self.candidate = None
        baseline = self.baseline
        # Recheck even at the same floor: an uncommitted publication may have
        # overwritten storage since the preceding Update. Only skip a duplicate
        # check for the candidate just validated within this call.
        if baseline is not None and not promoted:
            if product_intact(stage, path, baseline, floor, oldest=oldest):
                baseline.product_checked_ordinal = floor
            else:
                self.baseline = baseline = None
        return baseline

    def recipe_changed(self, stage, floor, *, oldest=None):
        """Roll checked recipe evidence or latch a detected change, even while hidden.

        Only a committed replacement product can clear this flag. Expired
        history must never erase a change already found by an earlier Update.
        A no-change check advances only the existing dependency set's ordinal;
        it neither validates the current recipe nor refreshes scientific inputs.
        """
        baseline = self.baseline
        if baseline is None or baseline.recipe is None:
            return True
        if not self.recipe_invalidated:
            self.recipe_invalidated = baseline.recipe.changed(stage, floor, oldest=oldest)
            if not self.recipe_invalidated:
                baseline.recipe = Recipe(floor, baseline.recipe.dependencies)
        return self.recipe_invalidated

    def verified_unchanged(self, evaluation):
        """Refresh recipe evidence after a full gate hit without authoring Stage.

        Keep the publication ordinal and owned attributes. The new plan records
        the observed committed floor and current input metadata, not payloads.
        Product evidence was independently refreshed before any Update writes.
        """
        self.baseline.recipe = evaluation.recipe
        self.baseline.inputs = evaluation.inputs

    def published(self, evaluation, ordinal, attributes):
        """Stage a candidate after all product writes complete successfully.

        Store request identities, sample descriptions, and publication metadata only.
        This does not advance the write floor or promote the committed baseline.
        """
        self.candidate = Record(evaluation.inputs, evaluation.recipe, evaluation.signature, ordinal, attributes)


def product_intact(stage, path, record, floor, *, oldest=None):
    """Check current product membership and changes since its last validation.

    Require a sealed publication, all owned attributes still present, and no
    retained changes after the last successful check (initially publication).
    Assume no changes in an expired prefix under the application check-frequency
    contract. Missing current attributes or native read errors still invalidate.
    Only membership metadata is inspected, never geometry payloads.

    Relies on exclusive Session ownership and the caller-reserved output ordinal.
    External same-ordinal product writes violate that application contract. Native
    frontier/query setup failures propagate rather than establishing validity.
    """
    import ovstage

    if record.ordinal > floor or not record.attributes:
        return False
    if oldest is None:
        oldest = oldest_preserved_ordinal(stage)
    checked = record.product_checked_ordinal if record.product_checked_ordinal is not None else record.ordinal
    if checked > floor:
        return False
    paths, path_list, query = _path_query(stage, path)
    try:
        expected = {paths.intern_token(name) for name in record.attributes}
        seen = set()
        # Change membership is per key, unlike a shared read-group ordinal.
        # Expiry only excludes an unavailable prefix; never ignore the suffix.
        start = max(checked + 1, oldest)
        if start <= floor:
            with stage.read_attributes(query, sorted(expected), ovstage.OrdinalRange.between(start, floor)) as read:
                read.wait()
                for group in read.groups():
                    stage.release_group(group)
                    return False
        with stage.read_attributes(query, sorted(expected), ovstage.OrdinalRange.latest(floor)) as read:
            read.wait()
            for group in read.groups():
                try:
                    if group.is_delete:
                        return False
                    seen.add(group.attribute)
                finally:
                    stage.release_group(group)
        return seen == expected
    except Exception:
        # Unavailable or uncommitted storage is not evidence of an intact product.
        return False
    finally:
        _release_path_query(paths, path_list, query)


@dataclass(frozen=True)
class Recipe:
    """Property dependencies observed at a committed Stage ordinal, without values.

    Ordinal is the actual write floor observed for evaluation or the last no-change
    check of these dependencies, not the caller's minimum Input Ordinal or later
    output publication. Recipe checks can roll independently of scientific inputs,
    including while hidden. Only candidate commitment clears observed invalidation.
    """

    ordinal: int
    dependencies: tuple[tuple[str, tuple[str, ...]], ...]

    def changed(self, stage, current_ordinal, *, oldest=None):
        """Ask native history whether any captured property was written or deleted.

        Query (observed, current] before Update product writes. Include absent reads
        so authoring a formerly defaulted property invalidates the recipe. A rewrite
        of an identical value counts as changed; no payload is inspected. Assume
        no changes in an expired prefix and inspect the retained suffix. Other
        native read failures remain invalidating, not an assume-unchanged case.
        This covers explicit property dependencies, not scene hierarchy changes.
        """
        import ovstage

        if current_ordinal < self.ordinal:
            return True
        if current_ordinal == self.ordinal:
            return False
        start = self.ordinal + 1
        if oldest is None:
            oldest = oldest_preserved_ordinal(stage)
        start = max(start, oldest)
        if start > current_ordinal:
            return False
        # Batch only identical property sets: a union across unlike sets would
        # also select unrelated (path, attribute) pairs, including our outputs.
        batches = {}
        for path, names in self.dependencies:
            batches.setdefault(tuple(sorted(set(names))), []).append(path)
        for names, batch_paths in batches.items():
            paths, path_list, query = _path_query(stage, batch_paths)
            try:
                tokens = [paths.intern_token(name) for name in names]
                with stage.read_attributes(query, tokens, ovstage.OrdinalRange.between(start, current_ordinal)) as read:
                    read.wait()
                    for group in read.groups():
                        # Tombstones are changes too, despite containing no tensors.
                        stage.release_group(group)
                        return True
            except ovstage.OvstageError:
                return True
            finally:
                _release_path_query(paths, path_list, query)
        return False


def oldest_preserved_ordinal(stage):
    """Read the native history frontier once for an application's stable Update.

    Session shares this observation across product and recipe checks. Standalone
    checks may read it themselves. Errors propagate rather than proving reuse.
    """
    with stage.get_oldest_preserved_ordinal() as frontier:
        frontier.wait()
        return frontier.fetch()
