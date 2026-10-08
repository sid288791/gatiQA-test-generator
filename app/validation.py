"""Input sanitization, no-invention guards, and contracts validation.

Merged from the former guards.py + contracts.py — they share one concern:
ensuring the LLM draft contains only values traceable to the user's input
or explicitly approved defaults.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from app.models import (
    Draft,
    ExpectationSchemaSpec,
    MissingExpectation,
    ValidationIssue,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]*")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CURRENCY_RE = re.compile(r"(?<![A-Za-z])[A-Z]{3}(?![A-Za-z])")


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


class InputTooLongError(ValueError):
    pass


def sanitize_untrusted_text(text: str, max_chars: int) -> str:
    cleaned = _CONTROL_CHARS.sub(" ", text).strip()
    if not cleaned:
        raise InputTooLongError("Input is empty after sanitization.")
    if len(cleaned) > max_chars:
        raise InputTooLongError(
            f"Input exceeds the maximum of {max_chars} characters (got {len(cleaned)})."
        )
    return cleaned


# ---------------------------------------------------------------------------
# No-invention guard helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _numbers(text: str) -> set[str]:
    found: set[str] = set()
    for raw in _NUMBER_RE.findall(text):
        try:
            found.add(str(Decimal(raw.replace(",", ""))))
        except InvalidOperation:
            continue
    return found


def _currencies(text: str) -> set[str]:
    return set(_CURRENCY_RE.findall(text))


def _norm_decimal(value: str) -> str | None:
    try:
        return str(Decimal(value))
    except InvalidOperation:
        return None


# ---------------------------------------------------------------------------
# No-invention guard (traceability check)
# ---------------------------------------------------------------------------


def check_draft_traceability(
    draft: Draft,
    sanitized_input: str,
    approved_defaults: dict[str, str],
) -> list[ValidationIssue]:
    """Every concrete value in the draft must trace to the input or an approved default.

    Invented values are stripped in-place and reported.
    """
    issues: list[ValidationIssue] = []
    text_tokens = _tokens(sanitized_input)
    text_lower = sanitized_input.lower()
    text_numbers = _numbers(sanitized_input)
    text_currencies = _currencies(sanitized_input)

    req = draft.request

    if req.query is not None:
        missing_words = [
            w for w in re.findall(r"[A-Za-z]{3,}", req.query)
            if w.lower() not in text_lower
        ]
        if missing_words:
            issues.append(ValidationIssue(
                code="INVENTED_QUERY",
                message=f"Query contains terms not in description: {', '.join(missing_words)}.",
                path="request.query",
            ))
            req.query = None

    if req.storeId is not None and req.storeId.lower() not in text_tokens:
        issues.append(ValidationIssue(
            code="INVENTED_STORE_ID",
            message=f"Store ID '{req.storeId}' not in description.",
            path="request.storeId",
        ))
        req.storeId = None

    if req.endpointUrl is not None and req.endpointUrl != approved_defaults.get("endpointUrl"):
        issues.append(ValidationIssue(
            code="INVENTED_ENDPOINT_URL",
            message="Endpoint URL is not an approved default.",
            path="request.endpointUrl",
        ))
        req.endpointUrl = None

    kept_products = []
    for product in draft.expectations.products:
        if product.sku.lower() not in text_tokens:
            issues.append(ValidationIssue(
                code="INVENTED_SKU",
                message=f"SKU '{product.sku}' not in description.",
                path="expectations.products[].sku",
            ))
            continue

        if (
            product.maximumRank is not None
            and str(Decimal(product.maximumRank)) not in text_numbers
        ):
            issues.append(ValidationIssue(
                code="INVENTED_THRESHOLD",
                message=f"Rank {product.maximumRank} for '{product.sku}' not in description.",
                path="expectations.products[].maximumRank",
            ))
            product.maximumRank = None

        if product.price is not None:
            _check_price(product.price, product.sku, text_numbers, text_currencies, approved_defaults, issues)
            if product.price.value is None:
                product.price = None

        if product.availability is not None:
            allowed = product.availability == approved_defaults.get("availability")
            if not allowed and product.availability.upper() not in {t.upper() for t in text_tokens}:
                issues.append(ValidationIssue(
                    code="INVENTED_AVAILABILITY",
                    message=f"Availability '{product.availability}' for '{product.sku}' not in description.",
                    path="expectations.products[].availability",
                ))
                product.availability = None

        kept_badges = []
        for badge in product.badges:
            if badge.lower() in text_tokens:
                kept_badges.append(badge)
            else:
                issues.append(ValidationIssue(
                    code="INVENTED_BADGE",
                    message=f"Badge '{badge}' for '{product.sku}' not in description.",
                    path="expectations.products[].badges",
                ))
        product.badges = kept_badges
        kept_products.append(product)

    draft.expectations.products = kept_products

    rc = draft.expectations.resultCount
    if rc is not None:
        for field_name in ("greaterThan", "lessThan", "equalTo"):
            val = getattr(rc, field_name, None)
            if val is not None and str(Decimal(val)) not in text_numbers:
                issues.append(ValidationIssue(
                    code="INVENTED_RESULT_COUNT",
                    message=f"resultCount.{field_name} value {val} not in description.",
                    path=f"expectations.resultCount.{field_name}",
                ))
                setattr(rc, field_name, None)
        if rc.greaterThan is None and rc.lessThan is None and rc.equalTo is None:
            draft.expectations.resultCount = None

    return issues


def _check_price(price, sku, text_numbers, text_currencies, approved_defaults, issues):
    value = _norm_decimal(price.value)
    if value is None or value not in text_numbers:
        issues.append(ValidationIssue(
            code="INVENTED_PRICE",
            message=f"Price '{price.value}' for '{sku}' not in description.",
            path="expectations.products[].price.value",
        ))
        price.value = None
        return

    if price.currency is not None:
        if price.currency not in text_currencies and price.currency != approved_defaults.get("currency"):
            issues.append(ValidationIssue(
                code="INVENTED_CURRENCY",
                message=f"Currency '{price.currency}' for '{sku}' not in description.",
                path="expectations.products[].price.currency",
            ))
            price.currency = None

    if price.tolerance is not None and price.tolerance != approved_defaults.get("priceTolerance"):
        issues.append(ValidationIssue(
            code="UNAPPROVED_DEFAULT",
            message=f"Tolerance '{price.tolerance}' for '{sku}' was not approved.",
            path="expectations.products[].price.tolerance",
        ))
        price.tolerance = None


# ---------------------------------------------------------------------------
# Contracts validation (expectation completeness)
# ---------------------------------------------------------------------------

DEFAULT_EXPECTATION_SCHEMAS: dict[str, list[str]] = {
    "sku_match": ["products[].sku"],
    "top_k": ["products[].maximumRank"],
    "pricing": ["products[].price.value", "products[].price.currency"],
    "availability": ["products[].availability"],
    "badges": ["products[].badges"],
    "result_count": ["resultCount"],
}

_KNOWN_EVALUATORS = set(DEFAULT_EXPECTATION_SCHEMAS)


def resolve_expectation_schemas(
    schemas: list[ExpectationSchemaSpec] | None,
    enabled_evaluators: list[str],
) -> tuple[dict[str, list[str]], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    resolved: dict[str, list[str]] = {}
    for evaluator in enabled_evaluators:
        if evaluator not in _KNOWN_EVALUATORS:
            issues.append(ValidationIssue(
                code="UNKNOWN_EVALUATOR",
                message=f"Evaluator '{evaluator}' has no known expectation schema.",
                path="profile.enabledEvaluators",
            ))
    if schemas:
        for spec in schemas:
            resolved[spec.evaluator] = list(spec.requiredFields)
    else:
        for evaluator in enabled_evaluators:
            if evaluator in _KNOWN_EVALUATORS:
                resolved[evaluator] = list(DEFAULT_EXPECTATION_SCHEMAS[evaluator])
    return resolved, issues


def _field_missing(draft: Draft, path: str) -> bool:
    if path == "resultCount":
        rc = draft.expectations.resultCount
        return rc is None or (
            rc.greaterThan is None and rc.lessThan is None and rc.equalTo is None
        )
    if not path.startswith("products[]."):
        return False
    field_path = path[len("products[]."):]
    products = draft.expectations.products
    if not products:
        return True
    for product in products:
        obj = product
        for part in field_path.split("."):
            if obj is None:
                break
            obj = getattr(obj, part, None)
        if obj is None or obj == [] or obj == "":
            return True
    return False


def validate_expectations(
    draft: Draft,
    enabled_evaluators: list[str],
    schemas: dict[str, list[str]],
) -> list[MissingExpectation]:
    missing: list[MissingExpectation] = []
    for evaluator in enabled_evaluators:
        required = schemas.get(evaluator)
        if not required:
            continue
        for field in required:
            if _field_missing(draft, field):
                skus = [p.sku for p in draft.expectations.products]
                missing.append(MissingExpectation(
                    evaluator=evaluator,
                    field=field,
                    productSku=skus[0] if len(skus) == 1 else None,
                    message=(
                        f"Evaluator '{evaluator}' requires '{field}' but it is missing "
                        f"for product(s) {skus or '(none)'}."
                    ),
                ))
    return missing


def validate_with_contracts(
    draft: Draft,
    enabled_evaluators: list[str],
    schemas: dict[str, list[str]],
) -> list[MissingExpectation]:
    """Seam for gatiQA-contracts: delegates to external package if available."""
    try:
        from gatiqa_contracts import validate_expectations as _external  # type: ignore
    except ImportError:
        return validate_expectations(draft, enabled_evaluators, schemas)
    return _external(draft, enabled_evaluators, schemas)  # type: ignore
