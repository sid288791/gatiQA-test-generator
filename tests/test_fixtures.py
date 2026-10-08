"""Synthetic generation-quality fixture set (data-driven, inline)."""

import asyncio

import pytest

from app.models import TestGenerationRequest
from app.service import TestGenerationService
from tests.fake_provider import scripted_call

FIXTURES = [
    {
        "id": "complete_search_pricing",
        "request": {
            "input": "Search for cordless drill in store 123. SKU-100 should appear in the top 3 and cost USD 99.99.",
            "profile": {"profileId": "search-basic", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
            "approvedDefaults": {"priceTolerance": "0.01"},
        },
        "modelOutput": {
            "request": {"query": "cordless drill", "storeId": "123"},
            "expectations": {"products": [
                {"sku": "SKU-100", "maximumRank": 3, "price": {"value": "99.99", "currency": "USD"}}
            ]},
            "clarifications": [],
        },
        "expectedStatus": "READY_FOR_REVIEW",
        "expectedIssues": [],
        "expectedClarificationCount": 0,
    },
    {
        "id": "ambiguous_currency",
        "request": {
            "input": "Search for hammer in store 55. SKU-200 should appear in the top 5 and cost 19.99.",
            "profile": {"profileId": "p", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
        },
        "modelOutput": {
            "request": {"query": "hammer", "storeId": "55"},
            "expectations": {"products": [
                {"sku": "SKU-200", "maximumRank": 5, "price": {"value": "19.99"}}
            ]},
            "clarifications": [],
        },
        "expectedStatus": "NEEDS_CLARIFICATION",
        "expectedIssues": [],
        "expectedMissing": [("pricing", "products[].price.currency")],
        "expectedClarificationCount": 1,
    },
    {
        "id": "missing_price_for_pricing_evaluator",
        "request": {
            "input": "Search for wrench in store 9. SKU-300 should appear in the top 2.",
            "profile": {"profileId": "p", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
        },
        "modelOutput": {
            "request": {"query": "wrench", "storeId": "9"},
            "expectations": {"products": [{"sku": "SKU-300", "maximumRank": 2}]},
            "clarifications": [],
        },
        "expectedStatus": "NEEDS_CLARIFICATION",
        "expectedIssues": [],
        "expectedMissing": [("pricing", "products[].price.value"), ("pricing", "products[].price.currency")],
    },
    {
        "id": "invented_sku_stripped",
        "request": {
            "input": "Search for drill in store 1. SKU-1 should appear in the top 2.",
            "profile": {"profileId": "p", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k"]},
        },
        "modelOutput": {
            "request": {"query": "drill", "storeId": "1"},
            "expectations": {"products": [
                {"sku": "SKU-1", "maximumRank": 2},
                {"sku": "FAKE-999", "maximumRank": 1},
            ]},
            "clarifications": [],
        },
        "expectedStatus": "NEEDS_CLARIFICATION",
        "expectedIssues": ["INVENTED_SKU"],
        "expectedProductCount": 1,
    },
    {
        "id": "multi_product_with_defaults",
        "request": {
            "input": "Search for paint in store 7. SKU-A should be in the top 2 at USD 10.00. SKU-B should be in the top 5 at USD 20.00.",
            "profile": {"profileId": "p", "version": 1,
                         "enabledEvaluators": ["sku_match", "top_k", "pricing"]},
            "approvedDefaults": {"priceTolerance": "0.05"},
        },
        "modelOutput": {
            "request": {"query": "paint", "storeId": "7"},
            "expectations": {"products": [
                {"sku": "SKU-A", "maximumRank": 2, "price": {"value": "10.00", "currency": "USD"}},
                {"sku": "SKU-B", "maximumRank": 5, "price": {"value": "20.00", "currency": "USD"}},
            ]},
            "clarifications": [],
        },
        "expectedStatus": "READY_FOR_REVIEW",
        "expectedIssues": [],
        "expectedProductCount": 2,
        "expectedDefaultFields": [
            "expectations.products[0].price.tolerance",
            "expectations.products[1].price.tolerance",
        ],
    },
]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_generation_quality(fixture, test_settings):
    svc = TestGenerationService(test_settings, _call_fn=scripted_call(fixture["modelOutput"]))
    req = TestGenerationRequest.model_validate(fixture["request"])
    resp = asyncio.run(svc.generate_async(req))

    assert resp.status.value == fixture["expectedStatus"], (
        f'{fixture["id"]}: got {resp.status.value}, issues={resp.validationIssues}, '
        f'missing={resp.missingExpectations}, clarifications={resp.clarifications}'
    )

    for code in fixture.get("expectedIssues", []):
        assert any(i.code == code for i in resp.validationIssues), \
            f'{fixture["id"]}: expected issue code {code}'

    for evaluator, field in fixture.get("expectedMissing", []):
        assert any(m.evaluator == evaluator and m.field == field
                   for m in resp.missingExpectations), \
            f'{fixture["id"]}: expected missing {evaluator}/{field}'

    if "expectedClarificationCount" in fixture:
        assert len(resp.clarifications) == fixture["expectedClarificationCount"]

    if "expectedProductCount" in fixture:
        assert len(resp.draft.expectations.products) == fixture["expectedProductCount"]

    for field in fixture.get("expectedDefaultFields", []):
        assert any(a.field == field for a in resp.appliedDefaults), \
            f'{fixture["id"]}: expected applied default {field}'
