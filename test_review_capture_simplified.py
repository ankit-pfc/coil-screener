from review_capture import ReviewLearningCaptureV5


def test_unusual_pattern_keeps_technical_translation_optional() -> None:
    capture = ReviewLearningCaptureV5.model_validate(
        {
            "reviewerName": "Amrut",
            "sequencePolicyVersion": 1,
            "baseAssessmentLocked": True,
            "basePath": "exception_territory",
            "failedBaseRules": ["compression"],
            "baseRationale": (
                "Price is not tightening under the repeated top level."
            ),
            "exceptionVerdict": "applies",
            "exceptionRationale": "",
            "commentary": "",
            "ruleProposal": {
                "name": "Rounded recovery",
                "patternKind": "rounded_base",
                "applicability": (
                    "The rounded recovery is clear despite weak compression."
                ),
                "proposedAction": "hold_for_human_review",
                "evidence": [
                    {
                        "id": "left",
                        "sequence": 1,
                        "date": "2020-03-01",
                        "price": 80,
                        "priceField": "low",
                        "role": "support",
                        "label": "Left low",
                    },
                    {
                        "id": "right",
                        "sequence": 2,
                        "date": "2020-09-01",
                        "price": 82,
                        "priceField": "low",
                        "role": "support",
                        "label": "Right low",
                    },
                ],
            },
        }
    )

    assert capture.rule_proposal is not None
    assert capture.rule_proposal.exclusions == ""
    assert capture.rule_proposal.detection_logic == ""
    assert capture.rule_proposal.confirmation == ""
    assert capture.rule_proposal.impacted_stages == []
    assert capture.rule_proposal.validation_plan == ""
