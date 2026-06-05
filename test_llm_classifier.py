"""
Test suite for LLM Classifier (Stages 6, 7, 8)
"""

import json
import os
import sys
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.llm_classifier import (
    CheapLLMClassifier,
    StrongLLMReranker,
    FinalApplicationQueue,
    process_stage_6,
    process_stage_7,
    process_stage_8
)


def test_cheap_llm_classifier_initialization():
    """Test that CheapLLMClassifier can be initialized with different providers."""
    print("Testing CheapLLMClassifier initialization...")
    
    # Test with lm_studio provider (doesn't require API key for local testing)
    try:
        classifier = CheapLLMClassifier(provider="lm_studio")
        assert classifier.provider == "lm_studio"
        assert classifier.client is not None
        print("  ✓ CheapLLMClassifier initialized with lm_studio")
    except Exception as e:
        print(f"  ✗ Failed to initialize CheapLLMClassifier: {e}")
        return False
    
    return True


def test_strong_llm_reranker_initialization():
    """Test that StrongLLMReranker can be initialized."""
    print("Testing StrongLLMReranker initialization...")
    
    # Test with a mock provider or lm_studio
    try:
        # We'll test that the class exists and can be instantiated
        # Note: Actual API calls would require valid credentials
        reranker = StrongLLMReranker(provider="lm_studio")
        assert reranker.provider == "lm_studio"
        print("  ✓ StrongLLMReranker initialized with lm_studio")
    except Exception as e:
        print(f"  ✗ Failed to initialize StrongLLMReranker: {e}")
        return False
    
    return True


def test_final_application_queue_scoring():
    """Test the FinalApplicationQueue scoring logic."""
    print("Testing FinalApplicationQueue scoring...")
    
    queue = FinalApplicationQueue()
    
    # Create a test job with all scoring components
    test_job = {
        'semantic_score': 0.85,  # 85%
        'cheap_llm_result': {'fit_score': 80},
        'strong_llm_result': {'final_score': 90, 'priority': 'high'},
        'days_old': 5,  # Recent job
        'features': {
            'pay': '120k - 150k',
            'work_type': 'Remote'
        }
    }
    
    score = queue.calculate_final_score(test_job)
    
    # Score should be weighted combination
    # semantic: 0.25 * 85 = 21.25
    # cheap: 0.20 * 80 = 16
    # strong: 0.35 * 90 = 31.5
    # recency: 0.05 * 100 = 5 (within 7 days)
    # salary: 0.05 * 0.92 = 4.6 (120k is 92% of way from 50k to 150k)
    # remote: 0.10 * 100 = 10
    # Total: ~88.35
    expected_min = 80
    expected_max = 95
    
    if expected_min <= score <= expected_max:
        print(f"  ✓ Final score {score:.1f} is in expected range [{expected_min}, {expected_max}]")
    else:
        print(f"  ✗ Final score {score:.1f} is outside expected range [{expected_min}, {expected_max}]")
        return False
    
    return True


def test_final_application_queue_priority():
    """Test priority determination logic."""
    print("Testing FinalApplicationQueue priority determination...")
    
    queue = FinalApplicationQueue()
    
    test_cases = [
        (85, "high", "high"),
        (70, "medium", "medium"),
        (55, "low", "low"),
        (40, "skip", "skip"),
        (90, "skip", "skip"),  # skip overrides high score
    ]
    
    all_passed = True
    for score, llm_priority, expected in test_cases:
        result = queue.determine_priority(score, llm_priority)
        if result == expected:
            print(f"  ✓ Score {score} + priority '{llm_priority}' → '{result}'")
        else:
            print(f"  ✗ Score {score} + priority '{llm_priority}' → '{result}' (expected '{expected}')")
            all_passed = False
    
    return all_passed


def test_validate_result():
    """Test result validation logic."""
    print("Testing result validation...")
    
    classifier = CheapLLMClassifier(provider="lm_studio")
    
    # Test valid result
    valid_result = {
        "fit_score": 82,
        "decision": "apply",
        "strengths": ["Strong Python skills"],
        "concerns": ["Limited AWS experience"]
    }
    validated = classifier._validate_result(valid_result)
    
    assert validated["fit_score"] == 82
    assert validated["decision"] == "apply"
    assert len(validated["strengths"]) == 1
    print("  ✓ Valid result passes validation")
    
    # Test result with out-of-range score
    invalid_score = {
        "fit_score": 150,
        "decision": "apply",
        "strengths": [],
        "concerns": []
    }
    validated = classifier._validate_result(invalid_score)
    assert validated["fit_score"] == 100  # Clamped to max
    print("  ✓ Score clamped to 100")
    
    # Test result with invalid decision
    invalid_decision = {
        "fit_score": 80,
        "decision": "definitely",
        "strengths": [],
        "concerns": []
    }
    validated = classifier._validate_result(invalid_decision)
    assert validated["decision"] == "maybe"  # Defaulted
    print("  ✓ Invalid decision defaulted to 'maybe'")
    
    return True


def test_salary_parsing():
    """Test salary parsing in FinalApplicationQueue."""
    print("Testing salary parsing...")
    
    queue = FinalApplicationQueue()
    
    # Test cases: (pay_range, expected_score_range)
    # The logic: extract max number, if >= 150000 return 1.0, if <= 50000 return 0.2,
    # else return 0.2 + (max_salary - 50000) / 100000 * 0.8
    test_cases = [
        ("120k - 150k", 0.8, 1.0),  # 150k max -> 1.0
        ("50k", 0.2, 0.2),  # 50k -> 0.2
        ("200k", 1.0, 1.0),  # 200k -> 1.0
        ("Not specified", 0.5, 0.5),  # Unknown -> 0.5
        ("", 0.5, 0.5),  # Empty -> 0.5
        ("80k - 100k", 0.44, 0.6),  # 100k max -> 0.2 + (100000-50000)/100000*0.8 = 0.6
    ]
    
    all_passed = True
    for pay_range, expected_min, expected_max in test_cases:
        score = queue._parse_salary_score(pay_range)
        if expected_min - 0.1 <= score <= expected_max + 0.1:
            print(f"  ✓ Salary '{pay_range}' → score {score:.2f} (expected {expected_min}-{expected_max})")
        else:
            print(f"  ✗ Salary '{pay_range}' → score {score:.2f} (expected {expected_min}-{expected_max})")
            all_passed = False
    
    return all_passed


def test_mock_stage_6_processing():
    """Test Stage 6 processing with mocked LLM calls."""
    print("Testing Stage 6 processing with mock...")
    
    # Create mock classifier
    mock_classifier = Mock(spec=CheapLLMClassifier)
    mock_classifier.classify.return_value = {
        "fit_score": 75,
        "decision": "apply",
        "strengths": ["Good skills match"],
        "concerns": ["Some gaps"]
    }
    
    # Create test jobs
    test_jobs = [
        {
            'metadata': {'job_id': 1},
            'features': {
                'title': 'Software Engineer',
                'description': 'We are looking for a Python developer...',
                'skills': ['Python', 'Django', 'PostgreSQL']
            }
        },
        {
            'metadata': {'job_id': 2},
            'features': {
                'title': 'Senior Developer',
                'description': 'Looking for experienced developer...',
                'skills': ['Java', 'Spring', 'AWS']
            }
        }
    ]
    
    # Run process_stage_6 with mock
    shortlisted = asyncio.run(process_stage_6(
        jobs=test_jobs,
        classifier=mock_classifier,
        candidate_profile="Experienced software engineer...",
        candidate_skills=["Python", "JavaScript"],
        batch_size=2
    ))
    
    # Verify results
    if len(shortlisted) == 2:
        print(f"  ✓ Stage 6 shortlisted {len(shortlisted)} jobs")
    else:
        print(f"  ✗ Expected 2 shortlisted jobs, got {len(shortlisted)}")
        return False
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("LLM CLASSIFIER TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_cheap_llm_classifier_initialization,
        test_strong_llm_reranker_initialization,
        test_final_application_queue_scoring,
        test_final_application_queue_priority,
        test_validate_result,
        test_salary_parsing,
        test_mock_stage_6_processing,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test {test.__name__} raised exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)