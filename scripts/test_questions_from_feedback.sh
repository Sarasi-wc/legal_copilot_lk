#!/bin/bash
# Test questions from professor assessment feedback
# Tests all 12 questions mentioned in the feedback

API_URL="http://localhost:8000/answer"

echo "=========================================="
echo "Testing Questions from Professor Assessment"
echo "=========================================="
echo ""

# Question 1
echo "Question 1: What does Article 9 of the Constitution say about Buddhism?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does Article 9 of the Constitution say about Buddhism?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 2
echo "Question 2: What rights are guaranteed under Article 10 of the Constitution?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What rights are guaranteed under Article 10 of the Constitution?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 3
echo "Question 3: What does Article 14(1)(e) of the Constitution protect?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does Article 14(1)(e) of the Constitution protect?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 4
echo "Question 4: How does Article 9 balance the foremost place given to Buddhism with the rights of other religions?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does Article 9 balance the foremost place given to Buddhism with the rights of other religions?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 5
echo "Question 5: What is the difference between the Constitution of Sri Lanka and the Civil Procedure Code?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the difference between the Constitution of Sri Lanka and the Civil Procedure Code?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 6
echo "Question 6: Can the State restrict religious freedom under Article 14(1)(e)? Under what conditions?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can the State restrict religious freedom under Article 14(1)(e)? Under what conditions?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 7
echo "Question 7: Which legal document contains Article 9?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which legal document contains Article 9?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 8
echo "Question 8: Under which articles are religious freedoms guaranteed alongside Article 9?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Under which articles are religious freedoms guaranteed alongside Article 9?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 9
echo "Question 9: What is the Civil Procedure Code primarily concerned with?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Civil Procedure Code primarily concerned with?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 10
echo "Question 10: Does Article 9 make Buddhism the state religion of Sri Lanka?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does Article 9 make Buddhism the state religion of Sri Lanka?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 11
echo "Question 11: What obligations does Article 9 place on private citizens?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What obligations does Article 9 place on private citizens?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "---"
echo ""

# Question 12
echo "Question 12: Does the Constitution define 'Buddha Sasana'?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does the Constitution define Buddha Sasana?",
    "retrieval_method": "hybrid_rerank",
    "top_k": 5,
    "include_verification": true
}' | python3 -m json.tool | head -50
echo ""
echo "=========================================="
echo "Testing Complete"
echo "=========================================="
