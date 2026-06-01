#!/bin/bash
# Test all questions from Professor Assessment
# After index rebuild - comprehensive testing

API_URL="http://localhost:8000/answer"

echo "Testing All Questions from Professor Assessment"
echo "=============================================="
echo ""

# Q1: What does Article 9 say about Buddhism?
echo "Q1: What does Article 9 say about Buddhism?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Article 9 say about Buddhism?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q1.json
echo "✅ Saved to /tmp/q1.json"
echo ""

# Q2: What rights are guaranteed under Article 10?
echo "Q2: What rights are guaranteed under Article 10?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What rights are guaranteed under Article 10?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q2.json
echo "✅ Saved to /tmp/q2.json"
echo ""

# Q3: What does Article 14(1)(e) protect?
echo "Q3: What does Article 14(1)(e) protect?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Article 14(1)(e) protect?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q3.json
echo "✅ Saved to /tmp/q3.json"
echo ""

# Q4: How does Article 9 balance Buddhism vs other religions?
echo "Q4: How does Article 9 balance Buddhism vs other religions?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Article 9 balance Buddhism vs other religions?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q4.json
echo "✅ Saved to /tmp/q4.json"
echo ""

# Q5: Constitution vs Civil Procedure Code difference?
echo "Q5: What is the difference between the Constitution and Civil Procedure Code?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between the Constitution of Sri Lanka and the Civil Procedure Code?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q5.json
echo "✅ Saved to /tmp/q5.json"
echo ""

# Q6: Can the State restrict Article 14(1)(e)? Under what conditions?
echo "Q6: Can the State restrict Article 14(1)(e)? Under what conditions?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can the State restrict Article 14(1)(e)? Under what conditions?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q6.json
echo "✅ Saved to /tmp/q6.json"
echo ""

# Q7: Which legal document contains Article 9?
echo "Q7: Which legal document contains Article 9?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "Which legal document contains Article 9?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q7.json
echo "✅ Saved to /tmp/q7.json"
echo ""

# Q8: Which articles guarantee religious freedom alongside Article 9?
echo "Q8: Which articles guarantee religious freedom alongside Article 9?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "Which articles guarantee religious freedom alongside Article 9?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q8.json
echo "✅ Saved to /tmp/q8.json"
echo ""

# Q9: What is the Civil Procedure Code primarily concerned with?
echo "Q9: What is the Civil Procedure Code primarily concerned with?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Civil Procedure Code primarily concerned with?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q9.json
echo "✅ Saved to /tmp/q9.json"
echo ""

# Q10: Does Article 9 make Buddhism the state religion?
echo "Q10: Does Article 9 make Buddhism the state religion of Sri Lanka?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "Does Article 9 make Buddhism the state religion of Sri Lanka?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q10.json
echo "✅ Saved to /tmp/q10.json"
echo ""

# Q11: What obligations does Article 9 place on private citizens?
echo "Q11: What obligations does Article 9 place on private citizens?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "What obligations does Article 9 place on private citizens?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q11.json
echo "✅ Saved to /tmp/q11.json"
echo ""

# Q12: Does the Constitution define Buddha Sasana?
echo "Q12: Does the Constitution define Buddha Sasana?"
curl -s -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d '{"question": "Does the Constitution define Buddha Sasana?", "retrieval_method": "hybrid_rerank", "top_k": 5, "include_verification": true}' \
  | python3 -m json.tool > /tmp/q12.json
echo "✅ Saved to /tmp/q12.json"
echo ""

echo "=============================================="
echo "All questions tested. Results saved to /tmp/q*.json"
echo ""
