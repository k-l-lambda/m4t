#!/bin/bash
# Comprehensive curl test cases for SeamlessM4T API
# Text-to-Text Translation (T2TT) across multiple language pairs

API_BASE="http://localhost:8000"

echo "=================================="
echo "SeamlessM4T API Test Cases"
echo "=================================="
echo ""

# 1. English to Chinese (Simplified)
echo "1. English → Chinese (Simplified)"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "source_lang": "eng",
    "target_lang": "cmn"
  }' | python3 -m json.tool
echo ""

# 2. Japanese to Chinese (Simplified)
echo "2. Japanese → Chinese (Simplified)"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "こんにちは、今日は良い天気ですね。",
    "source_lang": "jpn",
    "target_lang": "cmn"
  }' | python3 -m json.tool
echo ""

# 3. English to Japanese
echo "3. English → Japanese"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning! What a beautiful day.",
    "source_lang": "eng",
    "target_lang": "jpn"
  }' | python3 -m json.tool
echo ""

# 4. Korean to Chinese
echo "4. Korean → Chinese"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 만나서 반갑습니다.",
    "source_lang": "kor",
    "target_lang": "cmn"
  }' | python3 -m json.tool
echo ""

# 5. French to Japanese
echo "5. French → Japanese"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, comment allez-vous?",
    "source_lang": "fra",
    "target_lang": "jpn"
  }' | python3 -m json.tool
echo ""

# 6. Spanish to Chinese
echo "6. Spanish → Chinese"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hola, ¿cómo estás? Me alegro de conocerte.",
    "source_lang": "spa",
    "target_lang": "cmn"
  }' | python3 -m json.tool
echo ""

# 7. German to English
echo "7. German → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Guten Tag! Wie geht es Ihnen heute?",
    "source_lang": "deu",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 8. Chinese to English
echo "8. Chinese (Simplified) → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，今天天气真好啊。",
    "source_lang": "cmn",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 9. Italian to Spanish
echo "9. Italian → Spanish"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Buongiorno, piacere di conoscerti!",
    "source_lang": "ita",
    "target_lang": "spa"
  }' | python3 -m json.tool
echo ""

# 10. Portuguese to French
echo "10. Portuguese → French"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, como você está hoje?",
    "source_lang": "por",
    "target_lang": "fra"
  }' | python3 -m json.tool
echo ""

# 11. Russian to English
echo "11. Russian → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет! Как дела?",
    "source_lang": "rus",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 12. Arabic to English
echo "12. Arabic → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا، كيف حالك اليوم؟",
    "source_lang": "arb",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 13. Thai to English
echo "13. Thai → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "สวัสดีครับ วันนี้สบายดีไหมครับ?",
    "source_lang": "tha",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 14. Vietnamese to Chinese
echo "14. Vietnamese → Chinese"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Xin chào, hôm nay bạn khỏe không?",
    "source_lang": "vie",
    "target_lang": "cmn"
  }' | python3 -m json.tool
echo ""

# 15. Indonesian to English
echo "15. Indonesian → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Selamat pagi, apa kabar hari ini?",
    "source_lang": "ind",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 16. Turkish to English
echo "16. Turkish → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Merhaba, bugün nasılsınız?",
    "source_lang": "tur",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 17. Hindi to English
echo "17. Hindi → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "नमस्ते, आप कैसे हैं?",
    "source_lang": "hin",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 18. Persian to English
echo "18. Persian → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "سلام، حال شما چطور است؟",
    "source_lang": "pes",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 19. Hebrew to English
echo "19. Hebrew → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום, מה שלומך היום?",
    "source_lang": "heb",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 20. Polish to English
echo "20. Polish → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Cześć, jak się masz dzisiaj?",
    "source_lang": "pol",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 21. Dutch to German
echo "21. Dutch → German"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hallo, hoe gaat het met je vandaag?",
    "source_lang": "nld",
    "target_lang": "deu"
  }' | python3 -m json.tool
echo ""

# 22. Swedish to English
echo "22. Swedish → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hej, hur mår du idag?",
    "source_lang": "swe",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 23. Danish to English
echo "23. Danish → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hej, hvordan har du det i dag?",
    "source_lang": "dan",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 24. Finnish to English
echo "24. Finnish → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hei, mitä kuuluu tänään?",
    "source_lang": "fin",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

# 25. Czech to English
echo "25. Czech → English"
curl -s -X POST "$API_BASE/v1/text-to-text-translation" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ahoj, jak se máš dnes?",
    "source_lang": "ces",
    "target_lang": "eng"
  }' | python3 -m json.tool
echo ""

echo "=================================="
echo "All test cases completed!"
echo "=================================="
