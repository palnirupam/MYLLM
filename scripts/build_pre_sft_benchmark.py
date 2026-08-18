"""
scripts/build_pre_sft_benchmark.py — Generates the separate, immutable pre-SFT evaluation benchmark.
Covers all 12 target languages (English, Bengali, Hindi, Sanskrit, Arabic, Urdu, Russian, Chinese, Japanese, Korean, Tamil, Telugu).
Tasks: QA, definition, sentence completion, translation, summarization, instruction following.
Held out completely from any training data.
"""

import json
from pathlib import Path

BENCHMARK_TASKS = [
    # 1. English
    {"id": "bench_en_01", "language": "en", "task": "qa", "prompt": "What is the primary function of red blood cells in the human body?", "reference": "The primary function of red blood cells is to transport oxygen from the lungs to body tissues and carry carbon dioxide back to the lungs."},
    {"id": "bench_en_02", "language": "en", "task": "definition", "prompt": "Define 'machine learning' in one concise sentence.", "reference": "Machine learning is a subfield of artificial intelligence where algorithms learn patterns from data to make decisions or predictions without explicit rules."},
    {"id": "bench_en_03", "language": "en", "task": "sentence_completion", "prompt": "Complete the sentence logically: The speed of light in a vacuum is approximately", "reference": "300,000 kilometers per second (or 3 x 10^8 meters per second)."},
    {"id": "bench_en_04", "language": "en", "task": "translation", "prompt": "Translate to Bengali: 'Truth always triumphs over falsehood.'", "reference": "সত্য সর্বদা মিথ্যার ওপর জয়লাভ করে।"},
    {"id": "bench_en_05", "language": "en", "task": "summarization", "prompt": "Summarize why gravity is important for planetary orbits in one sentence.", "reference": "Gravity provides the necessary centripetal force that keeps planets in stable elliptical orbits around the Sun."},
    {"id": "bench_en_06", "language": "en", "task": "instruction_following", "prompt": "List three common operating systems as bullet points.", "reference": "- Linux\n- Windows\n- macOS"},

    # 2. Bengali (বাংলা)
    {"id": "bench_bn_01", "language": "bn", "task": "qa", "prompt": "বাংলাদেশের জাতীয় কবির নাম কী?", "reference": "বাংলাদেশের জাতীয় কবির নাম কাজী নজরুল ইসলাম।"},
    {"id": "bench_bn_02", "language": "bn", "task": "definition", "prompt": "সালোকসংশ্লেষণ কাকে বলে? সংক্ষেপে লিখুন।", "reference": "যে জৈব-রাসায়নিক প্রক্রিয়ায় সবুজ উদ্ভিদ সূর্যালোকের উপস্থিতিতে পানি ও কার্বন ডাই-অক্সাইড ব্যবহার করে গ্লুকোজ ও অক্সিজেন তৈরি করে, তাকে সালোকসংশ্লেষণ বলে।"},
    {"id": "bench_bn_03", "language": "bn", "task": "sentence_completion", "prompt": "বাক্যটি সম্পূর্ণ করুন: পৃথিবী সূর্যের চারদিকে ঘোরে এবং চাঁদ ঘোরে", "reference": "পৃথিবীর চারদিকে।"},
    {"id": "bench_bn_04", "language": "bn", "task": "translation", "prompt": "নিচের বাক্যটি ইংরেজিতে অনুবাদ করুন: 'জ্ঞানই শক্তি।'", "reference": "Knowledge is power."},
    {"id": "bench_bn_05", "language": "bn", "task": "summarization", "prompt": "গাছপালা কেন পরিবেশের জন্য অপরিহার্য তা এক বাক্যে লিখুন।", "reference": "গাছপালা অক্সিজেন সরবরাহ করে এবং বায়ুমণ্ডলের কার্বন ডাই-অক্সাইড শোষণ করে পরিবেশের ভারসাম্য বজায় রাখে।"},
    {"id": "bench_bn_06", "language": "bn", "task": "instruction_following", "prompt": "কম্পিউটারের তিনটি ইনপুট ডিভাইসের নাম বুলেটে লিখুন।", "reference": "- কীবোর্ড\n- মাউস\n- স্ক্যানার"},

    # 3. Hindi (हिंदी)
    {"id": "bench_hi_01", "language": "hi", "task": "qa", "prompt": "भारत की राजधानी का क्या नाम है?", "reference": "भारत की राजधानी नई दिल्ली है।"},
    {"id": "bench_hi_02", "language": "hi", "task": "definition", "prompt": "गुरुत्वाकर्षण की परिभाषा संक्षेप में लिखिए।", "reference": "गुरुत्वाकर्षण वह प्राकृतिक आकर्षण बल है जो द्रव्यमान वाली किन्हीं दो वस्तुओं के बीच कार्य करता है।"},
    {"id": "bench_hi_03", "language": "hi", "task": "sentence_completion", "prompt": "वाक्य पूरा कीजिए: जल ही जीवन है क्योंकि इसके बिना", "reference": "किसी भी जीव का अस्तित्व संभव नहीं है।"},
    {"id": "bench_hi_04", "language": "hi", "task": "translation", "prompt": "अंग्रेजी में अनुवाद कीजिए: 'सत्य की हमेशा जीत होती है।'", "reference": "Truth always triumphs."},
    {"id": "bench_hi_05", "language": "hi", "task": "summarization", "prompt": "सौर ऊर्जा का मुख्य लाभ एक वाक्य में लिखिए।", "reference": "सौर ऊर्जा एक अक्षय और प्रदूषण-मुक्त ऊर्जा स्रोत है जो सूर्य के प्रकाश से प्राप्त होती है।"},
    {"id": "bench_hi_06", "language": "hi", "task": "instruction_following", "prompt": "गणित के तीन बुनियादी संचालनों (operations) को बुलेट में लिखिए।", "reference": "- जोड़ (Addition)\n- घटाव (Subtraction)\n- गुणा (Multiplication)"},

    # 4. Sanskrit (संस्कृतम्)
    {"id": "bench_sa_01", "language": "sa", "task": "qa", "prompt": "विद्या किं ददाति?", "reference": "विद्या विनयं ददाति।"},
    {"id": "bench_sa_02", "language": "sa", "task": "sentence_completion", "prompt": "श्लोकांशं पूरयत: सत्यमेव जयते", "reference": "नानृतम्।"},
    {"id": "bench_sa_03", "language": "sa", "task": "translation", "prompt": "आङ्ग्लभाषायाम् अनुवादं कुरुत: 'वसुधैव कुटुम्बकम्'", "reference": "The whole world is one family."},

    # 5. Arabic (العربية)
    {"id": "bench_ar_01", "language": "ar", "task": "qa", "prompt": "ما هو أكبر كوكب في المجموعة الشمسية؟", "reference": "كوكب المشتري هو أكبر كواكب المجموعة الشمسية."},
    {"id": "bench_ar_02", "language": "ar", "task": "definition", "prompt": "عرف الذكاء الاصطناعي في جملة واحدة.", "reference": "الذكاء الاصطناعي هو علم إنشاء أنظمة وبرامج حاسوبية قادرة على محاكاة التفكير البشري واتخاذ القرارات."},
    {"id": "bench_ar_03", "language": "ar", "task": "translation", "prompt": "ترجم إلى الإنجليزية: 'العلم نور'", "reference": "Knowledge is light."},

    # 6. Urdu (اردو)
    {"id": "bench_ur_01", "language": "ur", "task": "qa", "prompt": "پاکستان کا قومی شاعر کون ہے؟", "reference": "علامہ محمد اقبال پاکستان کے قومی شاعر ہیں۔"},
    {"id": "bench_ur_02", "language": "ur", "task": "definition", "prompt": "کمپیوٹر نیٹ ورک کی سادہ تعریف کیا ہے؟", "reference": "کمپیوٹر نیٹ ورک دو یا زیادہ کمپیوٹرز کا ایسا نظام ہے جو معلومات اور وسائل کے باہمی تبادلے کے لیے جڑے ہوتے ہیں۔"},
    {"id": "bench_ur_03", "language": "ur", "task": "translation", "prompt": "انگریزی میں ترجمہ کریں: 'محنت میں عظمت ہے۔'", "reference": "There is dignity in labor."},

    # 7. Russian (Русский)
    {"id": "bench_ru_01", "language": "ru", "task": "qa", "prompt": "Какая планета находится ближе всего к Солнцу?", "reference": "Меркурий является ближайшей планетой к Солнцу."},
    {"id": "bench_ru_02", "language": "ru", "task": "definition", "prompt": "Что такое алгоритм простыми словами?", "reference": "Алгоритм — это четкая последовательность шагов и правил для решения конкретной задачи."},
    {"id": "bench_ru_03", "language": "ru", "task": "translation", "prompt": "Переведите на английский язык: 'Мир и дружба'", "reference": "Peace and friendship."},

    # 8. Chinese (中文) - Diagnostic Held-out
    {"id": "bench_zh_01", "language": "zh", "task": "qa", "prompt": "太阳系中最大的行星是哪一颗？", "reference": "木星是太阳系中体积和质量最大的行星。"},
    {"id": "bench_zh_02", "language": "zh", "task": "definition", "prompt": "简述什么是计算机算法：", "reference": "算法是解决特定问题的一组清晰、有限且明确的计算步骤。"},
    {"id": "bench_zh_03", "language": "zh", "task": "translation", "prompt": "将这句话翻译成英文：'千里之行，始于足下。'", "reference": "A journey of a thousand miles begins with a single step."},

    # 9. Japanese (日本語) - Diagnostic Held-out
    {"id": "bench_ja_01", "language": "ja", "task": "qa", "prompt": "光の三原色は何ですか？", "reference": "光の三原色は赤（Red）、緑（Green）、青（Blue）です。"},
    {"id": "bench_ja_02", "language": "ja", "task": "definition", "prompt": "人工知能の定義を一文で述べてください：", "reference": "人工知能とは、人間の学習、推論、判断などの知的能力をコンピュータ上で模倣する技術です。"},
    {"id": "bench_ja_03", "language": "ja", "task": "translation", "prompt": "英語に翻訳してください：'継続は力なり。'", "reference": "Continuity is power (or Perseverance pays off)."},

    # 10. Korean (한국어) - Diagnostic Held-out
    {"id": "bench_ko_01", "language": "ko", "task": "qa", "prompt": "지구에서 가장 가까운 별은 무엇인가요?", "reference": "지구에서 가장 가까운 별은 태양입니다."},
    {"id": "bench_ko_02", "language": "ko", "task": "definition", "prompt": "운영체제(OS)의 역할을 한 문장으로 설명하세요.", "reference": "운영체제는 컴퓨터의 하드웨어 자원을 관리하고 사용자와 응용 프로그램에 인터페이스를 제공하는 소프트웨어입니다."},
    {"id": "bench_ko_03", "language": "ko", "task": "translation", "prompt": "영어로 번역하세요: '시작이 반이다.'", "reference": "Well begun is half done."},

    # 11. Tamil (தமிழ்)
    {"id": "bench_ta_01", "language": "ta", "task": "qa", "prompt": "சூரியனுக்கு மிக அருகில் உள்ள கோள் எது?", "reference": "சூரியனுக்கு மிக அருகில் உள்ள கோள் புதன் (Mercury) ஆகும்."},
    {"id": "bench_ta_02", "language": "ta", "task": "definition", "prompt": "ஒளிச்சேர்க்கை என்றால் என்ன? சுருக்கமாக கூறுக.", "reference": "பச்சைத் தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தி கார்பன் டை ஆக்சைடு மற்றும் நீரிலிருந்து உணவு தயாரிக்கும் செயல்முறை ஒளிச்சேர்க்கை எனப்படும்."},
    {"id": "bench_ta_03", "language": "ta", "task": "translation", "prompt": "ஆங்கிலத்தில் மொழிபெயர்க்கவும்: 'வாய்மையே வெல்லும்'", "reference": "Truth alone triumphs."},

    # 12. Telugu (తెలుగు) - Diagnostic Held-out
    {"id": "bench_te_01", "language": "te", "task": "qa", "prompt": "సౌర మండలంలో అతి పెద్ద గ్రహం ఏది?", "reference": "సౌర మండలంలో బృహస్పతి (Jupiter) అతి పెద్ద గ్రహం."},
    {"id": "bench_te_02", "language": "te", "task": "definition", "prompt": "కంప్యూటర్ నెట్‌వర్క్ అంటే ఏమిటి?", "reference": "సమాచారాన్ని మరియు వనరులను పరస్పరం పంచుకోవడానికి రెండు లేదా అంతకంటే ఎక్కువ కంప్యూటర్లను అనుసంధానించే వ్యవస్థను కంప్యూటర్ నెట్‌వర్క్ అంటారు."},
    {"id": "bench_te_03", "language": "te", "task": "translation", "prompt": "ఆంగ్లంలోకి అనువదించండి: 'సత్యమే జయిస్తుంది'", "reference": "Truth alone triumphs."}
]


def main():
    out_dir = Path("dhruva-v1-assets/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_pre_sft.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for item in BENCHMARK_TASKS:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Immutable Pre-SFT Benchmark successfully built: {out_file}")
    print(f"[+] Total benchmark test cases: {len(BENCHMARK_TASKS)} covering all 12 target languages.")


if __name__ == "__main__":
    main()
