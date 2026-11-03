import streamlit as st
import requests
import urllib.parse
from PIL import Image
import numpy as np
import platform
import os
import time
from paddleocr import PaddleOCR
import cv2
import re
import traceback

st.set_page_config(
    page_title="의약품 성분 충돌 감지 시스템",
    page_icon="💊",
    layout="wide"
)

# PaddleOCR 엔진 생성
@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    """PaddleOCR 엔진을 한 번 생성해 재사용합니다."""
    return PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

# 의료 면책 문구
MEDICAL_DISCLAIMER = (
    "⚠️ **의료 면책 안내**\n\n"
    "본 결과는 AI가 공공데이터(식약처 DUR·e약은요)를 바탕으로 생성한 **참고용 정보**이며, "
    "의학적 진단·처방·복약 지시를 대체하지 않습니다. AI는 오류나 누락이 있을 수 있습니다.\n\n"
    "**복용 여부, 병용, 용량 변경 등 모든 의학적 판단은 반드시 의사 또는 약사 등 전문가에게 "
    "진단·상담을 받은 후 결정하세요.** "
    "발진, 호흡곤란, 출혈, 심한 복통 등 이상 증상이 있으면 즉시 진료를 받거나 119에 연락하세요."
)

# ==========================================
# 경량화 모델 프로파일 (GGUF 및 규칙 기반만 유지)
# ==========================================
MODEL_PROFILES = {
    "LLM 끄기 · 규칙+공공데이터만 (가장 가벼움)": {
        "backend": "none",
        "short": "규칙 기반 (LLM 미사용)",
        "max_new_tokens": 0,
    },
}
DEFAULT_PROFILE_KEY = list(MODEL_PROFILES.keys())[0]


@st.cache_resource(show_spinner=False)
def get_gguf_llm(repo_id: str, filename: str):
    """4bit 양자화 GGUF 모델을 llama.cpp로 적재합니다."""
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=0,  # CPU 환경 중심 구동 시 0, GPU 활용 시 -1 변경 가능
        verbose=False,
    )


def _generate_with_profile(profile: dict, messages: list) -> str:
    """GGUF 백엔드를 통해 답변을 생성합니다."""
    max_new = profile["max_new_tokens"]
    llm = get_gguf_llm(profile["repo_id"], profile["filename"])
    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_new,
        temperature=0.0,
        repeat_penalty=1.05,
    )
    return out["choices"][0]["message"]["content"].strip()


# 약리 계열 사전
INGREDIENT_CLASS = {
    "아세트아미노펜": ("해열진통제 (NSAIDs 아님)", "해열진통제(APAP)"),
    "아세틸살리실산": ("NSAIDs(비스테로이드성 소염진통제) · 항혈소판 작용", "NSAIDs"),
    "이부프로펜": ("NSAIDs(비스테로이드성 소염진통제)", "NSAIDs"),
    "나프록센": ("NSAIDs(비스테로이드성 소염진통제)", "NSAIDs"),
    "이소트레티노인": ("레티노이드 (여드름 치료제)", "레티노이드"),
    "세티리진": ("항히스타민제", "항히스타민제"),
    "이소프로필안티피린": ("피린계 해열진통제", "피린계"),
    "카페인": ("중추신경 흥분제", "카페인"),
    "비타민B군": ("비타민", "비타민B"),
}


def build_class_facts(base_ingredients, second_drug=""):
    second_ingredients = []
    if second_drug and second_drug != "없음":
        try:
            second_ingredients = list(get_drug_ingredients(second_drug) or [])
        except Exception:
            second_ingredients = []
        for ing in extract_ingredients_from_text(second_drug):
            if ing not in second_ingredients:
                second_ingredients.append(ing)

    lines = []
    for label, ings in (("기준 약품", base_ingredients or []), ("함께 복용 약품", second_ingredients)):
        for ing in ings:
            cls = INGREDIENT_CLASS.get(ing)
            lines.append(f"- {label} 성분 {ing}: " + (cls[0] if cls else "계열 정보 없음(확인 불가)"))

    base_groups = {}
    for ing in base_ingredients or []:
        if ing in INGREDIENT_CLASS:
            base_groups.setdefault(INGREDIENT_CLASS[ing][1], []).append(ing)
    second_groups = {}
    for ing in second_ingredients:
        if ing in INGREDIENT_CLASS:
            second_groups.setdefault(INGREDIENT_CLASS[ing][1], []).append(ing)

    same = [g for g in base_groups if g in second_groups]
    if not second_drug or second_drug == "없음":
        lines.append("- 같은 계열 중복: 비교 약품 없음")
    elif same:
        for g in same:
            lines.append(
                f"- 같은 계열 중복: 있음 ({g}: {', '.join(base_groups[g])} / {', '.join(second_groups[g])})"
            )
    else:
        lines.append("- 같은 계열 중복: 없음 (사전 기준, 사전에 없는 성분은 확인 불가)")

    return "\n".join(lines) if lines else "- 계열 정보 없음(확인 불가)"


def build_rule_based_report(base_drug: str, second_drug: str, base_ingredients: list, dur_msg: str) -> str:
    ingredients = ", ".join(base_ingredients) if base_ingredients else "사전 미등록"
    return (
        "### 📋 규칙 기반 복약 안내 (LLM 미사용)\n\n"
        f"**1. 기준 약품 / 주성분**: {base_drug} / {ingredients}\n\n"
        f"**2. 함께 복용하려는 약품**: {second_drug if second_drug and second_drug != '없음' else '미입력'}\n\n"
        f"**3. 식약처 DUR 조회 결과**: {dur_msg}\n\n"
        "**4. 안내**: 동일 성분 또는 같은 계열(예: 소염진통제)의 약을 함께 복용하면 부작용 위험이 커질 수 있습니다. "
        "정확한 병용 가능 여부는 약사에게 확인하세요."
    )


def analyze_with_local_llm(
    ocr_text: str,
    base_drug: str,
    second_drug: str,
    base_ingredients: list,
    dur_msg: str,
    profile_key: str = DEFAULT_PROFILE_KEY
) -> str:
    profile = MODEL_PROFILES.get(profile_key, MODEL_PROFILES[DEFAULT_PROFILE_KEY])
    class_text = build_class_facts(base_ingredients, second_drug)
    dur_msg_plain = dur_msg.split(" (병용금기 여부는")[0]

    if profile["backend"] == "none":
        return (
            build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
            + "\n\n**약리 계열 확인 (사전 기준)**\n" + class_text
            + "\n\n---\n" + MEDICAL_DISCLAIMER
        )

    system_prompt = (
        "당신은 복약 정보를 쉽게 풀어 설명하는 보조 AI입니다. 의사나 약사를 대체하지 않습니다. "
        "제공된 [OCR 추출 텍스트], [복용 약물 정보], [식약처 DUR 공식 조회 결과]만을 근거로 "
        "아래 3가지 항목으로 간결하게 답변하세요.\n\n"
        "1. 🔍 [OCR 약품 및 주성분 판독]: OCR 오탈자를 문맥으로 보정하여 식별된 약품명과 주성분을 정리하세요.\n"
        "2. ⚠️ [성분 중복 및 약리학적 계열 충돌 분석]: 성분이 직접 중복되는지, 동일 약리 계열(예: NSAIDs)인지 분석하세요.\n"
        "3. 💡 [환자 맞춤 3줄 복약 지도]: 주의사항을 일상 용어로 3줄 요약하세요.\n\n"
        "반드시 지킬 규칙:\n"
        "- 질병 진단, 처방, 용량 변경 지시는 절대 하지 마세요.\n"
        "- 제공된 근거에 없는 내용은 지어내지 말고 '확인 불가'라고 쓰세요.\n"
        "- 약리 계열(예: NSAIDs 여부)은 [약리 계열 근거]에 적힌 내용만 사용하고, 기억으로 추측하지 마세요.\n"
        "- 확실하지 않으면 안전한 쪽(복용 전 약사 상담)으로 안내하세요.\n"
        "- 마지막 줄에 '정확한 복용 여부는 반드시 의사 또는 약사와 상담하세요.'를 쓰세요."
    )

    user_prompt = (
        f"- [이미지 OCR 추출 텍스트]: {ocr_text if ocr_text else '업로드된 이미지 없음'}\n"
        f"- [기준 약품명]: {base_drug} (등록 성분: {', '.join(base_ingredients) if base_ingredients else '사전 미등록'})\n"
        f"- [함께 복용하려는 약품명]: {second_drug if second_drug and second_drug != '없음' else '미입력'}\n"
        f"- [식약처 DUR 병용금기 조회 결과]: {dur_msg}\n"
        f"- [약리 계열 근거(사전 정의)]:\n{class_text}\n\n"
        "위 근거 데이터를 바탕으로 복약 분석 리포트를 작성해주세요."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    t_gen = time.time()
    try:
        report = _generate_with_profile(profile, messages)
    except Exception as e:
        print(f"[LLM 오류 -> 규칙 기반 대체] {e}", flush=True)
        report = (
            f"⚠️ LLM 실행에 실패하여 규칙 기반 안내로 대체합니다. (원인: {e})\n\n"
            + build_rule_based_report(base_drug, second_drug, base_ingredients, dur_msg_plain)
        )
    print(f"[LLM 추론] 완료 ({time.time() - t_gen:.1f}초, {profile['short']})", flush=True)

    return (
        report
        + "\n\n**약리 계열 확인 (사전 기준 · 자동 검증)**\n" + class_text
        + "\n\n---\n" + MEDICAL_DISCLAIMER
    )


# 테스트용 의약품 정보
DRUG_INFO = {
    "타이레놀": {
        "ingredients": ["아세트아미노펜"],
        "description": "해열 및 진통 목적으로 사용되는 의약품입니다.",
        "warning": "아세트아미노펜을 포함한 다른 의약품과 성분 중복 여부를 확인해야 합니다."
    },
    "아스피린": {
        "ingredients": ["아세틸살리실산"],
        "description": "해열·진통 등에 사용되는 의약품입니다.",
        "warning": "다른 의약품과 함께 사용할 경우 상호작용 및 주의사항을 확인해야 합니다."
    },
    "이지엔6": {
        "ingredients": ["제품별 주성분 확인 필요"],
        "description": "제품 종류에 따라 주성분이 다른 진통 관련 의약품입니다.",
        "warning": "정확한 제품명을 확인하여 주성분을 확인해야 합니다."
    },
    "아로나민": {
        "ingredients": ["비타민B군"],
        "description": "비타민 보충을 목적으로 사용하는 제품입니다.",
        "warning": "다른 영양제와 함께 사용하는 경우 동일 영양성분의 중복 여부를 확인해야 합니다."
    },
    "이부프로펜": {
        "ingredients": ["이부프로펜"],
        "description": "해열·진통·소염 목적으로 사용되는 의약품입니다.",
        "warning": "다른 소염진통제와 함께 사용하는 경우 성분 및 상호작용을 확인해야 합니다."
    },
    "나프록센": {
        "ingredients": ["나프록센"],
        "description": "해열·진통·소염 목적으로 사용되는 의약품입니다.",
        "warning": "다른 소염진통제와 함께 사용하는 경우 상호작용 여부를 확인해야 합니다."
    },
    "이소티논": {
        "ingredients": ["이소트레티노인"],
        "description": "여드름 치료에 사용되는 의약품입니다.",
        "warning": "다른 의약품과 함께 사용하는 경우 상호작용 및 주의사항을 확인해야 합니다."
    },
    "지르텍": {
        "ingredients": ["세티리진"],
        "description": "알레르기 증상 완화에 사용하는 항히스타민제입니다.",
        "warning": "다른 항히스타민제와 함께 사용하는 경우 주의사항을 확인해야 합니다."
    },
    "게보린": {
        "ingredients": ["아세트아미노펜", "이소프로필안티피린", "카페인"],
        "description": "해열·진통 목적으로 사용하는 복합 성분 의약품입니다.",
        "warning": "제품의 정확한 성분표를 확인하고 다른 진통제와의 성분 중복 여부를 확인해야 합니다."
    },
    "판콜": {
        "ingredients": ["제품별 복합 성분"],
        "description": "감기 증상 완화를 목적으로 사용하는 복합 성분 의약품입니다.",
        "warning": "제품 종류에 따라 성분이 다를 수 있으므로 정확한 제품명을 확인해야 합니다."
    }
}

DRUG_LIST = list(DRUG_INFO.keys())


# OCR 성분 사전
INGREDIENT_DATABASE = {
    "아세트아미노펜": ["아세트아미노펜", "acetaminophen", "paracetamol"],
    "아세틸살리실산": ["아세틸살리실산", "아스피린", "acetylsalicylic acid", "aspirin"],
    "이부프로펜": ["이부프로펜", "ibuprofen"],
    "나프록센": ["나프록센", "naproxen"],
    "이소트레티노인": ["이소트레티노인", "isotretinoin", "이소티논"],
    "세티리진": ["세티리진", "cetirizine"],
    "이소프로필안티피린": ["이소프로필안티피린", "isopropylantipyrine"],
    "카페인": ["카페인", "caffeine"],
    "비타민B군": ["비타민b", "vitamin b", "비타민 b"]
}


def is_valid_api_key(api_key):
    if not api_key:
        return False
    if api_key.strip() == "":
        return False
    return True


# PaddleOCR: 이미지 전처리 및 글자 인식
def extract_text_from_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        height, width = image_cv.shape[:2]
        longest_side = max(height, width)
        if longest_side < 1600:
            scale = min(3.0, 1600 / longest_side)
            image_cv = cv2.resize(
                image_cv, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(gray)
        sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        )
        sharpened = cv2.filter2D(clahe, -1, sharpen_kernel)
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        ocr_engine = get_ocr_engine()
        candidates = []

        for processed in (image_cv, enhanced_bgr):
            results = ocr_engine.predict(processed)

            for result in results:
                result_data = getattr(result, "json", None)
                if callable(result_data):
                    result_data = result_data()

                if not isinstance(result_data, dict):
                    continue

                result_data = result_data.get("res", result_data)
                texts = result_data.get("rec_texts", []) or []
                scores = result_data.get("rec_scores", []) or []

                for index, recognized_text in enumerate(texts):
                    recognized_text = str(recognized_text).strip()
                    if not recognized_text:
                        continue

                    if index < len(scores):
                        try:
                            if float(scores[index]) < 0.25:
                                continue
                        except (TypeError, ValueError):
                            pass

                    candidates.append(recognized_text)

        unique_lines = []
        for line in candidates:
            if line not in unique_lines:
                unique_lines.append(line)

        if not unique_lines:
            return (
                "OCR_ERROR: PaddleOCR가 글자를 인식하지 못했습니다. "
                "사진을 밝고 선명하게 촬영한 뒤 다시 시도하세요."
            )

        matching_lines = []
        for line in unique_lines:
            try:
                if detect_drug_name_from_text(line):
                    matching_lines.append(line)
            except Exception:
                pass

        if matching_lines:
            unique_lines.sort(
                key=lambda line: (
                    line in matching_lines,
                ),
                reverse=True
            )

        return "\n".join(unique_lines)

    except Exception as e:
        return (
            "OCR_ERROR: PaddleOCR 실행 중 오류가 발생했습니다. "
            f"상세 오류: {e}"
        )


DRUG_ALIASES = {
    "타이레놀": ["타이레놀", "타이레놀정", "타이레놀500mg", "타이레놀 500mg", "TYLENOL"],
    "아스피린": ["아스피린", "ASPIRIN"],
    "이지엔6": ["이지엔6", "이지엔", "EZN6"],
    "아로나민": ["아로나민", "ARONAMIN"],
    "이부프로펜": ["이부프로펜", "IBUPROFEN"],
    "나프록센": ["나프록센", "NAPROXEN"],
    "이소티논": ["이소티논", "ISOTRETINOIN"],
    "지르텍": ["지르텍", "ZYRTEC"],
    "게보린": ["게보린", "GEVORIN"],
    "판콜": ["판콜", "PANCOLD"]
}


def normalize_drug_text(text):
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def detect_drug_name_from_text(text):
    detected_drugs = []
    normalized_text = normalize_drug_text(text)

    for drug_name in DRUG_LIST:
        aliases = DRUG_ALIASES.get(drug_name, [drug_name])
        for alias in aliases:
            normalized_alias = normalize_drug_text(alias)
            if normalized_alias and normalized_alias in normalized_text:
                if drug_name not in detected_drugs:
                    detected_drugs.append(drug_name)
                break

    return detected_drugs


def extract_ingredients_from_text(text):
    found_ingredients = []
    text_lower = text.lower()

    for ingredient, keywords in INGREDIENT_DATABASE.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                if ingredient not in found_ingredients:
                    found_ingredients.append(ingredient)
                break

    return found_ingredients


def get_drug_ingredients(drug_name):
    if drug_name not in DRUG_INFO:
        return []
    return DRUG_INFO[drug_name]["ingredients"]


def compare_ingredients(drug_a, drug_b):
    ingredients_a = get_drug_ingredients(drug_a)
    ingredients_b = get_drug_ingredients(drug_b)

    if not ingredients_a or not ingredients_b:
        return []

    conflicts = []
    for ingredient in ingredients_a:
        if ingredient in ingredients_b:
            if ingredient not in conflicts:
                conflicts.append(ingredient)

    return conflicts


def compare_ocr_with_drug(ocr_ingredients, drug_name):
    drug_ingredients = get_drug_ingredients(drug_name)
    conflicts = []

    for ingredient in ocr_ingredients:
        if ingredient in drug_ingredients:
            if ingredient not in conflicts:
                conflicts.append(ingredient)

    return conflicts


_EASY_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
_EASY_CACHE = {}


def fetch_drugs_from_api(keyword, api_key, num_rows=4):
    if not is_valid_api_key(api_key):
        return [], "API 인증이 필요합니다. (사이드바 승인 버튼을 눌러주세요)"

    cache_key = (keyword, num_rows)
    cached = _EASY_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < 3600:
        return cached[1], None

    query = urllib.parse.urlencode(
        {"itemName": keyword, "type": "json", "numOfRows": num_rows},
        quote_via=urllib.parse.quote,
    )

    try:
        response = None
        for key_part in _dur_key_variants(api_key):
            response = requests.get(
                f"{_EASY_URL}?serviceKey={key_part}&{query}", timeout=10
            )
            if response.status_code == 200:
                break
            if "SERVICE_KEY_IS_NOT_REGISTERED" not in (response.text or ""):
                break

        if response.status_code != 200:
            return [], _dur_http_error(response, "e약은요")

        try:
            data = response.json()
        except ValueError:
            return [], "e약은요 응답을 해석하지 못했습니다."

        root = data.get("response", data) if isinstance(data, dict) else {}
        header = root.get("header", {}) or {}
        code = str(header.get("resultCode", "00"))
        if code not in ("00", "0"):
            return [], f"e약은요 API 오류 (resultCode {code})"

        items = (root.get("body", {}) or {}).get("items", [])
        if isinstance(items, dict):
            items = items.get("item", items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        extracted_drugs = []
        for item in items:
            extracted_drugs.append({
                "itemSeq": item.get("itemSeq", ""),
                "itemName": item.get("itemName", "제품명 미확인"),
                "itemImage": item.get("itemImage") or "https://via.placeholder.com/200x150?text=No+Image",
                "efcyQesitm": item.get("efcyQesitm", "효능 정보 없음"),
                "useMethodQesitm": item.get("useMethodQesitm", "용법 정보 없음"),
                "intrcQesitm": item.get("intrcQesitm", "상호작용 정보 없음"),
                "atpnWarnQesitm": item.get("atpnWarnQesitm", ""),
                "entpName": item.get("entpName", "제조사 미확인"),
            })

        if extracted_drugs:
            _EASY_CACHE[cache_key] = (time.time(), extracted_drugs)
            return extracted_drugs, None

        return [], None

    except requests.exceptions.Timeout:
        return [], "e약은요 서버 응답 시간이 초과되었습니다."
    except Exception as e:
        return [], f"e약은요 조회 중 오류가 발생했습니다: {e}"


_DUR_URL = "https://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03"
_DUR_CACHE = {}
_DUR_CACHE_TTL = 3600


def _dur_key_variants(api_key):
    k = api_key.strip()
    raw = urllib.parse.unquote(k)
    variants = []
    for cand in (urllib.parse.quote(raw, safe=""), k):
        if cand and cand not in variants:
            variants.append(cand)
    return variants


def _dur_get(api_key, item_name, page, rows):
    query = urllib.parse.urlencode(
        {
            "typeName": "병용금기",
            "itemName": item_name,
            "type": "json",
            "numOfRows": rows,
            "pageNo": page,
        },
        quote_via=urllib.parse.quote,
    )
    last = None
    for key_part in _dur_key_variants(api_key):
        last = requests.get(f"{_DUR_URL}?serviceKey={key_part}&{query}", timeout=10)
        if last.status_code == 200:
            return last
        if "SERVICE_KEY_IS_NOT_REGISTERED" not in (last.text or ""):
            return last
    return last


def _dur_http_error(response, label="DUR"):
    detail = ""
    try:
        head = response.json().get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
        parts = [str(head.get("errMsg", "")), str(head.get("returnAuthMsg", ""))]
        detail = " / ".join(p for p in parts if p)
    except Exception:
        detail = (response.text or "").strip().replace("\n", " ")[:120]
    return f"{label} 서버 응답 실패 (HTTP {response.status_code})" + (f" - {detail}" if detail else "")


def _dur_fetch_all(api_key, item_name, max_pages=10, rows=100):
    items_all = []

    for page in range(1, max_pages + 1):
        response = _dur_get(api_key, item_name, page, rows)

        if response.status_code != 200:
            return None, _dur_http_error(response)

        try:
            data = response.json()
        except ValueError:
            return None, "DUR 응답을 해석하지 못했습니다."

        root = data.get("response", data) if isinstance(data, dict) else {}
        header = root.get("header", {}) or {}
        code = str(header.get("resultCode", "00"))

        if code not in ("00", "0"):
            return None, f"DUR API 오류 (resultCode {code})"

        body = root.get("body", {}) or {}
        items = body.get("items", [])

        if isinstance(items, dict):
            items = items.get("item", items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        items_all.extend(items)

        try:
            total = int(body.get("totalCount", 0) or 0)
        except (TypeError, ValueError):
            total = 0

        if not items or len(items_all) >= total:
            break

    return items_all, None


def _dur_name_variants(drug_name):
    names = {drug_name.strip().replace(" ", "")}
    try:
        for ing in get_drug_ingredients(drug_name):
            names.add(str(ing).strip().replace(" ", ""))
    except Exception:
        pass
    return {n for n in names if len(n) >= 2}


def _dur_find_match(items, target_names):
    for item in items:
        candidates = [
            str(item.get("MIXTURE_ITEM_NAME", "") or ""),
            str(item.get("MIXTURE_INGR_KOR_NAME", "") or ""),
        ]
        for cand in candidates:
            cand_clean = cand.strip().replace(" ", "")
            if len(cand_clean) < 2:
                continue
            for target in target_names:
                if target in cand_clean or cand_clean in target:
                    return item
    return None


def check_dur_contraindication(drug_name_a, drug_name_b, api_key):
    if not is_valid_api_key(api_key):
        return {
            "has_conflict": False,
            "risk_level": "인증 필요",
            "status": "no_key",
            "message": "DUR API 인증키가 없어 병용금기 조회를 하지 못했습니다. (사이드바 승인 버튼을 눌러주세요)"
        }

    if not drug_name_a or not drug_name_b:
        return {
            "has_conflict": False,
            "risk_level": "미입력",
            "status": "no_input",
            "message": "비교할 두 약물의 이름을 모두 입력해주세요."
        }

    cache_key = (drug_name_a.strip(), drug_name_b.strip())
    cached = _DUR_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _DUR_CACHE_TTL:
        return cached[1]

    names_a = _dur_name_variants(drug_name_a)
    names_b = _dur_name_variants(drug_name_b)

    try:
        for query_name, target_names in (
            (drug_name_a, names_b),
            (drug_name_b, names_a),
        ):
            items, err = _dur_fetch_all(api_key, query_name)

            if err:
                return {
                    "has_conflict": False,
                    "risk_level": "조회 실패",
                    "status": "error",
                    "message": err
                }

            hit = _dur_find_match(items, target_names)

            if hit:
                reason = hit.get("PROHBT_CONTENT", "병용 금기 약물입니다.")
                result = {
                    "has_conflict": True,
                    "risk_level": "위험 (병용금기)",
                    "status": "ok",
                    "message": f"[{drug_name_a}]와(과) [{drug_name_b}]는 식약처 DUR 병용금기 데이터에 등록되어 있습니다.\n\n사유: {reason}"
                }
                _DUR_CACHE[cache_key] = (time.time(), result)
                return result

        result = {
            "has_conflict": False,
            "risk_level": "등록 정보 없음",
            "status": "ok",
            "message": f"식약처 DUR에서 [{drug_name_a}]와(과) [{drug_name_b}] 간 등록된 병용금기 정보를 찾지 못했습니다."
        }
        _DUR_CACHE[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        return {
            "has_conflict": False,
            "risk_level": "오류",
            "status": "error",
            "message": f"DUR 점검 중 오류가 발생했습니다: {e}"
        }


def predict_drug_and_conflict(drug_name):
    info = DRUG_INFO.get(drug_name)
    if info is None:
        return drug_name, {
            "has_conflict": False,
            "risk_level": "정보 없음",
            "message": "등록된 의약품 정보가 없습니다.",
            "synergy_message": "공공 DB에서 정보를 확인해주세요."
        }

    return drug_name, {
        "has_conflict": False,
        "risk_level": "확인 필요",
        "message": f"주성분: {', '.join(info['ingredients'])}\n\n{info['warning']}",
        "synergy_message": info["description"]
    }


# ==========================================
# 사이드바: 버튼 클릭으로 API 및 DUR 자동 승인
# ==========================================
st.sidebar.header("⚙️ 시스템 및 API 설정")

if "api_approved" not in st.session_state:
    st.session_state.api_approved = True

if st.sidebar.button("🚀 시스템 자동 API 및 DUR 승인", type="primary"):
    st.session_state.api_approved = True
    st.sidebar.success("✅ API 및 DUR 승인이 완료되었습니다!")

DEFAULT_API_KEY = os.environ.get("DUR_API_KEY") or "3192031ee657d1b69eb0be91c00ceb22b3b9545236320c992bd87aafe23c8cf0"

if st.session_state.api_approved:
    user_api_key = DEFAULT_API_KEY
    st.sidebar.info("🔓 공공 API 연동이 활성화되어 있습니다.")
else:
    user_api_key = ""
    st.sidebar.warning("🔒 API 연동이 비활성화 상태입니다. 상단 버튼을 눌러주세요.")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM 경량화 설정")
selected_profile = st.sidebar.selectbox(
    "모델 크기 선택 (기기 사양에 맞게)",
    list(MODEL_PROFILES.keys()),
    index=0,
    key="llm_profile"
)
st.sidebar.caption(
    "경량 GGUF 모드는 별도 고성능 GPU 없이 노트북 환경에서도 가볍게 구동됩니다."
)
st.sidebar.warning("⚠️ 본 서비스는 참고용입니다. 복용 전 의사·약사 등 전문가 상담이 필요합니다.")


# 제목
st.title("💊 의약품/영양제 성분 충돌 감지 AI 시스템")

st.caption(
    "식약처 공공데이터(DUR/e약은요)와 경량 로컬 LLM(EXAONE-3.5)을 결합한 "
    "지능형 의약품 성분 및 병용금기 분석 시스템"
)

st.warning(MEDICAL_DISCLAIMER)

st.caption(
    "약품 또는 성분표 이미지를 업로드하고 함께 복용 중인 약물을 선택하여 성분 및 상호작용을 분석하세요."
)

col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("1. 약품 이미지 스캔 및 약물 선택")

    uploaded_file = st.file_uploader(
        "약봉지 또는 영양제 라벨 사진을 올려주세요",
        type=["jpg", "jpeg", "png"],
        key="drug_image"
    )

    st.markdown("---")

    selected_drug_manual = st.selectbox(
        "테스트용 약품 선택",
        DRUG_LIST,
        key="selected_drug"
    )

    # 함께 복용 중인 다른 약물 선택 (selectbox 구현)
    secondary_drug = st.selectbox(
        "함께 복용 중인 다른 약물 선택",
        ["없음"] + DRUG_LIST,
        key="secondary_drug"
    )

with col2:

    st.subheader("2. AI 성분 인식 및 충돌 분석")

    ocr_text = ""
    detected_drugs = []
    detected_ingredients = []

    if uploaded_file:

        try:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="업로드된 약품 이미지",
                width=300
            )

            with st.spinner("📖 이미지에서 글자를 읽는 중..."):
                ocr_text = extract_text_from_image(uploaded_file)

            if ocr_text.startswith("OCR_ERROR"):
                st.error(ocr_text)
            elif ocr_text:
                st.markdown("#### 🔎 OCR 인식 결과")
                st.text_area(
                    "이미지에서 읽은 문자",
                    ocr_text,
                    height=180,
                    key="ocr_result"
                )

                detected_drugs = detect_drug_name_from_text(ocr_text)
                detected_ingredients = extract_ingredients_from_text(ocr_text)

                st.markdown("#### 💊 인식된 의약품")
                if detected_drugs:
                    for drug in detected_drugs:
                        st.success(f"✅ {drug}")
                else:
                    st.info("등록된 테스트 의약품명을 찾지 못했습니다.")

                st.markdown("#### 🧬 인식된 성분")
                if detected_ingredients:
                    for ingredient in detected_ingredients:
                        st.success(f"✅ {ingredient}")
                else:
                    st.warning("현재 성분 데이터베이스에서 인식된 성분을 찾지 못했습니다.")
            else:
                st.warning("이미지에서 문자를 인식하지 못했습니다.")

        except Exception as e:
            st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")

    drug_name = selected_drug_manual

    if detected_drugs:
        drug_name = detected_drugs[0]
        st.success(f"📷 이미지에서 확인된 기준 약품: {drug_name}")

    drug_name, conflict = predict_drug_and_conflict(drug_name)

    st.markdown("---")
    st.markdown("#### 🧬 기준 약품 성분")
    base_ingredients = get_drug_ingredients(drug_name)

    if base_ingredients:
        for ingredient in base_ingredients:
            st.info(f"**{ingredient}**")

    st.markdown("#### 🚦 기본 성분 분석")
    st.warning(f"🟡 **확인 필요**\n\n{conflict['message']}")

    if detected_ingredients:
        ocr_conflicts = compare_ocr_with_drug(detected_ingredients, drug_name)
        st.markdown("#### 🔬 이미지 성분과 기준 약품 비교")
        if ocr_conflicts:
            st.error("⚠️ 동일 성분이 확인되었습니다: " + ", ".join(ocr_conflicts))
        else:
            st.success("이미지에서 인식된 성분과 기준 약품 성분 간 직접적인 중복이 확인되지 않았습니다.")

    # 다른 약물 비교 (selectbox 선택값 적용)
    dur_result = None
    if secondary_drug and secondary_drug != "없음":

        st.markdown("---")
        st.markdown("#### 🛑 다른 약물과 성분 비교")
        st.write(f"비교 대상: **{drug_name} + {secondary_drug}**")

        secondary_detected = detect_drug_name_from_text(secondary_drug)

        if secondary_detected:
            secondary_name = secondary_detected[0]
            conflicts = compare_ingredients(drug_name, secondary_name)

            if conflicts:
                st.error("⚠️ 성분 중복 가능성이 확인되었습니다.\n\n" + ", ".join(conflicts))
            else:
                st.info("등록된 성분 정보 기준으로 직접적인 성분 중복이 확인되지 않았습니다.")

        if not is_valid_api_key(user_api_key):
            st.warning("DUR 실시간 조회를 사용하려면 사이드바에서 API 연동 승인 버튼을 눌러주세요.")
        else:
            with st.spinner("식약처 DUR 병용금기 DB 조회 중..."):
                dur_result = check_dur_contraindication(
                    drug_name,
                    secondary_drug.strip(),
                    user_api_key
                )

            if dur_result["has_conflict"]:
                st.error(f"🚨 **{dur_result['risk_level']}**\n\n{dur_result['message']}")
            elif dur_result["risk_level"] in ["조회 실패", "조회 시간 초과", "네트워크 오류", "오류"]:
                st.warning(f"⚠️ **{dur_result['risk_level']}**\n\n{dur_result['message']}")
            else:
                st.success(f"ℹ️ **{dur_result['risk_level']}**\n\n{dur_result['message']}")

    # ==========================================
    # 온디바이스 LLM 심층 분석 UI
    # ==========================================
    st.markdown("---")
    st.markdown(f"#### 🧠 온디바이스 LLM ({MODEL_PROFILES[selected_profile]['short']}) 심층 복약 분석")
    st.caption("OCR 노이즈 보정, 약리학적 계열(NSAIDs 등) 중복 추론 및 식약처 데이터 기반 환자 맞춤형 요약")

    if st.button(
        "🚀 AI 복약 분석 실행 (선택한 모델 크기로 실행)",
        type="primary",
        use_container_width=True,
        key="run_local_llm_btn"
    ):
        try:
            with st.spinner(f"{MODEL_PROFILES[selected_profile]['short']} 모델이 복약 분석을 생성 중입니다..."):
                if dur_result is None:
                    dur_msg_for_llm = conflict["message"] + " (식약처 DUR 병용금기는 조회하지 않음)"
                elif dur_result.get("status") == "ok":
                    dur_msg_for_llm = dur_result["message"]
                else:
                    dur_msg_for_llm = (
                        "식약처 DUR 조회 불가: " + dur_result["message"]
                        + " (병용금기 여부는 '확인 불가'로 안내할 것)"
                    )
                llm_report = analyze_with_local_llm(
                    ocr_text=ocr_text if not ocr_text.startswith("OCR_ERROR") else "",
                    base_drug=drug_name,
                    second_drug=secondary_drug if secondary_drug != "없음" else "",
                    base_ingredients=base_ingredients,
                    dur_msg=dur_msg_for_llm,
                    profile_key=selected_profile
                )
            if llm_report.startswith("⚠️ LLM 실행에 실패"):
                st.warning("LLM을 실행하지 못해 규칙 기반 안내로 대체했습니다.")
            else:
                st.success("✅ AI 약사 종합 분석 완료")
            st.markdown(llm_report)

        except Exception as llm_err:
            error_trace = traceback.format_exc()
            st.error(f"❌ 로컬 LLM 실행 중 오류가 발생했습니다: {llm_err}")
            with st.expander("🛠️ 에러 상세 원인(Traceback) 확인하기"):
                st.code(error_trace, language="python")

# e약은요 상세 정보
st.markdown("---")
st.subheader(f"💡 식약처 e약은요 기반 [{drug_name}] 상세 정보")

if st.button("공공 DB에서 실시간 정보 불러오기", key="api_search_button"):
    if not is_valid_api_key(user_api_key):
        st.error("사이드바에서 API 인증 승인을 완료해주세요.")
    else:
        with st.spinner("식약처 e약은요 API 호출 중..."):
            drug_list, drug_error = fetch_drugs_from_api(
                drug_name,
                user_api_key,
                num_rows=4
            )

        if drug_list:
            cols = st.columns(len(drug_list))
            for idx, drug in enumerate(drug_list):
                with cols[idx]:
                    try:
                        st.image(drug["itemImage"], use_container_width=True)
                    except Exception:
                        st.write("이미지 없음")

                    st.markdown(f"**{drug['itemName']}**")
                    st.caption(f"🏢 {drug['entpName']}")
                    st.caption(f"제품 코드: `{drug['itemSeq']}`")

                    efcy_text = drug["efcyQesitm"]
                    if len(efcy_text) > 100:
                        efcy_text = efcy_text[:100] + "..."

                    st.info(f"**효능:** {efcy_text}")

                    with st.expander("📋 복약 안내 및 상호작용"):
                        st.write("**용법:**")
                        st.write(drug["useMethodQesitm"])
                        st.write("**상호작용:**")
                        st.write(drug["intrcQesitm"])
                        if drug["atpnWarnQesitm"]:
                            st.warning(f"**주의사항:** {drug['atpnWarnQesitm']}")
        else:
            st.warning(
                "공공 DB 데이터를 불러오지 못했습니다.\n\n"
                + (f"원인: {drug_error}" if drug_error else "검색 결과가 없습니다.")
            )

st.markdown("---")
st.caption("💊 본 프로그램의 의약품 정보는 공공데이터를 기반으로 제공되는 참고용 정보입니다.")
st.caption(
    "⚠️ 본 시스템은 의학적 진단·처방을 제공하지 않으며 AI 결과에는 오류가 있을 수 있습니다. "
    "실제 복용 여부 및 의약품 병용 여부는 반드시 의사 또는 약사 등 전문가에게 진단·상담을 받으시기 바랍니다."
)
