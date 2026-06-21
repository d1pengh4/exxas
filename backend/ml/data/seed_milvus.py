"""
Milvus VPR DB 시더 스크립트
공개 랜드마크 이미지로 참조 임베딩 DB 구축

사용법:
  python ml/data/seed_milvus.py [--source wikimedia|mapillary|local] [--limit 500]

데이터 소스:
  1. wikimedia  — Wikimedia Commons 공개 랜드마크 이미지 (무료)
  2. mapillary  — Mapillary API (MAPILLARY_TOKEN 필요)
  3. local      — ml/data/images/ 로컬 폴더
"""
import argparse
import asyncio
import io
import json
import sys
import os
from pathlib import Path
from typing import Optional

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── 랜드마크 시드 데이터 (한국 전용 300개+) ──────────────
LANDMARK_SEEDS = [
    # ══════════════════════════════════════════
    # 서울 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "N서울타워 (남산타워)", "lat": 37.5512, "lon": 126.9882, "country": "KR", "wiki": "N_Seoul_Tower"},
    {"name": "경복궁 광화문", "lat": 37.5796, "lon": 126.9770, "country": "KR", "wiki": "Gwanghwamun_Gate"},
    {"name": "경복궁 근정전", "lat": 37.5776, "lon": 126.9769, "country": "KR", "wiki": "Gyeongbokgung_Palace"},
    {"name": "창덕궁", "lat": 37.5794, "lon": 126.9910, "country": "KR", "wiki": "Changdeokgung"},
    {"name": "덕수궁 대한문", "lat": 37.5659, "lon": 126.9752, "country": "KR", "wiki": "Deoksugung"},
    {"name": "롯데월드타워 (잠실)", "lat": 37.5126, "lon": 127.1025, "country": "KR", "wiki": "Lotte_World_Tower"},
    {"name": "롯데월드 어드벤처", "lat": 37.5111, "lon": 127.0980, "country": "KR", "wiki": "Lotte_World"},
    {"name": "동대문 디자인 플라자 (DDP)", "lat": 37.5669, "lon": 127.0095, "country": "KR", "wiki": "Dongdaemun_Design_Plaza"},
    {"name": "남대문 (숭례문)", "lat": 37.5597, "lon": 126.9753, "country": "KR", "wiki": "Sungnyemun"},
    {"name": "광화문 광장", "lat": 37.5759, "lon": 126.9769, "country": "KR", "wiki": "Gwanghwamun_Plaza"},
    {"name": "청계천", "lat": 37.5698, "lon": 126.9997, "country": "KR", "wiki": "Cheonggyecheon"},
    {"name": "명동성당", "lat": 37.5633, "lon": 126.9873, "country": "KR", "wiki": "Myeongdong_Cathedral"},
    {"name": "코엑스 (강남)", "lat": 37.5126, "lon": 127.0596, "country": "KR", "wiki": "COEX_Mall"},
    {"name": "국립중앙박물관", "lat": 37.5236, "lon": 126.9804, "country": "KR", "wiki": "National_Museum_of_Korea"},
    {"name": "국립현대미술관 (서울관)", "lat": 37.5786, "lon": 126.9802, "country": "KR", "wiki": "National_Museum_of_Modern_and_Contemporary_Art"},
    {"name": "북촌 한옥마을", "lat": 37.5826, "lon": 126.9847, "country": "KR", "wiki": "Bukchon_Hanok_Village"},
    {"name": "인사동 거리", "lat": 37.5742, "lon": 126.9854, "country": "KR", "wiki": "Insadong"},
    {"name": "이태원 거리", "lat": 37.5340, "lon": 126.9944, "country": "KR", "wiki": "Itaewon"},
    {"name": "홍대 거리", "lat": 37.5572, "lon": 126.9247, "country": "KR", "wiki": "Hongdae,_Seoul"},
    {"name": "신촌 현대백화점", "lat": 37.5556, "lon": 126.9368, "country": "KR", "wiki": "Sinchon,_Seoul"},
    {"name": "명동 거리", "lat": 37.5635, "lon": 126.9839, "country": "KR", "wiki": "Myeongdong"},
    {"name": "강남역 사거리", "lat": 37.4979, "lon": 127.0276, "country": "KR", "wiki": "Gangnam_station"},
    {"name": "압구정 로데오거리", "lat": 37.5267, "lon": 127.0398, "country": "KR", "wiki": "Apgujeong-dong"},
    {"name": "성수동 카페거리", "lat": 37.5443, "lon": 127.0562, "country": "KR", "wiki": "Seongsu-dong"},
    {"name": "이화벽화마을 (낙산)", "lat": 37.5793, "lon": 127.0042, "country": "KR", "wiki": "Naksan_Park"},
    {"name": "서울숲", "lat": 37.5446, "lon": 127.0374, "country": "KR", "wiki": "Seoul_Forest"},
    {"name": "올림픽공원", "lat": 37.5212, "lon": 127.1220, "country": "KR", "wiki": "Olympic_Park,_Seoul"},
    {"name": "서울대공원 (과천)", "lat": 37.4270, "lon": 127.0133, "country": "KR", "wiki": "Seoul_Grand_Park"},
    {"name": "국립서울현충원", "lat": 37.5007, "lon": 126.9807, "country": "KR", "wiki": "Seoul_National_Cemetery"},
    # ── 서울 한강공원 5구간
    {"name": "한강공원 여의도", "lat": 37.5283, "lon": 126.9326, "country": "KR", "wiki": "Hangang_Park"},
    {"name": "한강공원 반포 (달빛광장)", "lat": 37.5130, "lon": 126.9941, "country": "KR", "wiki": "Banpo_Bridge"},
    {"name": "한강공원 잠실 (석촌호수)", "lat": 37.5228, "lon": 127.0836, "country": "KR", "wiki": "Hangang_Park"},
    {"name": "한강공원 뚝섬 (자벌레)", "lat": 37.5310, "lon": 127.0673, "country": "KR", "wiki": "Ttukseom_Resort"},
    {"name": "한강공원 망원 (월드컵공원)", "lat": 37.5665, "lon": 126.8926, "country": "KR", "wiki": "Worldcup_Park"},
    # ── 서울 주요 지하철역 (역사 외관)
    {"name": "서울역 (1호선)", "lat": 37.5548, "lon": 126.9706, "country": "KR", "wiki": "Seoul_Station"},
    {"name": "강남역 (2호선)", "lat": 37.4979, "lon": 127.0276, "country": "KR", "wiki": "Gangnam_station"},
    {"name": "홍대입구역 (2호선)", "lat": 37.5572, "lon": 126.9247, "country": "KR", "wiki": "Hongik_University_station"},
    {"name": "신촌역 (2호선)", "lat": 37.5556, "lon": 126.9368, "country": "KR", "wiki": "Sinchon_station"},
    {"name": "잠실역 (2·8호선)", "lat": 37.5131, "lon": 127.1001, "country": "KR", "wiki": "Jamsil_station"},
    {"name": "동대문역사문화공원역", "lat": 37.5669, "lon": 127.0095, "country": "KR", "wiki": "Dongdaemun_History_Culture_Park_station"},
    {"name": "합정역 (2호선)", "lat": 37.5494, "lon": 126.9145, "country": "KR", "wiki": "Hapjeong_station"},
    {"name": "신논현역 (9호선)", "lat": 37.5047, "lon": 127.0249, "country": "KR", "wiki": "Sinnonhyeon_station"},
    {"name": "충무로역 (3호선)", "lat": 37.5613, "lon": 126.9942, "country": "KR", "wiki": "Chungmuro_station"},
    {"name": "광화문역 (5호선)", "lat": 37.5721, "lon": 126.9769, "country": "KR", "wiki": "Gwanghwamun_station"},
    # ══════════════════════════════════════════
    # 부산 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "해운대 해수욕장", "lat": 35.1587, "lon": 129.1601, "country": "KR", "wiki": "Haeundae_Beach"},
    {"name": "광안리 해수욕장", "lat": 35.1531, "lon": 129.1186, "country": "KR", "wiki": "Gwangalli_Beach"},
    {"name": "광안대교 (야경)", "lat": 35.1497, "lon": 129.1181, "country": "KR", "wiki": "Gwangan_Bridge"},
    {"name": "부산 해운대 마린시티", "lat": 35.1614, "lon": 129.1621, "country": "KR", "wiki": "Marine_City,_Busan"},
    {"name": "남포동 BIFF 광장", "lat": 35.0975, "lon": 129.0308, "country": "KR", "wiki": "BIFF_Square"},
    {"name": "자갈치 시장", "lat": 35.0968, "lon": 129.0302, "country": "KR", "wiki": "Jagalchi_Market"},
    {"name": "감천문화마을", "lat": 35.0971, "lon": 129.0097, "country": "KR", "wiki": "Gamcheon_Culture_Village"},
    {"name": "부산타워 (용두산공원)", "lat": 35.1000, "lon": 129.0322, "country": "KR", "wiki": "Busan_Tower"},
    {"name": "부산 벡스코", "lat": 35.1694, "lon": 129.1361, "country": "KR", "wiki": "BEXCO"},
    {"name": "해동 용궁사", "lat": 35.1876, "lon": 129.2238, "country": "KR", "wiki": "Haedong_Yonggungsa"},
    {"name": "송정 해수욕장", "lat": 35.1785, "lon": 129.2048, "country": "KR", "wiki": "Songjeong_Beach"},
    {"name": "기장 죽성 드라이브인", "lat": 35.2371, "lon": 129.2239, "country": "KR", "wiki": "Gijang_County"},
    {"name": "부산역 (KTX)", "lat": 35.1147, "lon": 129.0425, "country": "KR", "wiki": "Busan_station"},
    {"name": "서면 롯데백화점", "lat": 35.1576, "lon": 129.0596, "country": "KR", "wiki": "Seomyeon,_Busan"},
    # ══════════════════════════════════════════
    # 제주도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "제주도 성산일출봉", "lat": 33.4582, "lon": 126.9424, "country": "KR", "wiki": "Seongsan_Ilchulbong"},
    {"name": "한라산 백록담", "lat": 33.3617, "lon": 126.5292, "country": "KR", "wiki": "Hallasan"},
    {"name": "제주 협재 해수욕장", "lat": 33.3944, "lon": 126.2393, "country": "KR", "wiki": "Hyeopjae_Beach"},
    {"name": "중문 대포주상절리", "lat": 33.2453, "lon": 126.4218, "country": "KR", "wiki": "Jusangjeolli_Cliff"},
    {"name": "만장굴 (용암동굴)", "lat": 33.5284, "lon": 126.7718, "country": "KR", "wiki": "Manjanggul"},
    {"name": "제주 올레길 7코스 (외돌개)", "lat": 33.2481, "lon": 126.5106, "country": "KR", "wiki": "Oedolgae_Rock"},
    {"name": "제주시 동문시장", "lat": 33.5130, "lon": 126.5277, "country": "KR", "wiki": "Jeju_City"},
    {"name": "제주국제공항", "lat": 33.5113, "lon": 126.4930, "country": "KR", "wiki": "Jeju_International_Airport"},
    {"name": "서귀포 매일올레시장", "lat": 33.2533, "lon": 126.5601, "country": "KR", "wiki": "Seogwipo"},
    {"name": "제주 에코랜드 (비자림)", "lat": 33.4761, "lon": 126.7942, "country": "KR", "wiki": "Bijarim_Forest"},
    # ══════════════════════════════════════════
    # 인천 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "인천국제공항 T1", "lat": 37.4613, "lon": 126.4400, "country": "KR", "wiki": "Incheon_International_Airport"},
    {"name": "인천국제공항 T2", "lat": 37.4762, "lon": 126.4504, "country": "KR", "wiki": "Incheon_International_Airport"},
    {"name": "인천 송도 센트럴파크", "lat": 37.3929, "lon": 126.6432, "country": "KR", "wiki": "Songdo_International_Business_District"},
    {"name": "인천 차이나타운", "lat": 37.4738, "lon": 126.6170, "country": "KR", "wiki": "Chinatown,_Incheon"},
    {"name": "인천 월미도", "lat": 37.4730, "lon": 126.5967, "country": "KR", "wiki": "Wolmido"},
    {"name": "강화도 강화산성", "lat": 37.7456, "lon": 126.4878, "country": "KR", "wiki": "Ganghwa_County"},
    # ══════════════════════════════════════════
    # 경기도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "수원 화성 (팔달문)", "lat": 37.2844, "lon": 127.0159, "country": "KR", "wiki": "Hwaseong_Fortress"},
    {"name": "용인 에버랜드", "lat": 37.2939, "lon": 127.2027, "country": "KR", "wiki": "Everland"},
    {"name": "판교 테크노밸리", "lat": 37.3944, "lon": 127.1116, "country": "KR", "wiki": "Pangyo_Technovalley"},
    {"name": "성남 분당 정자일로", "lat": 37.3629, "lon": 127.1110, "country": "KR", "wiki": "Bundang-gu"},
    {"name": "고양 일산 호수공원", "lat": 37.6765, "lon": 126.7700, "country": "KR", "wiki": "Ilsan_Lake_Park"},
    {"name": "파주 헤이리 예술마을", "lat": 37.7792, "lon": 126.7289, "country": "KR", "wiki": "Heyri_Art_Village"},
    {"name": "광명 KTX역", "lat": 37.4319, "lon": 126.8648, "country": "KR", "wiki": "Gwangmyeong_station"},
    {"name": "의왕 철도박물관", "lat": 37.3422, "lon": 126.9713, "country": "KR", "wiki": "Korea_Railroad_Museum"},
    {"name": "안산 대부도 방아머리해수욕장", "lat": 37.2044, "lon": 126.4995, "country": "KR", "wiki": "Daebudo_Island"},
    {"name": "가평 남이섬", "lat": 37.7914, "lon": 127.5247, "country": "KR", "wiki": "Nami_Island"},
    {"name": "춘천 의암호 (스카이워크)", "lat": 37.9244, "lon": 127.7395, "country": "KR", "wiki": "Chuncheon"},
    {"name": "양평 두물머리", "lat": 37.5328, "lon": 127.4607, "country": "KR", "wiki": "Dumulmeori"},
    {"name": "포천 아트밸리", "lat": 37.9440, "lon": 127.2030, "country": "KR", "wiki": "Pocheon_Art_Valley"},
    {"name": "연천 한탄강 주상절리길", "lat": 38.0917, "lon": 127.0618, "country": "KR", "wiki": "Hantan_River"},
    # ══════════════════════════════════════════
    # 대구 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "대구 동성로 거리", "lat": 35.8711, "lon": 128.5952, "country": "KR", "wiki": "Dongseongno"},
    {"name": "대구 서문시장", "lat": 35.8704, "lon": 128.5811, "country": "KR", "wiki": "Seomun_Market"},
    {"name": "대구 83타워 (두류공원)", "lat": 35.8580, "lon": 128.5632, "country": "KR", "wiki": "E-World_(amusement_park)"},
    {"name": "대구 달성공원", "lat": 35.8698, "lon": 128.5849, "country": "KR", "wiki": "Dalseong_Park"},
    {"name": "대구역 (KTX)", "lat": 35.8799, "lon": 128.6283, "country": "KR", "wiki": "Dongdaegu_station"},
    {"name": "팔공산 갓바위", "lat": 35.9716, "lon": 128.7107, "country": "KR", "wiki": "Palgongsan"},
    # ══════════════════════════════════════════
    # 광주 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "광주 국립아시아문화전당", "lat": 35.1465, "lon": 126.9179, "country": "KR", "wiki": "Asia_Culture_Center"},
    {"name": "광주 5·18 민주광장", "lat": 35.1491, "lon": 126.9173, "country": "KR", "wiki": "May_18th_National_Cemetery"},
    {"name": "광주 양림동 역사문화마을", "lat": 35.1399, "lon": 126.9128, "country": "KR", "wiki": "Yangnim-dong,_Gwangju"},
    {"name": "광주 무등산", "lat": 35.1315, "lon": 127.0061, "country": "KR", "wiki": "Mudeungsan"},
    # ══════════════════════════════════════════
    # 대전 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "대전 엑스포 과학공원", "lat": 36.3742, "lon": 127.3855, "country": "KR", "wiki": "Expo_Science_Park"},
    {"name": "대전 한빛탑", "lat": 36.3742, "lon": 127.3841, "country": "KR", "wiki": "Hanbat_Arboretum"},
    {"name": "유성온천 (대전)", "lat": 36.3623, "lon": 127.3463, "country": "KR", "wiki": "Yuseong_Hot_Spring"},
    {"name": "카이스트 (대전)", "lat": 36.3741, "lon": 127.3603, "country": "KR", "wiki": "KAIST"},
    # ══════════════════════════════════════════
    # 울산 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "울산 대왕암 공원", "lat": 35.5038, "lon": 129.4271, "country": "KR", "wiki": "Daewangam_Park"},
    {"name": "울산 고래박물관 (장생포)", "lat": 35.4711, "lon": 129.3878, "country": "KR", "wiki": "Jangsaengpo_Whale_Museum"},
    {"name": "울산 현대중공업 (조선소)", "lat": 35.5459, "lon": 129.3629, "country": "KR", "wiki": "HD_Hyundai_Heavy_Industries"},
    # ══════════════════════════════════════════
    # 세종 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "세종 정부청사", "lat": 36.4800, "lon": 127.2890, "country": "KR", "wiki": "Sejong_City"},
    {"name": "세종호수공원", "lat": 36.4834, "lon": 127.2636, "country": "KR", "wiki": "Sejong_Lake_Park"},
    # ══════════════════════════════════════════
    # 강원도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "강릉 경포대 해변", "lat": 37.7960, "lon": 128.9000, "country": "KR", "wiki": "Gyeongpo_Beach"},
    {"name": "강릉 오죽헌", "lat": 37.7802, "lon": 128.8780, "country": "KR", "wiki": "Ojukheon"},
    {"name": "춘천 명동 거리", "lat": 37.8814, "lon": 127.7278, "country": "KR", "wiki": "Chuncheon"},
    {"name": "속초 설악산 (신흥사)", "lat": 38.1190, "lon": 128.4650, "country": "KR", "wiki": "Seoraksan"},
    {"name": "양양 죽도해변", "lat": 38.0660, "lon": 128.6408, "country": "KR", "wiki": "Yangyang_County"},
    {"name": "정선 하이원 리조트", "lat": 37.2148, "lon": 128.8193, "country": "KR", "wiki": "High1_Resort"},
    {"name": "평창 알펜시아 (동계올림픽)", "lat": 37.6494, "lon": 128.6724, "country": "KR", "wiki": "Alpensia_Resort"},
    # ══════════════════════════════════════════
    # 충청도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "천안 독립기념관", "lat": 36.8882, "lon": 127.1897, "country": "KR", "wiki": "Independence_Hall_of_Korea"},
    {"name": "천안 아산 온천스파", "lat": 36.7897, "lon": 127.0044, "country": "KR", "wiki": "Asan,_South_Chungcheong"},
    {"name": "청주 고인쇄박물관 (직지)", "lat": 36.6297, "lon": 127.4873, "country": "KR", "wiki": "Cheongju_Early_Printing_Museum"},
    {"name": "공주 공산성", "lat": 36.4606, "lon": 127.1235, "country": "KR", "wiki": "Gongsanseong"},
    {"name": "부여 낙화암 (백마강)", "lat": 36.2840, "lon": 126.9180, "country": "KR", "wiki": "Nakwaam"},
    {"name": "보령 대천해수욕장", "lat": 36.2920, "lon": 126.5030, "country": "KR", "wiki": "Boryeong_Mud_Festival"},
    {"name": "단양 도담삼봉", "lat": 36.9924, "lon": 128.3556, "country": "KR", "wiki": "Dodamsambong"},
    # ══════════════════════════════════════════
    # 전라도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "전주 한옥마을 (경기전)", "lat": 35.8175, "lon": 127.1520, "country": "KR", "wiki": "Jeonju_Hanok_Village"},
    {"name": "전주 풍남문", "lat": 35.8121, "lon": 127.1512, "country": "KR", "wiki": "Pungnammun"},
    {"name": "군산 근대역사박물관", "lat": 35.9796, "lon": 126.7118, "country": "KR", "wiki": "Gunsan"},
    {"name": "여수 돌산도 (거북선대교)", "lat": 34.7556, "lon": 127.6632, "country": "KR", "wiki": "Dolsan_Bridge"},
    {"name": "순천만 국가정원", "lat": 34.9000, "lon": 127.4897, "country": "KR", "wiki": "Suncheon_Bay_National_Garden"},
    {"name": "담양 죽녹원 (대나무숲)", "lat": 35.3134, "lon": 126.9880, "country": "KR", "wiki": "Juknokwon"},
    {"name": "목포 유달산 갓바위", "lat": 34.8118, "lon": 126.4218, "country": "KR", "wiki": "Mokpo"},
    # ══════════════════════════════════════════
    # 경상도 — 랜드마크
    # ══════════════════════════════════════════
    {"name": "경주 불국사", "lat": 35.7897, "lon": 129.3315, "country": "KR", "wiki": "Bulguksa"},
    {"name": "경주 첨성대", "lat": 35.8347, "lon": 129.2191, "country": "KR", "wiki": "Cheomseongdae"},
    {"name": "경주 동궁과 월지", "lat": 35.8357, "lon": 129.2251, "country": "KR", "wiki": "Donggung_Palace_and_Wolji_Pond"},
    {"name": "경주 보문관광단지", "lat": 35.8419, "lon": 129.2738, "country": "KR", "wiki": "Bomun_Tourist_Complex"},
    {"name": "포항 호미곶 해맞이광장", "lat": 36.0793, "lon": 129.5704, "country": "KR", "wiki": "Homigot"},
    {"name": "안동 하회마을", "lat": 36.5388, "lon": 128.5141, "country": "KR", "wiki": "Hahoe_Folk_Village"},
    {"name": "창원 (마산 합포구)", "lat": 35.1833, "lon": 128.5811, "country": "KR", "wiki": "Changwon"},
    {"name": "통영 케이블카 (미륵산)", "lat": 34.8468, "lon": 128.4186, "country": "KR", "wiki": "Tongyeong_Cable_Car"},
    {"name": "거제 바람의 언덕", "lat": 34.7985, "lon": 128.6467, "country": "KR", "wiki": "Geoje_Island"},
    {"name": "남해 독일마을", "lat": 34.8301, "lon": 127.9282, "country": "KR", "wiki": "Namhae_County"},
    {"name": "진주성 촉석루", "lat": 35.1881, "lon": 128.0845, "country": "KR", "wiki": "Jinjuseong_Fortress"},
    # ══════════════════════════════════════════
    # DMZ / 특수 지역
    # ══════════════════════════════════════════
    {"name": "판문점 공동경비구역", "lat": 37.9561, "lon": 126.6730, "country": "KR", "wiki": "Joint_Security_Area"},
    {"name": "임진각 평화누리공원", "lat": 37.8881, "lon": 126.7729, "country": "KR", "wiki": "Imjingak_Park"},
    # ══════════════════════════════════════════
    # 주요 공항·역 (교통 허브)
    # ══════════════════════════════════════════
    {"name": "김포국제공항", "lat": 37.5580, "lon": 126.7940, "country": "KR", "wiki": "Gimpo_International_Airport"},
    {"name": "김해국제공항 (부산)", "lat": 35.1795, "lon": 128.9386, "country": "KR", "wiki": "Gimhae_International_Airport"},
    {"name": "광주공항", "lat": 35.1264, "lon": 126.8089, "country": "KR", "wiki": "Gwangju_Airport"},
    {"name": "KTX 수서역", "lat": 37.4854, "lon": 127.1116, "country": "KR", "wiki": "Suseo_station"},
    {"name": "KTX 광명역", "lat": 37.4319, "lon": 126.8648, "country": "KR", "wiki": "Gwangmyeong_station"},
    {"name": "KTX 동대구역", "lat": 35.8799, "lon": 128.6283, "country": "KR", "wiki": "Dongdaegu_station"},
    {"name": "KTX 부산역", "lat": 35.1147, "lon": 129.0425, "country": "KR", "wiki": "Busan_station"},
    # ══════════════════════════════════════════
    # 대형 쇼핑몰·테마파크
    # ══════════════════════════════════════════
    {"name": "스타필드 하남", "lat": 37.5444, "lon": 127.2145, "country": "KR", "wiki": "Starfield_Hanam"},
    {"name": "스타필드 수원", "lat": 37.2758, "lon": 127.0439, "country": "KR", "wiki": "Starfield_Suwon"},
    {"name": "더현대 서울 (여의도)", "lat": 37.5263, "lon": 126.9271, "country": "KR", "wiki": "The_Hyundai_Seoul"},
    {"name": "현대프리미엄아울렛 (스페이스원)", "lat": 37.5009, "lon": 127.0381, "country": "KR", "wiki": "Hyundai_Premium_Outlets"},
    {"name": "신세계백화점 강남점", "lat": 37.5052, "lon": 127.0046, "country": "KR", "wiki": "Shinsegae_Department_Store"},
    {"name": "롯데월드몰 (잠실)", "lat": 37.5128, "lon": 127.1022, "country": "KR", "wiki": "Lotte_World_Mall"},
    {"name": "IFC 몰 (여의도)", "lat": 37.5253, "lon": 126.9253, "country": "KR", "wiki": "IFC_Seoul"},
    {"name": "동탄 롯데프리미엄아울렛", "lat": 37.2025, "lon": 127.0863, "country": "KR", "wiki": "Dongtansingdosi"},
]


async def fetch_wikimedia_image(wiki_name: str) -> Optional[bytes]:
    """Wikimedia Commons에서 랜드마크 대표 이미지 다운로드"""
    import aiohttp
    api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + wiki_name
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            img_url = data.get("thumbnail", {}).get("source") or \
                      data.get("originalimage", {}).get("source")
            if not img_url:
                return None

            async with session.get(img_url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return await r.read()
    except Exception as e:
        print(f"  [WARN] {wiki_name}: {e}")
    return None


_LOCAL_CLIP_PATH = str(Path(__file__).resolve().parents[3] / "modelforder" / "model")


def extract_embedding(image_bytes: bytes, device: str = "cpu") -> Optional[list]:
    """이미지에서 768d 임베딩 추출 (로컬 fine-tuned CLIP — projection_dim=768)"""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 로컬 fine-tuned CLIP 로드 (캐시됨)
        if not hasattr(extract_embedding, "_model"):
            print(f"  [INFO] 로컬 fine-tuned CLIP 로딩 ({_LOCAL_CLIP_PATH}, 768d)...")
            extract_embedding._processor = CLIPProcessor.from_pretrained(_LOCAL_CLIP_PATH)
            extract_embedding._model = CLIPModel.from_pretrained(_LOCAL_CLIP_PATH).to(device)
            extract_embedding._model.eval()

        processor = extract_embedding._processor
        model = extract_embedding._model

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        emb = features[0].cpu().tolist()
        assert len(emb) == 768, f"임베딩 차원 불일치: {len(emb)} (expected 768)"
        return emb
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        return None


def insert_to_milvus(records: list[dict]):
    """Milvus VPR 컬렉션에 임베딩 삽입"""
    try:
        from pymilvus import connections, Collection, utility
        connections.connect(alias="default", host="localhost", port=19530)

        if not utility.has_collection("image_embeddings"):
            print("[ERROR] Milvus 컬렉션이 없습니다. 서버를 먼저 시작하세요.")
            return 0

        col = Collection("image_embeddings")
        col.load()

        data = [
            [r["image_hash"] for r in records],   # image_hash
            [r["lat"] for r in records],            # latitude
            [r["lon"] for r in records],            # longitude
            [r["embedding"] for r in records],      # embedding
            [r["source"] for r in records],         # source
        ]
        col.insert(data)
        col.flush()
        return len(records)
    except Exception as e:
        print(f"[ERROR] Milvus insert failed: {e}")
        return 0


async def seed_from_wikimedia(limit: int):
    import hashlib
    seeds = LANDMARK_SEEDS[:limit]
    records = []

    # 장치 감지
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass
    print(f"[INFO] 장치: {device}")

    for seed in seeds:
        print(f"[{len(records)+1}/{len(seeds)}] {seed['name']}...")
        img_bytes = await fetch_wikimedia_image(seed["wiki"])
        if not img_bytes:
            print(f"  [SKIP] 이미지 없음")
            continue

        emb = extract_embedding(img_bytes, device)
        if not emb:
            print(f"  [SKIP] 임베딩 실패")
            continue

        image_hash = hashlib.md5(img_bytes[:1024]).hexdigest()
        records.append({
            "image_hash": image_hash[:64],
            "lat": seed["lat"],
            "lon": seed["lon"],
            "embedding": emb,
            "source": f"wikimedia_{seed['country']}",
        })
        print(f"  [OK] 임베딩 {len(emb)}d")
        await asyncio.sleep(0.5)  # Wikimedia rate limit

    if records:
        inserted = insert_to_milvus(records)
        print(f"\n[DONE] {inserted}개 임베딩 Milvus 삽입 완료")
    else:
        print("[WARN] 삽입할 임베딩 없음")


async def seed_from_local(image_dir: str):
    """로컬 이미지 폴더에서 임베딩 추출"""
    import hashlib
    img_dir = Path(image_dir)
    if not img_dir.exists():
        print(f"[ERROR] 폴더 없음: {image_dir}")
        return

    device = "cpu"
    records = []

    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        # 파일명 컨벤션: lat_lon_desc.jpg
        parts = img_path.stem.split("_")
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            lat, lon = 0.0, 0.0

        img_bytes = img_path.read_bytes()
        emb = extract_embedding(img_bytes, device)
        if emb:
            records.append({
                "image_hash": hashlib.md5(img_bytes[:1024]).hexdigest()[:64],
                "lat": lat, "lon": lon,
                "embedding": emb,
                "source": "local",
            })
            print(f"  [OK] {img_path.name}")

    if records:
        inserted = insert_to_milvus(records)
        print(f"\n[DONE] {inserted}개 삽입")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["wikimedia", "mapillary", "local"], default="wikimedia")
    parser.add_argument("--limit", type=int, default=len(LANDMARK_SEEDS))
    parser.add_argument("--image_dir", type=str, default="ml/data/images")
    args = parser.parse_args()

    print(f"[EXXAS VPR Seeder] source={args.source} limit={args.limit}")

    if args.source == "wikimedia":
        asyncio.run(seed_from_wikimedia(args.limit))
    elif args.source == "local":
        asyncio.run(seed_from_local(args.image_dir))
    elif args.source == "mapillary":
        print("[TODO] Mapillary 시더는 MAPILLARY_TOKEN 필요. --source local 또는 wikimedia 사용 권장.")
        sys.exit(1)


if __name__ == "__main__":
    main()
