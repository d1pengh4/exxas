"""
macOS SSL 인증서 패치 + BLAS 스레드 픽스
Celery worker, FastAPI 모두 import 시 자동 적용
"""
import ssl
import os
import certifi

_CERT = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _CERT)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERT)
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=_CERT)

# macOS 26 Accelerate/BLAS hang fix:
# scipy.linalg (via Accelerate framework) 초기화가 macOS 26에서 hang됨.
# VECLIB_MAXIMUM_THREADS=1 로 단일 스레드 초기화 강제.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def get_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=_CERT)
