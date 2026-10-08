"""
app/core/offline.py

Pelindung mode offline (Offline Guard).
Jika UDARASIAGA_OFFLINE=1, memblokir pemanggilan jaringan eksternal dan
memastikan aplikasi hanya mengakses file lokal di data/.
"""

import os


class OfflineGuardException(RuntimeError):
    """Dilempar saat panggilan eksternal dicoba ketika mode offline aktif."""
    pass


def is_offline() -> bool:
    """Mengembalikan True jika mode offline aktif."""
    return os.environ.get("UDARASIAGA_OFFLINE", "1") == "1"


def assert_online_allowed(source_name: str = "eksternal"):
    """Memastikan pemanggilan online diizinkan."""
    if is_offline():
        raise OfflineGuardException(
            f"Pemanggilan jaringan ke {source_name} diblokir: UDARASIAGA_OFFLINE=1 aktif. "
            f"Gunakan data lokal dari data/events."
        )
