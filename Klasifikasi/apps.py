from django.apps import AppConfig


class KlasifikasiConfig(AppConfig):
    name = 'Klasifikasi'

    def ready(self):
        import Klasifikasi.signals
        _ = Klasifikasi.signals
