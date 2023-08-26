from django.contrib import admin

from solo.admin import SingletonModelAdmin

from options.models import Options


@admin.register(Options)
class OptionsAdmin(SingletonModelAdmin):
    pass
