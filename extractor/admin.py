from django.contrib import admin
from .models import Document, DocumentPage, Folder, Tag, Activity

admin.site.register(Folder)
admin.site.register(Tag)
admin.site.register(Activity)

class DocumentPageInline(admin.TabularInline):
    model = DocumentPage
    extra = 0

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['filename', 'status', 'file_type', 'is_starred', 'folder', 'created_at']
    list_filter = ['status', 'file_type', 'is_starred']
    inlines = [DocumentPageInline]
