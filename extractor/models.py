import uuid
from django.db import models


class Folder(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name       = models.CharField(max_length=200)
    parent     = models.ForeignKey('self', null=True, blank=True,
                                   on_delete=models.CASCADE, related_name='children')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']


class Tag(models.Model):
    id    = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name  = models.CharField(max_length=64, unique=True)
    color = models.CharField(max_length=20, default='gray')  # gray|red|blue|green|yellow|purple

    class Meta:
        ordering = ['name']


class Document(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename   = models.CharField(max_length=512)
    pdf_file   = models.FileField(upload_to='documents/pdfs/')
    thumbnail  = models.ImageField(upload_to='documents/thumbnails/', null=True, blank=True)
    page_count = models.PositiveIntegerField(default=0)
    status     = models.CharField(max_length=20, default='uploaded')  # uploaded|partial|complete
    folder     = models.ForeignKey(Folder, null=True, blank=True,
                                   on_delete=models.SET_NULL, related_name='documents')
    tags       = models.ManyToManyField(Tag, blank=True, related_name='documents')
    is_starred = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    file_size  = models.PositiveBigIntegerField(default=0)
    file_type  = models.CharField(max_length=20, default='pdf')  # pdf|image|other
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']


class DocumentPage(models.Model):
    id            = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document      = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='pages')
    page_number   = models.PositiveIntegerField()  # 1-based, matches page.pageNo in JS
    session_id    = models.CharField(max_length=64, blank=True, default='')
    image_w       = models.PositiveIntegerField(default=0)
    image_h       = models.PositiveIntegerField(default=0)
    layout_blocks      = models.JSONField(default=list)
    ocr_blocks         = models.JSONField(default=list)
    placed_images      = models.JSONField(default=list)  # Blank Document only
    status             = models.CharField(max_length=20, default='idle')
    structured_content = models.JSONField(null=True, blank=True)
    structure_status   = models.CharField(max_length=20, default='none')
    # structure_status: none | auto_parsed | edited

    class Meta:
        unique_together = ('document', 'page_number')
        ordering = ['page_number']


class Activity(models.Model):
    id        = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document  = models.ForeignKey(Document, null=True, blank=True,
                                  on_delete=models.SET_NULL, related_name='activities')
    doc_label = models.CharField(max_length=512, default='')
    action    = models.CharField(max_length=40)  # uploaded|ocr_started|ocr_completed|marked_review
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
