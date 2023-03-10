from django.contrib import admin
# from models import User_Model
from summary.models import User_Model, AdminModel, AdvancedSummary, TranslationsModel, Notifications, SummaryModel, ApprovalModel, ContentModal, TwitterAnalysis, MeetingConversionModel
admin.site.register(User_Model)
admin.site.register(AdminModel)
admin.site.register(AdvancedSummary)
admin.site.register(TranslationsModel)
admin.site.register(SummaryModel)
admin.site.register(Notifications)
admin.site.register(ApprovalModel)
admin.site.register(ContentModal)
admin.site.register(TwitterAnalysis)
admin.site.register(MeetingConversionModel)
