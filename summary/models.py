from django.db import models
from django.db.models import CharField
from django_mysql.models import ListCharField
import json
# Create your models here.


class User_Model(models.Model):
    user_id = models.BigAutoField(primary_key=True)
    user_Name = models.CharField(max_length=50)
    user_Email = models.EmailField(max_length=254)

    user_Mobile = models.CharField(max_length=10)
    user_UserName = models.CharField(max_length=50)
    user_PWD = models.CharField(max_length=250)
    user_is_Active = models.BooleanField(default=False)

    class Meta:
        db_table = "user_tbl"


class AdminModel(models.Model):
    admin_id = models.BigAutoField(primary_key=True)
    admin_usr_nm = models.CharField(max_length=50)
    admin_pwd = models.CharField(max_length=50)

    class Meta:
        db_table = 'admin_tbl'


class SummaryModel(models.Model):
    summary_approved_by = models.ForeignKey(
        'AdminModel', on_delete=models.CASCADE,
        blank=True,
        null=True)
    summary_id = models.BigAutoField(primary_key=True)
    summary_added_by = models.ForeignKey(
        "User_Model", on_delete=models.CASCADE)
    summary_text = models.CharField(max_length=5000)
    summary_org_file = models.TextField()
    summary_author = models.CharField(max_length=50)

    summary_timestamp = models.DateTimeField(auto_now_add=True)
    summary_approval_status = models.BooleanField(default=False)
    summary_added_to_twitter = models.BooleanField(default=False)
    summary_added_to_instagram = models.BooleanField(default=False)
    summary_availalabe_trans = models.BooleanField(default=False)

    summary_advance_available = models.BooleanField(default=False)
    adv_summary_PK = models.ForeignKey(
        'AdvancedSummary', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        db_table = 'summary_tbl'


class AdvancedSummary(models.Model):
    adv_summary_id = models.BigAutoField(primary_key=True)
    adv_summaryTextID = models.ForeignKey(
        "SummaryModel", on_delete=models.CASCADE)
    adv_summary_added_by = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    adv_summary_text = models.TextField()
    adv_summary_approval_status = models.BooleanField(default=False)
    adv_summary_author = models.CharField(max_length=50, default="NONE")
    adv_summary_approved_by = models.ForeignKey(
        'AdminModel', on_delete=models.CASCADE, null=True, blank=True)
    adv_summary_added_to_twitter = models.BooleanField(default=False)
    adv_summary_added_to_instagram = models.BooleanField(default=False)
    adv_avaibale_trans = models.BooleanField(default=False)

    adv_summary_approved_timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'adv_summary'


class TranslationsModel(models.Model):
    transID = models.BigAutoField(primary_key=True)
    transLang = models.CharField(max_length=20)
    transText = models.CharField(max_length=5000)
    transDoneBy = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    transDoneAt = models.DateTimeField(auto_now_add=True)
    transToTwitter = models.BooleanField(default=False)
    transToInsagram = models.BooleanField(default=False)
    transApprovalStatus = models.BooleanField(default=False)
    transApprovedBy = models.ForeignKey(
        'AdminModel', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        db_table = 'transactions_table'


class Notifications(models.Model):
    notifID = models.BigAutoField(primary_key=True)
    notificationAddedby = models.CharField(max_length=50)
    noticationsMSG = models.TextField()
    notificationSentTo = models.CharField(max_length=50)
    notificationTimeStamp = models.DateTimeField(auto_now_add=True)
    noticatinosView = ListCharField(
        base_field=CharField(max_length=10),
        size=6,
        max_length=(6 * 11),  # 6 * 10 character nominals, plus commas
    )

    class Meta:
        db_table = 'notifications_table'


ApprovalStaus = (
    ('pending', 'Pending'),
    ('approved', 'Approved'),
    ('rejected', 'Rejected')

)


class ApprovalModel(models.Model):
    approvalID = models.AutoField(primary_key=True)
    approvalType = models.CharField(max_length=50)
    approdeval_sent_by = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    approvalStatus = models.CharField(
        choices=ApprovalStaus, default='pending', max_length=12)
    approval_type_id = models.CharField(max_length=50)
    approval_remaks = models.TextField(max_length=500)
    approvedBy = models.ForeignKey(
        'AdminModel', null=True, blank=True, on_delete=models.CASCADE)
    approval_time = models.DateField(auto_now_add=True, null=True)

    class Meta:
        db_table = 'approvals'


class ContentModal(models.Model):
    contentID = models.AutoField(primary_key=True)
    contentType = models.CharField(max_length=50)
    contentQN = models.CharField(max_length=200)
    contentAddedBy = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    contentApprovalStatus = models.CharField(
        choices=ApprovalStaus, default='pending', max_length=12)
    contentApproval_remaks = models.TextField(max_length=500, default="N?A")
    contentApprovedBy = models.ForeignKey(
        'AdminModel', null=True, blank=True, on_delete=models.CASCADE)
    contentAdded_time = models.DateTimeField(auto_now_add=True, null=True)
    content_text = models.TextField(max_length=5000, null=True, blank=True)
    contentIMGPath = models.TextField(max_length=150, null=True, blank=True)

    class Meta:
        db_table = 'content'


class TwitterAnalysis(models.Model):
    analysis_id = models.AutoField(primary_key=True)
    twitter_hashtag = models.CharField(max_length=200)
    analysis_AddedBy = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    analysis_ApprovalStatus = models.CharField(
        choices=ApprovalStaus, default='pending', max_length=12)
    analysis_approval_remarks = models.TextField(max_length=500, default="N?A")
    analysis_approved_by = models.ForeignKey(
        'AdminModel', null=True, blank=True, on_delete=models.CASCADE)
    analysis_Added_time = models.DateTimeField(auto_now_add=True, null=True)
    analysis_twitter_path = models.TextField(
        max_length=150, null=True, blank=True)
    analysis_neutral = models.CharField(max_length=20, blank=True, null=True)
    analysis_positive = models.CharField(max_length=20, blank=True, null=True)
    analysis_negative = models.CharField(max_length=20, blank=True, null=True)
    analysis_very_nagitive = models.CharField(
        max_length=20, blank=True, null=True)

    class Meta:
        db_table = 'twitter_analysis'


class MeetingConversionModel(models.Model):
    meetingID = models.AutoField(primary_key=True)
    meetingfileLocation = models.CharField(max_length=500)
    meetindAddedBy = models.ForeignKey(
        'User_Model', on_delete=models.CASCADE, null=True, blank=True)
    meeting_ApprovalStatus = models.CharField(
        choices=ApprovalStaus, default='pending', max_length=12)
    meeting_approval_remarks = models.TextField(max_length=500, default="N?A")
    meeting_approved_by = models.ForeignKey(
        'AdminModel', null=True, blank=True, on_delete=models.CASCADE)
    meeting_Added_time = models.DateTimeField(auto_now_add=True, null=True)

    meetinfileChunks = models.CharField(
        max_length=2000, null=True, blank=True, default="NOVALUVE")
    meetingTransscription = models.CharField(
        max_length=1500, blank=True, null=True)

    def set_chunks(self, chunks):
        self.meetinfileChunks = json.dumps(chunks)

    def getListChunks(self):
        return json.loads(self.meetinfileChunks)

    class Meta:
        db_table = 'meeting-conversations'
