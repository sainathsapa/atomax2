# Generated by Django 4.1.7 on 2023-03-04 19:18

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0018_approvalmodel_approval_remaks'),
    ]

    operations = [
        migrations.CreateModel(
            name='ContentModal',
            fields=[
                ('contentID', models.AutoField(primary_key=True, serialize=False)),
                ('contentType', models.CharField(max_length=50)),
                ('contentApprovalStatus', models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')], default='pending', max_length=12)),
                ('contentApproval_remaks', models.TextField(max_length=500)),
                ('contentAdded_time', models.DateField(auto_now_add=True, null=True)),
                ('content_text', models.TextField(max_length=5000)),
                ('contentIMGPath', models.TextField(max_length=150)),
                ('contentAddedBy', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='summary.user_model')),
                ('contentApprovedBy', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='summary.adminmodel')),
            ],
            options={
                'db_table': 'content',
            },
        ),
    ]
