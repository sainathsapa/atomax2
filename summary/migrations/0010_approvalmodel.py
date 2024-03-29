# Generated by Django 4.1.3 on 2022-11-18 06:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0009_remove_summarymodel_adv_avaibale_trans_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ApprovalModel',
            fields=[
                ('approvalID', models.AutoField(primary_key=True, serialize=False)),
                ('approvalType', models.CharField(max_length=50)),
                ('approvalStatus', models.BooleanField(default=False)),
                ('approval_type_id', models.CharField(max_length=50)),
                ('approval_time', models.DateField(auto_now_add=True, null=True)),
                ('approvedBy', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='summary.adminmodel')),
            ],
            options={
                'db_table': 'approvals',
            },
        ),
    ]
