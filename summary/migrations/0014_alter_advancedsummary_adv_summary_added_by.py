# Generated by Django 4.1.3 on 2022-11-18 11:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0013_approvalmodel_approdeval_sent_by'),
    ]

    operations = [
        migrations.AlterField(
            model_name='advancedsummary',
            name='adv_summary_added_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='summary.user_model'),
        ),
    ]