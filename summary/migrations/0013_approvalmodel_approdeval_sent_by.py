# Generated by Django 4.1.3 on 2022-11-18 11:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0012_translationsmodel_transtoinsagram_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='approvalmodel',
            name='approdeval_sent_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='summary.user_model'),
        ),
    ]
