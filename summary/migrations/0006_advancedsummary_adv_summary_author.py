# Generated by Django 4.1.3 on 2022-11-18 06:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0005_rename_adv_summary_advancedsummary_adv_summary_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='advancedsummary',
            name='adv_summary_author',
            field=models.CharField(default='NONE', max_length=50),
        ),
    ]