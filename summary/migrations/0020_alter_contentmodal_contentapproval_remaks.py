# Generated by Django 4.1.7 on 2023-03-04 19:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0019_contentmodal'),
    ]

    operations = [
        migrations.AlterField(
            model_name='contentmodal',
            name='contentApproval_remaks',
            field=models.TextField(default='N?A', max_length=500),
        ),
    ]
