# Generated by Django 4.1.7 on 2023-03-04 19:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0021_alter_contentmodal_contentimgpath_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='contentmodal',
            name='contentQN',
            field=models.CharField(default='test', max_length=200),
            preserve_default=False,
        ),
    ]
