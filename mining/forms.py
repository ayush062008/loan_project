from django import forms
from django.conf import settings

ALLOWED = (".csv", ".xlsx", ".xls")


class UploadForm(forms.Form):
    file = forms.FileField(label="Dataset")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(ALLOWED):
            raise forms.ValidationError("Only .csv and .xlsx files are supported.")
        if f.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise forms.ValidationError(f"File is larger than {settings.MAX_UPLOAD_MB} MB.")
        return f
