# Local Transcriber

Turns an audio or video recording on this PC into a transcript. The recording is not uploaded.

## Start

Double-click `Run Transcriber.bat`. That opens the window with this Python interpreter:

`C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe`

A console window stays open while the app is running. Close the transcriber window when you are finished. You can also run:

```
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe transcriber_app.py
```

## Model weights

Whisper weights are cached by Hugging Face on this PC, in:

`C:\Users\User\.cache\huggingface\hub`

The small model is already in that cache, and it is the default. Choosing a model size that is not cached downloads that model once from Hugging Face. That download is the only network use. If `HF_HOME` or `HUGGINGFACE_HUB_CACHE` is set, the cache is there instead.
