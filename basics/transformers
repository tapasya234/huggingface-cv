from transformers import pipeline

classification = pipeline("sentiment-analysis")
print(
    classification(
        [
            "I'm excited about the Hugging Face Computer Vision course.",
            "I'm not a big fan of sweet items.",
        ]
    )
)

altTextGenerator = pipeline(task="image-to-text", model="ydshieh/vit-gpt2-coco-en")
print(
    altTextGenerator(
        ["./basics/image7.jpg", "./basics/image8.jpg", "./basics/image9.jpg"]
    )
)
