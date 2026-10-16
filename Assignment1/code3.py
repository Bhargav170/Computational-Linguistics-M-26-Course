with open ("./browndata/brown-train.txt","r") as reading_file, open ("brown-train.txt","w") as writing_file:
    for sentences in reading_file:
        sentences=sentences.lower()
        writing_file.write(sentences)