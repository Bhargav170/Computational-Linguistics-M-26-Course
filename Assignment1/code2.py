with open ("srilm_clean_sentences.txt","r") as reading_doc, open ("bnc_test.txt","w")as writing_doc:
    for sentence in reading_doc:
        sentence=sentence.lower()
        writing_doc.write(sentence)

    