from TextRank import sentence_segmentation, TextRank


text = sentence_segmentation('data/news.txt')
textrank = TextRank(text, similarity_type='origin')
summaries = textrank.get_key_sentences(3)
print(summaries)
