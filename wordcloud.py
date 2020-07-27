# word cloud
"""
Created on Sun Jul 26 20:48:04 2020

@author: kamakshi Gupta
"""
#pip install WordCloud
from wordcloud import WordCloud

import matplotlib.pyplot as plt
 
# Create a list of word
text = ("Yoshita Richa Ayushi Richa Ayushi Swarna Yoshita Aishwarya Akanksha Indira Sushmita ")
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
