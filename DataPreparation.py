#%%libraries 
import email #for extract emails' body

import pandas as pd # create new a data fram

import re # for labeling
#%%
class DataPreparation:

    def __init__(self, df):
        self.df = df

    # to extract emails' body
    #https://www.kaggle.com/sudhirrd007/data-cleaning-and-transformation
    def bodyExtraction(self, messages):
        column = []
        for message in messages:
            e = email.message_from_string(message)
            column.append(e.get_payload())
        return column


    # to label data... 
    def labeling(self, messages):
       
        angerVoc = "difficult|issues|issue|wrong|goddammit|damned|hell|bloody|arguably|full stop|drop dead|angry|sodding|attitude problem|bad|issue|worn down|painful|nauseum|incompetent|accompanied|complaint|grievance|complainant|lawsuit|protest|ugly|monstrous|makes me sick"
        sadnessVoc = "sadness|sad|insult|annoy|sorry|problems|problem|regret|sad|down|upset|miserable|feeling under the weather|I’m blue|disappointed|frightened|gloomy|hurt|anguish"
        fearVoc = "concerned|fear|nervous|phobia|danger|threat|horror|panic|scare|terror|stress|tension|dismay|panic|dread"
        joyVoc= "approval|attractive|happy|attracted|best|good|assure|fun|play|successful|thank|thanks|birthday|transferred|happy|nice|great|lovely|thank God|good for|pleased|Thanks|thanks"
        loveVoc = "love|adore|cherish|soulmate|heart|apple of my eye|rock my world|affection|enchant|fancy|passion|sweetheart|sweetie|yearning"
        surpriseVoc = "really!|is that a fact?|you would not believe|surprise|never expected it"

        sentiment = []
        for row in messages:
            if re.findall(angerVoc, row) :         sentiment.append('anger')
            elif re.findall(sadnessVoc, row):      sentiment.append('sadness')
            elif re.findall(fearVoc, row):         sentiment.append('fear')
            elif re.findall(joyVoc, row):          sentiment.append('joy')
            elif re.findall(loveVoc, row):         sentiment.append('love')
            elif re.findall(surpriseVoc, row):     sentiment.append('surprise')
            
            else:                                  sentiment.append('normal')
        
        return sentiment

    
    #creating a separate dataset
    def newData(self, columnOne, columnTwo):
        
        Newdf = pd.DataFrame(index=range(0,len(self.df)),columns=['Email', 'Sentiment'])

        for i in range(0,len(self.df)):
            Newdf['Email'][i] = columnOne[i]
            Newdf['Sentiment'][i] = columnTwo[i]

        return Newdf
           


# %%
