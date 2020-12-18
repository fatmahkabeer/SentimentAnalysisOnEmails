# Emotion detection on emails.

Here, I would like to share my capstone project idea. The idea comes tome because of the problems that I face. The first one is, sometimes when applying for a university or sending a job application, where it important to me, afterward, when I received the reply, I always move my eyes quickly over the email lines trying to know the final decision quickly, I wish there is a tool that told me the final decision quilly in a word. The second problem is, sometimes when I am in a good mood and enjoy my life, suddenly, I read an email that could destroy my day, in such a situation, I also wish if there is a warning message telling me do not open this email now, there is bad news in here. 
This issue is also showing on a Business level. For example, customer service employees receive a bunch of emails every day, and they need to find a way to reply to emails quickly based on the company’s priority. In other words, if a company gives a high priority to fix issues for anger clients, they will need a way to detect the emotion inside each email without the need to open and read each email. 

I used Sentiment Analysis to come up with a solution to those problems. I thought about building an assistance tool based on AI, specifically, natural language processing. To add an emoji beside each email subject so we can detect email-sender emotion.

I used the Enron email dataset. I extract the email body from each row in the Message column then I built a function, with regex assistance, to label emails based on what I expect would be the email sender emotion. 

I implemented Decision Tree Classifier and Logistic Regression, machine learning models, just to know who it would perform. However, I was not satisfied with the results. Then I implemented Bidirectional LSTM, Deep Learning, and I liked its performance. 

I also built a function to check the model. its result was slightly satisfactory even though it did not work well with every single email that I feed it.


The dataset: https://www.kaggle.com/wcukierski/enron-email-dataset


