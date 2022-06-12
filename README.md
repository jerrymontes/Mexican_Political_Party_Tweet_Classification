# Mexican_Political_Party_Tweet_Classifier
**Project by: Jerry Montes**

## Problem Statement

I sought to build a multi-class classification model that, when presented with a tweet from one of the 6 official Twitter accounts associated with Mexican political parties, is capable of determining which party posted it.  **My success criteria involved developing a model that outperformed a baseline model whose accuracy rate was determined by the proportions of each class in the data.**

### Background

Seven national political parties are currently registered in Mexico, which is a federal presidential republic.  From oldest to newest, these are: Partido Acción Nacional (PAN), Partido Revolucionario Institucional (PRI), Partido de la Revolución Democrática (PRD), Partido del Trabajo (PT), Partido Verde Ecologista de México (PV), Movimiento Ciudadano (MC), and Movimiento de Regeneración Nacional (Morena).  A 2021 study found that the majority of Mexicans do not trust their government, whose presidential office was held by the same political party for 71 years straight, whose ex-presidents' alleged wrongdoing would have been investigated following 40% participation in a 2021 referendum, and whose scandals include a governor pleading guilty to embezzling public funds meant for the purchase of chemotherapy for use in state-run hospitals.

When surveyed about what they felt was the primary problem the country was currently facing, the vast majority of respondents reported that Crime was the country's biggest problem.  In fact, three out of every four Mexicans consider their city to be unsafe.  In 2020 around 3,700 women were murdered in Mexico, which is experiencing a femicide crisis.  Corruption was the second most common answer Mexicans gave when asked about what they felt was the primary problem afflicting the country.  Three out of every five Mexicans believe that the majority of police are corrupt, although law enforcement is certainly not the only branch of government that is largely distrusted.  In 2014, public officials submitted false declarations that stated that a Mexico City school's existing walls and floors were being renovated.  In reality its fourth floor, where the school's owner resided, was being expanded without any consideration for whether the building had been built to withstand that additional weight.  This school (which contained the owner's apartment and its marble flooring and bathtub) collapsed during a September 2017 earthquake, killing 19 children and eight adults and serving as an example of how corruption can have deadly consequences.  Given that 44% of the population lives in poverty, it is not surprising that the third most common answer given by Mexicans who were asked about their country's biggest problem involved the economy.  

Mexican political parties certainly have one thing in common: they agree that Mexico needs change.  The proposed path forward differs between parties, along with the specific focus of each party's agenda.  

Six of the seven political parties currently registered with the country's National Electoral Institute (INE) have an official Twitter account.  What follows is a brief description of the six parties with Twitter accounts.
  

## Political party classes:

- To understand where on the spectrum each political party falls, please refer to this 2017 graphic: https://img.buzzfeed.com/buzzfeed-static/static/2017-12/24/21/asset/buzzfeed-prod-fastlane-02/sub-buzz-2594-1514170795-5.png
- "izquierda económica" means **economic left**
- "derecha económica" means **economic right**
- "izquierda política" means **political left**
- "derecha política" means **political right**

- It is worth noting that two of the political parties in the graphic have since lost their registered status after obtaining less than 3% of the vote in 2018 elections. 

**Partido Acción Nacional (National Action Party):** 
This party was registered on March 30th, 1946.  Its current slogan (in English) is "Action or Mexico"

**Partido Revolucionario Institucional (Institutional Revolutionary Party):**
This party was registered on March 30th, 1946.  It held the presidential office for 71 consecutive years, and its current slogan (in English) is "Revolutionaries"

**Partido de la Revolución Democrática (Party of the Democratic Revolution):**
This party was registered on May 26th , 1989.  Its current slogan (in English) is "A new dawn"

**Partido Verde Ecologista de México (Ecological Green Party of Mexico):** 
This party was registered on January 13th, 1993.  Its current slogan (in English) is "Green gets it done"

**Movimiento Ciudadano (Citizens’ Movement):**
This party was registered on June 30th, 1999.  Its current slogan (in English) is "The future is orange"

**Movimiento de Regeneración Nacional (National Regeneration Movement):**
This party was registered on July 9th, 2014.  It currently holds the presidential office, and its current slogan (in English) is "Mexico's hope"


### Approach

I implemented both scikit-learn algorithms and neural networks built using the TensorFlor API to classify text from the tweets obtained.

#### Notebook/Code

|Index|Title|Description|
|---|---|---|
|1|[01_Data_Collection]() | This notebook contains all of the code utilized to pull the 3,200 most recent tweets from the 6 official Twitter accounts associated with Mexican political parties using Twitter's **Twitter API v2**.
|2|[02_Data_Cleaning_& EDA]() | This notebook contains all of the Data Cleaning and Exploratory Data Analysis performed on the data.|
|3|[03_Preprocessing_& Scikit_Learn_Modeling]() | This notebook contains all of the Scikit-learn model development.|
|4|[04_Preprocessing_& Tensorflow_Modeling]() | This notebook contains all of the Tensorflow neural network model development.|

#### Exploratory Data Analysis

During the EDA I performed, I calculated the number of observations of each class in the DataFrame created by concatenating each class' individual DataFrame.  I confirmed that five classes each constituted 16.7% of the data and one class constituted 16.6% of the data.  The largest class made up 16.694% of the data, while the second largest class made up 16.688% of the data. 

I then explored each tweet's word count and character length and noted that the tweets with the shortest word count typically consisted only of links that were shared by each user.  I also noted that the tweets with the shortest character length consisted primarily of emojis.  Plotting the word count distribution revealed a normal distribution, and plotting the character length distribution revealed that around a third of the tweets consisted of roughly 140 characters. 

An anaysis of each party's 10 most common unigrams, bigrams, trigrams, 4-grams, and 5-grams using a Spanish-language stop words list revealed that some of the most common words used by the political parties include political party names, their party's slogan, and the current president's name.  In order to analyze the 10 most common words and sets of words used by each party that did not include these common political references, I created another stop words list that contained all of the stop words from the first list with a few additions.  These included each party's username, acronmyn, slogan, and politicans who appeared in the list of the 10 most used n-grams.  An analysis of the 10 most common n-grams obtained using this second stop words list allowed for a better understanding of the issues currently afflicting the country that each party mentions most often.  For instance, the word "women" appeared on the list of 10 most common unigrams for three of the six political parties with official Twitter accounts.  The unigram "covid19" and the bigram "covid 19" appeared on the list of only one party's 10 most common unigrams and bigrams.  Only one of the parties' list of 10 most common unigrams contained the unigram "violence," which might come as a surprise given that crime is the problem cited the most by Mexicans asked about the country's most critical problems.

#### Modeling

#### Scikit-Learn
When developig my scikit-learn models, I created a baseline model using a DummyClassifier object who accuracy scores corresponded to the proportion of the largest class in the data.  I then created a vectorized representations of each tweet using a CountVectorizer object and the original Spanish-language stop words list before creating another matrix using a CountVectorizer object and the second Spanish-langauge stop words list.  To both of the two vectorized representations of text I then applied the following classification algorithms: 
- Logistic Regression
- KNearest Neighbors
- Support Vector Classifier
- Multinomial Naive Bayes
- Decision Tree Classifier
- Extra Trees Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

The model with the highest accuracy score was developed using a Counvectorizer, the original stop words list, and a Multinomial Naive Bayes Classifier.

#### Tensorflow

When developing my Tensorflow model, I utilized a Hugging Face tokenizer that had been trained to classify Spanish news headlines.  Although the prebuilt model was developed using Pytorch, I was able to obtain its weights using TFAutoModelForSequenceClassification.

My Final Model Architecture consisted of:
  - Sequential
  - Input Layer:
    1. BERT Model
    2. Pooling
  - Hidden Layer 1:
    1. Dense
    2. ReLU activation
  - Output Layer :
    1. Dense
    2. Softmax activation

**Metrics:**

##### Scikit-Learn 
- I utilized accuracy, precision, recall, and f1 scores for each model.  For the latter three metrics, 'PartidoMorenaMx' was set as the positive label given that it is the party that currently holds the presidential office, and therefore I expected all parties to mention that party the most in their tweets.  Given the balanced classes, I utilized a 'macro' averaging method.

- Scikit-Learn Model's performance -
**Baseline Model**
Test Accuracy: .1669, Test Precision: .0278, Test Recall: .1667, Test F1: .0477

**Best Scikit-Learn Model**
Test Accuracy: .9040, Test Precision: .9055, Test Recall: .9040, Test F1: .9045

##### Tensorflow
- I utilized used a  categorical cross-entropy loss function, Adam optimizer, and focused on accuracy.

- Tensorflow Model’s performance -
Accuracy: 0.9741, Validation Accuracy: 0.9883

**Test both my best scikit-learn model and my best Tensorflow neural network model on my Stremlit app by clicking here:**
(add Streamlit link here)

### Recommendations:

Given that Twitter users often utilize hashtags, which might contain several words without any separation between them, the CountVectorizer object classified each hashtag as a single word when in fact it may have contained several words.  This made it difficult to compare the most common unigrams in each class beacuse some were clearly several words combined into a single hashtag while others really were a single word. 

While the models themselves employed the use of either a Spanish-language stemming tokenizer or a lemma tokenizer depending on which was identified as the best parameter by the RandomizedSearchCV, both of the Spanish stop words lists did not contain words in their lemma or stem form.  This means that because the words in the text were transformed to either their lemma or stem, the CountVectorizer was unable to stop a word in the text that may originally have matched a word on the stop words list prior to being transformed to its lemma or stem from being included in the vectorized representation of the text.  This means that the scikit-learn models would have had a more difficult time learning the differences between the corpora given the frequent appearance of stop words in their lemma or stem form across all classes.

### Conclusions:

As of 2021, every person born to someone who themselves was born in Mexico is able to obtain Mexican citizenship without residing in Mexico and take part in national elections.  This means that it is important for every person of Mexican descent to be able to understand the message that each political party is displaying and whether or not there is a discrepancy between the issues they claim to care about most and the issues they actually mention the most on Twitter.

In a multi-party
__________


### Sources
- https://www.cia.gov/the-world-factbook/field/government-type/
- https://www.theatlantic.com/international/archive/2012/07/party-ruled-mexico-70-years-returns-power/326386/
- https://www.eleconomista.com.mx/politica/7-datos-sobre-la-corrupcion-en-Mexico-segun-Transparencia-Internacional-20171014-0001.html
- https://www.elfinanciero.com.mx/opinion/alejandro-moreno/el-problema-de-la-inseguridad/


### Images

- mexican_flag: https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Flag_of_Mexico.svg/1200px-Flag_of_Mexico.svg.png
- usa_flag: :https://upload.wikimedia.org/wikipedia/en/thumb/a/a4/Flag_of_the_United_States.svg/1280px-Flag_of_the_United_States.svg.png
- pri_logo: https://s3.amazonaws.com/s3.timetoast.com/public/uploads/photos/2509205/pri.jpg?1334977688
- gortari: https://yt3.ggpht.com/a/AATXAJwv0AbFNeg17Vwe0sWbxUYTzRS_WpyNY4E70A=s900-c-k-c0xffffffff-no-rj-mo
- duarte: https://mananarm.blob.core.windows.net.optimalcdn.com/images/2015/08/10/javierduarte-focus-0-0-628-524.jpg
- jail_time: https://cdn.laoctava.com/app/uploads/2020/10/8012687659001-scaled.jpg
- travel_advisory: https://travelmaps.state.gov/TSGMap/?extent=-124.207939566,14.44327709,-84.313397286,33.446969624
- femicide: https://www.refinery29.com/images/10192406.png?crop=40:21
- rebsamen: https://www.buzzfeed.com/mx/rafaelcabrera/estas-imagenes-muestran-como-fue-la-ampliacion-del-colegio
- mexico_minimum_wage: https://cdn.vox-cdn.com/thumbor/V1HSR6hyvI0anxP-mw2xeZcKDjE=/0x0:1800x1800/1200x0/filters:focal(0x0:1800x1800):no_upscale()/cdn.vox-cdn.com/uploads/chorus_asset/file/15990768/Mexico_minimum_wage.jpg
- prd_logo: http://www.lavozdetamaulipas.mx/wp-content/uploads/2019/02/logo.png
- pan_logo: https://mexico.quadratin.com.mx/www/wp-content/uploads/2014/11/PAN-LOGO-1.jpg
- pv_logo: https://pvemcolima.org.mx/wp-content/uploads/2020/07/Logo-partido-verde-2020.png
- mc_logo: https://www.brandemia.org/sites/default/files/inline/images/logo_movimiento_ciudadano-nuevo.jpg
- morena_logo: https://laextra.mx/wp-content/uploads/2018/11/MORENA-logo.png
- political_spectrum: https://www.buzzfeed.com/mx/yuririaavila/los-politicos-se-tachan-de-izquierda-o-derecha-pero-eso-que