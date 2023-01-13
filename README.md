# All_About_Question_Answering
Our practice for question answering in natural language processing 

## Squad2.0 Dataset 
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. 
数据集的文章来源于wikipedia，包含人物、电子产品、城市、宗教等不同主题的词条文章442篇；作为问题的context片段19035个；问题共130319（有答案86821个，无答案43498个）个。 
### Squad2.0 Dataset Structure

📦Squad2.0

 ┣ 📂data

 ┃ ┣ 📂paragraphs
 
┃ ┃ ┣ 📂context

 ┃ ┃ ┗ 📂qas

 ┃ ┃ ┃ ┣ 📂answers

 ┃ ┃ ┃ ┗ 📂question

 ┃ ┗ 📂title

 ┗ 📂version

### Squad V2.0 Description 

* Total answerable question-answer pairs in trainset 86821 
* Total unanswerable question-answer pairs in devset 20302  
* Example
    *Passage:  Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
    Query:  When did Beyonce start becoming popular?
    Answer:  {'text': 'in the late 1990s', 'answer_start': 269}*  

