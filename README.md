# MagiClass

## Table of contents
* [General information](#general-information)
* [Modules](#modules)
* [Technical requirements](#technical-requirements)
* [Status](#status)
* [Contributing](#contributing)
* [Author](#author)
* [License](#license)


## General information

MagiClass is a Machine Learning Flask app that performs preprocessing, modeling and text classification tasks so to deliver a tagged dataset via its API.
 
The current version is focused on English corpora and binary classification (e.g. spam vs. not spam).


## Routes

MagiClass offers 3 sub-endpoints which need a directory name and a file name to be specified:
<ul>
<li><strong>.../preprocess/{directory}/{filename}</strong><br>
the name of the folder followed by the name of the csv file to be preprocessed.</li>
<li><strong>.../model/{directory}/{filename}</strong><br>
the name of the folder followed by the name of the clean json file to be split as train/test dataset.</li>
<li><strong>.../classify/{directory}/{filename}</strong><br>
the name of the folder followed by the name of clean json the file to be tagged.</li>
</ul>


## Modules


### 1. Preprocessing

This first task allows two types of preprocessing through the parameter 'pipe':
<ul>
<li><strong>'min'</strong>: .../preprocess/{directory}/{filename}?<strong>pipe=min</strong><br>
cleaning (i.e. remove urls, emails, phone numbers, special characters) and language detection only</li>
<li><strong>'max'</strong>: .../preprocess/{directory}/{filename}?<strong>pipe=max</strong><br>
metadata extraction (urls, emails, phone numbers), emphasis markers (series of capital letters, exclamation and question marks) performed before cleaning and language detection so to allow further analysis</li>
</ul>

<strong>Default: 'min'.</strong>

Output: json ; also save a json file (clean_{filename}.json) which is stored in 'data/{directory}/dataset/'.


### 2. Modeling

This second task includes mandatory parameters:
<ul>
<li><strong>'target'</strong>: the name of the field that contains the dependant variable - that is Y<br>
E.g. .../model/spam/train?<strong>target=category</strong></li>
<li><strong>'target_value'</strong>: the value to be encoded 1<br>
E.g. .../model/spam/train?target=category&<strong>target_value=spam</strong></li>
<li><strong>'factor'</strong>: the name of the field that contains the independant variable - that is X<br>
E.g. .../model/spam/train?target=category&target_value=spam&<strong>factor=text</strong></li>
</ul>

Customizable parameters are listed below:
<ul>
<li><strong>'narrow_lang'</strong>: reduce the dataset to a specific language only, depending on the 'lang' field generated through the preprocessing phase<br>  
E.g. .../model/spam/train?<strong>narrow_lang=en</strong><br>
<strong>Default: 'none'</strong></li>
<li><strong>'lemmatizer'</strong>: if True, lemmatize the text designated as factor using spaCy (En)<br>
<strong>Default: False</strong></li>
<li><strong>'vectorizer'</strong>: choose between TfidfVectorizer ['tfidf'] and CountVectorizer ['bow'] to be included in the modeling pipeline<br>
E.g. .../model/spam/train?<strong>vectorizer=bow</strong><br>
<strong>Default: 'tfidf'</strong></li>
<li><strong>'resampler'</strong>: if True, resample the dataset in case the target distribution (sum of target value / sum of not target value) is less than 0.85 or more than 1.15<br>
<strong>Default: True</strong></li>
<li><strong>'classifiers'</strong>: define one ore more classifiers to be run - available are Naive Bayes algorithms: Multinomial ('MBN'), Bernoulli ('BNB') and Complement ('CNB')<br>
E.g. .../model/spam/train?</strong>classifiers=BNB+MNB</strong><br>
<strong>Default: 'BNB+CNB+MNB'</strong></li>
<li><strong>'mode'</strong>: if <strong>'single'</strong>, one classifier has to be specified in the 'classifiers' parameter so the model can be saved with pickle in 'data/{directory}/model/'; otherwise, the listed classifiers are compared across various metrics<br>
<strong>Default: 'benchmark'</strong></li>
<li><strong>'metric'</strong>: define the core metric to select the best model among <strong>'accuracy', 'roc_auc_score', 'f1', 'precision' and 'recall'</strong><br>
<strong>Default: 'f1'</strong> (especially relevant in a binary classification problem)</li>
</ul>

Output: json ; also save a json file (benchmark.json) which is stored in 'data/{directory}/model/'.

The output provides useful information to evaluate models as well as the quality of the dataset. In addition to the metrics mentioned above, there are given the train accuracy compared with the test accuracy as an <strong>overfitting indicator</strong>, the size of the train dataset, and the target ratio as an <strong>imbalance indicator</strong>.

Hyperparameters for Naive Bayes algorithms are automatically tuned using <strong>GridSearchCV</strong>.


### 3. Classification

This final task delivers the tagged dataset via .../classify/{directory}/{filename}. There is no need to define paramaters: the ones used in the modeling phase are automatically retrieved from the 'benchmark' json file.

Output: json


## Technical requirements

The application is built with <strong>Python</strong> 3.8 and <strong>Flask</strong> 1.1.2. 

Libraries include <strong>Pandas, LangDetect, spaCy, Scikit-Learn</strong>.

```bash
pip3 install -r requirements.txt
```


## Status

This project is in progress.


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## Author

* Initial work: Stephanie BLANCHET, Data Scientist & Python Developer.
* stephanie.blanchet.it@gmail.com


## License

This project is licensed under the MIT License - see [MIT](https://choosealicense.com/licenses/mit/) for details.
