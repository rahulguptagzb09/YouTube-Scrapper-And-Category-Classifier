# YouTube-Scrapper-And-Category-Classifier
YouTube Scrapper And Category Classifier
Scraping Data-<br>
The scraping is done by YouTube Data API V3. The API provides search list function which takes search query as parameter along with other parameters like region, type. This API return result in JSON format. <br>
I wrote a function which uses this API and return a dictionary with column names as keys and content data as values. Through this I was able to get maximum, accurate and relevant results.<br>
The scraping script generates a CSV file from the results.<br>
Text Classification-<br>
For text classification I used one model from each category mentioned in assignment.<br>
1.	From first category, I used SVM model because it was more accurate and scalable. SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces. Support vector machines algorithm categorizes unlabelled data, and is one of the most widely used clustering algorithms in industrial applications.<br>
SVM Accuracy Score:  32.91015625<br>
Precision: 0.329102 <br>
Recall: 0.329102 <br>
F1: 0.329102<br>
2.	From second category, I used shallow NN model because it was based on learning data representations, as opposed to task-specific algorithms. Learning can be supervised, semi-supervised or unsupervised. The NN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output. The NN gives better results on datasets that are not easily separable and are to complicated for naïve algorithms to classify.<br>
Loss: 0.166<br>
Accuracy: 0.941<br>  
F1 Score: 0.789 <br>
Precision: 0.950 <br>
Recall: 0.680<br>
3.	From third category, I used shallow RNN model because in which data can flow in any direction, are used for applications such as language modelling. Long short-term memory is particularly effective for this use. RNNs can use their internal state (memory) to process sequences of inputs. This makes them applicable to tasks such as unsegmented, connected handwriting recognition or speech recognition. RNNs are better at understanding the sequence of text than any other because they does not lose the order of the text.<br>
Loss: 0.464<br>
Accuracy: 0.833<br> 
F1 Score: 0.000 <br>
Precision: 0.000 <br>
Recall: 0.000<br>
