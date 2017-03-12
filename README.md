# CharNER: Character-Level Named Entity Recognition

This work reimplements a model from [[1]] for Named Entity Recognition (NER).

In [[1]], researchers introduce a model for NER. 
As a contribution, the model eliminates the need for feature engineering and external resources, and uses only labeled data. 
Since, features and external resources are usually language specific, eliminating them makes NER language independent. 
The model also offers a potential solution for different problems like part-of-speech tagging. 
It briefly uses characters instead of words to recognize and tag named entities. 
It uses bidirectional Long Short-Term Memories (LSTMs) to convert each character to tag probabilities. 
Then, it converts these probabilities to word level tags with a viterbi decoder. 
The model is tested in 7 different languages, showing similar performance with state-of-the-art techniques.

[1]: http://www.aclweb.org/anthology/C/C16/C16-1087.pdf
