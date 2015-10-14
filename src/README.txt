README
------

Define the directories of the input/output files by the global variable workingDirectory.
Input (variables name):
       clickFileName, buyFileName, testFileName 
Output (variable name):
       submissionFile

Due to memory limitations we separated the flow to 2 phases:
Phase 1- classification task:
main(testRunFlag=False) runs the following procedures:
1. loadData() – load the click and buy training source files.
2. extractFeatures() – preprocessing (parsing, feature extraction and feature engineering for the aggregated sessions).
3. createLabelForClassification() – add the label for each session (if exist in the ‘buy’ source file= 1, else= 0). 
4. dataPreparation() – scaling the features’ values.
5. classifySessions() – performs supervised learning over the processed training set and export the model to file.

Phase 2- recommender task:
main(testRunFlag=True) runs the following procedures:
1. loadData() – load the test source file.
2. extractFeatures() – preprocessing (parsing, feature extraction and feature engineering for the aggregated sessions).
3. dataPreparation() – scaling the features’ values.
4. predict() -  load the classification model from the first phase and predict which of the test sessions will buy.
5. TimeSpentRecommender() – recommend the items each of the predicted sessions will buy (by the percentage of time spent viewing each item in session).

