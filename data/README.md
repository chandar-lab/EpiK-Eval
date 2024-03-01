## Dataset Format
* **`unsegmented_documents`**: All unsegmented documents, each named with a unique id `<id>.txt`.
* **`segmented_documents`**: All segmented documents, each named with a unique id and part number `<id>part<part>.txt`. Documents share the same id in both `unsegmented_documents` and `segmented_documents`. Thus, `segmented_documents/0000part00.txt` and `segmented_documents/0000part01` are the first and second part respectively of `unsegmented_documents/0000.txt`.
* **`qa_examples.csv`**: Contains the training question-answer pairs.
* **`eval_questions.csv`**: Validation and test question-answer pairs.

`names.txt` contains a list of names which is sampled during dataset generation.