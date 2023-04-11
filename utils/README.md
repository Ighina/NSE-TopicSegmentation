# Data Loading Utility Functions

This folder contains the functions used to load a variety of different datasets in the format required by train_fit.py function.

# Adding your own data

If you want to add your own data, you can expand the load_dataset function inside load_datasets.py so that it returns the data in a pre-defined format.
The output of the function should be a list of lists of the form \[\[\[train document list\], \[test document list\], \[validation document list\]\]\] if not using cross validation, alternatively the output should resemble the format \[\[\[train document list 1\], \[test document list 1\], \[validation document list 1\]\], \[\[train document list 2\], \[test document list 2\], \[validation document list 2\], ..., \[\[train document list k\], \[test document list k\], \[validation document list k\],\]\]\] for a definable number of folders or splits k.

Each document list (see above train/test/validation document lists) should be a list of tuples such that d\[0\] = \[list of document's sentences\], d\[1\] = \[text segmentation target (see later)\] and d\[2\] = document identifier.

The text segmentation target is a list of binary values 1 and 0, where 0 means that no topic boundary is present at the corresponding sentence index and, conversely, 1 indicates the presence of a topic boundary corresponding to the sentence having the same index in the document. In our experiments, the topic boundaries (i.e. 1s) were placed at the index of sentences concluding a topically coherent segment, but placing the topic boundaries at the index of sentences starting a topically coherent segment should work as well.

To summarise, let's suppose we have three documents:

- doc1: " The UK is also considered a part of the "Big Four", or G4, an unofficial group of four European nations. It was a member state of the European Communities (EC) and its successor, the European Union (EU), from its accession in 1973 until its withdrawal in 2020.\n\n In 43 AD, Britannia referred to the Roman province that encompassed modern day England and Wales. Great Britain encompassed the whole island, taking in the land north of the River Forth known to the Romans as Caledonia in modern Scotland (i.e. "greater" Britain). In the Middle Ages, the name "Britain" was also applied to a small part of France now known as Brittany. As a result, Great Britain (likely from the French "Grande Bretagne") came into use to refer specifically to the island, with Brittany often referred to as "Little Britain"."

- doc2: "She pulled out of the election after being unable to gain the necessary endorsement of 100 MPs, allowing Rishi Sunak to become Conservative leader and Prime Minister unopposed. Sunak later retained Mordaunt in his cabinet, continuing as leader of the House of Commons.\n\n Penelope Mary Mordaunt was born on 4 March 1973 in Torquay, Devon. The daughter of a former paratrooper, she states she was named after the Arethusa-class cruiser HMS Penelope."

- doc3: "However, on January 10, 2023, Stephanie announced her resignation as co-CEO and Chairwoman, with Nick Khan being the sole CEO and Vince returning as Executive Chairman.\n\n The event included matches that resulted from scripted storylines, where wrestlers portrayed heroes, villains, or less distinguishable characters in scripted events that built tension and culminated in a wrestling match or series of matches. Results were predetermined by WWE's writers on the Raw and SmackDown brands, while storylines were produced on WWE's weekly television shows, Monday Night Raw and Friday Night SmackDown."

We name the collection "NewDocs", where in each document "\n\n" indicates a change of topic. We use doc1 as training set, doc2 as validation set and doc3 as validation set and we do not need to include cross validation splits as an option. 
Then, inside load_datasets.py add a condition such as:

```
elif dataset=="NewDocs": # you can decide any arbitrary name, as you will pass this as an argument to train_fit.py
    doc1_targets = []
    doc1_sentences = nltk.sent_tokenizer(doc1) # we encourage the use of nltk built in punkt tokenizer for sentence tokenization of english
    for sentence in doc1_sentences:
        if sentence.startswith("\n\n"): # the use of \n\n to separate topic is just an example. In real world case, we would want to get rid of these symbols at this stage
            doc1_targets[-1] = 1
        doc1_targets.append(0)
    
    # repeat for doc2 and doc3

    return [[[(doc1_sentences, doc1_targets, "doc1")], [(doc2_sentences, doc2_targets, "doc2")], [(doc3_sentences, doc3_targets, "doc3")]]] # i.e. [[[training set], [test set], [validation set]]]
```

This is the general idea: feel free to personalise the above, but be careful to return the final results in the pre-defined format for compatibility with our training function.