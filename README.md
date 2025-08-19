# ZeroChat
An AI companion that has human like memory, understands, remembers and knows you. Can perform various tasks, but the main focus is the memory and chat System.

---

# Architecture:

---

ZeroMain.py 

The main loop: loads the LLM, sends the messenge, gets the result. 

---

ZeroIdentityCloud.py

Is the ChromaDB identity clouds Vector Space.

It stores all the preferenses/character traits of the "self" of the AI. 

The preferenses are clasified into 2 types:
Global 
Themed

The global ones are the preferenses that affect all the desisions and answers.
Themed are the preferenses that affect only a theme, and they are connected to the theme anchor. A preferense can be connected to multiple themes. 

---

ZeroInformationCluster.py

Stores all the facts, raw facts, connected to each other via similarity edges.

Search happenes via getting all the facts that have a greater similarity to the querrie then n%, and then getting the greater then n% similarity for each one of them, and so on for j times. (actual numbers need some testing)

---

ZeroUserData.py

Stores all the facts about user that the user told about. These are then grouped by importanse, and retrieved ether by similarity or by importanse. 

---

ZeroContextMemory.py

Each user messenge gets stored here, and gets the metadata "User". The reply of the AI is also stored, and is connected to its corresponding prompt via edge. Then, the User inputs are interconnected if the similarity is more then n%, and the AI replies are interconected if the similarity is more then n%.

The decay mechanic:

Each memory also gets a Time To Tive number asigned when inputed into the context memory. 

