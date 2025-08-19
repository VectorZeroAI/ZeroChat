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


The decay funktion:

Each memory also gets a Time To Live number asigned when inputed into the context memory.
The TTL is the "current internal clock time + 10".

Each time a user sends a messenge to the AI, the Internal Clock advanses by 1. 

Each memory also gets a timestamp in the internal units time. 

Each time a memory is retrieved, its TTL is increased by 2. 

When a threashhold for the memory size is hit, all the memories that have the TTL smaler then the current internal clock time are deleted.


The retreaval funktion:

When a users messenge is recived, 3 most similar user messenges are retrieved with their corresponding AI answer. 

And 

When a users messenge is recieved, 3 of the newest memories are retrieved too. 


Notes:

A memory is user messenge and its corresponding reply.


