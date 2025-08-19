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
Themed are the preferenses that affect only a theme, and they are grouped by the theme they affect.