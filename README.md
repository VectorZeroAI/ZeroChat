# ZeroChat
An AI companion that has human like memory, understands, remembers and knows you. Can perform various tasks, but the main focus is the memory and chat System.

---

# Architecture:


## Note:
*All the modules follow this API shema: {modulename}.update(messenge, type) and {modulename}.get(user_messenge)*
*type can be ether ai or user.*
*the get method provides the information taht should be recalled when answering the messnge of the user, effectively providing memory.*
*the update method is used to provide new entries to the modules database/backend.*
*This ensures taht modules can be effortlessly plugged into a different system with no complications.*

---

## ZeroMain.py 

The main loop: loads the LLM, sends the messenge, gets the result. 

---

## ZeroIdentityCloud.py

Is the ChromaDB identity clouds Vector Space.

It stores all the preferenses/character traits of the "self" of the AI. 

The preferenses are clasified into 2 types:
Global 
Themed

The global ones are the preferenses that affect all the desisions and answers.
Themed are the preferenses that affect only a theme, and they are connected to the theme anchor. A preferense can be connected to multiple themes. 

ChromaDB is used because the character traits amount tend to get exponentially bigger the more "realistic" the character becomes. 

---

## ZeroInformationCluster.py

Stores all the facts, raw facts, connected to each other via similarity edges.

Search happenes via getting all the facts that have a greater similarity to the querrie then n%, and then getting the greater then n% similarity for each one of them, and so on for j times. (actual numbers need some testing)

The storage backend is lanceDB with the following table layout: 

1. ID (UUID)
2. Text
3. Embedding
4. Metadata (json)
5. Connections (list)

The connections are just UUIDs of the different "facts" stored in the backend. 

---

## ZeroUserData.py

Stores all the facts about user that the user told about. These are then grouped by importanse, and retrieved ether by similarity or by importanse. 

Backend: LanceDB with the following layout: 

1. ID (UUID)
2. Text 
3. Embedding
4. Metadata

---

## ZeroContextMemory.py

A context memery module using LanceDB. 

The colums layout is as following: 

1. ID (UUID)
2. Text (TEXT)
3. Embedding
5. Connections (list)
6. Creation_time (integer) # internal timer
7. TTL (integer) # in the internal time 
8. Type (Binary) # 1 = AI, 0 = user.

Each user messenge gets stored here, and gets the type "User". The reply of the AI is also stored, and is connected to its corresponding prompt via edge. Then, the User inputs are interconnected if the similarity is more then n%, and the AI replies are interconected if the similarity is more then n% .

The Ais messenge and User reply get stored in searate entries, connected via the Connections method.  (adding the UUID of the Users messenge to the list in connections. )

Connections are links by ID. 

### The decay funktion:

Each memory also gets a Time To Live number asigned when inputed into the context memory.
The TTL is the "current internal clock time + 10".

Each time a user sends a messenge to the AI, the Internal Clock advanses by 1. 

Each memory also gets a timestamp in the internal units time. 

Each time a memory is retrieved, its TTL is increased by 2. 

When a threashhold for the memory size is hit, all the memories that have the TTL smaler then the current internal clock time are deleted.


### The retreaval funktion:

When a users messenge is recived, 3 most similar user messenges are retrieved with their corresponding AI answer. 

And 

When a users messenge is recieved, 3 of the newest memories are retrieved too. 


*Clarification:*
*A memory is user messenge and its corresponding reply.*


---

This prototype should only have ZeroContextMemory.py and the core loop.

The LLM provider is operouter for now, with Gemini integration planned later on. 


--- 

# Programm style


Use an prosedural loop for the main file, with OOP for memory funktions, like this: 

`# main-loop`
`while True:`
`  print("your turn")`
`  messenge = input("You:")`
`  if messenge == "Exit":`
`    #exit the programm`
`  else:`
`    pass`
`  context.update(messenge)`
`  #all the other memory subsystems`
`  answer = call_chat(messenge)`
`  print(answer)`
`  context.update(messenge)`

### Explanation

This is used because OOP allows to standartilise the usage of many memory systems in one for loop. 

Although this is just my opinion, but it makes more sense to use Objekts in the memory systems, as to not mix backends. 

Each memory system has a separate backend, wich is also why I desided to use OOP for those. It makes more sense, since its just easier to understand that there are for example 3 objekts, each a different memory subsystem, then to understand 6 get and update funktions. 
Just makes more sense to me.