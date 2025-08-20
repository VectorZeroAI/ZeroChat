"""
The identity memory part.
It is defined by an Indentity.txt file.
The Indentity.txt file contains all the personality traits the AI should have. 
At the start, an LLM should be called, to turn those plain text parameters into valid .json inputs.
The json inputs should be formated in a similar way to this example.

example:

[
global:{name:katrin
preferenses:("writes in full sentenses";
"explains a lot";
"likes to experiense new things"
)
politics:("hates russia"
"loves opensourse"
)
}
]

The json inputs are then embedded into chromaDB in this way:
The titels (e.g. global, politics) are anchors, and the contents are the embeddings connected to them.

When the identity information is retrieved, everything connected to the node "global" is retrieved, and if thema is provided, everything connected to that thema is retrieved too.
"""