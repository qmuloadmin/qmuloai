# Qmulo AI
Qmulo is an extremely limited (and simplistic) API format for conversational AI. I'm primarily making this for my own 
purposes and use.

This repo currently consists of the client, a TUI application. 

The TUI has the following goals:

1) Conversational, multiline chat 
    1) State saved to disk
2) RAG of local files and code projects
   1) RAG vector database is local, not remote
   2) Embeddings are processed locally
3) Commands for modifying the chat
   1) Use vector similarity search to eliminate need to know the commands
   2) "Slash" style commands, like slack

## Technologies Used

This currently uses fastembed.rs for embedding models. qdrant is used for vector similarity searches and RAG lookups. 

The intent is to keep as much local as possible, and only the large models are running on the remote server.
