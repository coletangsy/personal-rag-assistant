1. Content relevancy 
Add the "Ranking Agent", based on the retrived contents and rank the relavency of the content, route back to "Research Agent" if the contents are in a low-quality
    - add the agent
    - update the graph structure
    - potential needed enhancement on the embedding strategy

2. Output safety
Not only consolidate the final answer, also review if any data safety issues (credentials, personal information, etc included)
    - update the agent instructions
    - potential assign to another agent if it is needed

3. Frequent asked questions
If there is a set of questions frequent asked/ asked before, return the previous questions (this requires structure change)
    - have the first step goes to the space and review
    - add the news asked questions into the space

4. Code refactor
Potential to split all the agent related code into a seperate python file.