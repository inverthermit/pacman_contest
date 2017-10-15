readme.txt

The methods used in the project are Heuristic Search and Game Theory.

The two agents both uses OffensiveReflexAgent, which is an agent that has both the ability to offense and defense.

Function heuristicSearch() is the main function for implementing aStar heuristic search.
The heuristic function we use is the manhattan distance between points.
This function returns the first step to the closest goal point in the list.

Function gameTheoryCalculation() is the main function for implementing game theory to the game.
It mainly solves the stalement between our team and the enemy.
It chooses the best position on the border of our side which is the best strategy.
The game theory method is triggered when two teams are stucked.
