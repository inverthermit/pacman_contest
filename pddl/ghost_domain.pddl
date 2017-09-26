(define (domain ghost_ai)
   (:requirements :typing)
   (:types node)
   (:predicates
       (has_food ?location -node)
       (is_wall ?location -node)
       (is_opponent_ghost ?location -node)
       (is_opponent_pacman ?location -node)
       (is_home ?location -node)
       (is_visited ?location)
       (at ?location - node)
       (connected ?n1 ?n2 - node)
       (eat_food ?start ?end -node)
       (move_ghost ?start ?end -node)

            ;; get the direction of the ghost from 5 steps away and start to avoid it! (do it in eat and move)
	       )
    (:action move_ghost ;; Ghost move
         :parameters (?start ?end -node)
         :precondition (
         and
         (at ?start)
         (connected ?start ?end)
         (not (is_opponent_ghost ?end))
         (not (is_wall ?end))
         )
         :effect (and
         (not (at ?start))
         (at ?end)
         (is_visited ?end)
         )
      )
    (:action eat_pacman
        :parameters (?start ?end -node)
        :precondition (
        and
        (at ?start)
        (is_opponent_pacman ?end)
        (connected ?start ?end)
        )
        :effect (and
        (at ?end)
        (not (at ?start))
        (not (is_opponent_pacman ?end)))
        (is_visited ?end)
     )

)
