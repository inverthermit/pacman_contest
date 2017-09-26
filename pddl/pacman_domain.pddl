(define (domain pacman_ai)
   (:requirements :typing)
   (:types node)
   (:predicates
            (has_food ?location -node)
            (is_wall ?location -node)
            (is_ghost ?location -node)
            (is_home ?location -node)
            (is_opponent_pacman ?location -node)
            (at ?location - node)
            (connected ?n1 ?n2 - node)
            (eat_food ?start ?end -node)
            (move ?start ?end -node)
            ;; get the direction of the ghost from 5 steps away and start to avoid it! (do it in eat and move)
	       )
    (:action move
         :parameters (?start ?end -node)
         :precondition (
         and
         (at ?start)
         (connected ?start ?end)
        ;  (not (is_wall ?end))
         )
         :effect (and
         (not (at ?start))
         (at ?end))

      )
    (:action eat_food
        :parameters (?start ?end -node)
        :precondition (
        and
        (at ?start)
        (has_food ?end)
        (connected ?start ?end)
        ; (not (is_home ?end))
        ; (not (is_wall ?end))
        )
        :effect (and
        (at ?end)
        (not (at ?start))
        (not (has_food ?end)))

     )

     (:action eat_pacman
         :parameters (?start ?end -node)
         :precondition (
         and
         (at ?start)
         (connected ?start ?end)
         (is_home ?end)
         (not (is_wall ?end))
         (is_opponent_pacman ?end)
         )
         :effect (and
         (at ?end)
         )
      )





)
