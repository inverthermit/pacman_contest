(define (problem pacman-01)
    (:domain pacman_ai)
    (:objects a00 a01 - node)
    (:init
        (connected a00 a01)
        (connected a01 a00)
        (has_food a01)
        (at a00)
    )
    (:goal
        (and
            (at a01)
            (not (at a00))
            (not (has_food a01))
        )
    )
)
