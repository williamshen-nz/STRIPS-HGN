(define (problem BW-3-9804-32)
    (:domain blocksworld)
    (:objects b1 b2 b3 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b3)
        (on-table b3)
        (clear b1)
        (clear b2)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on-table b3)
        )
    )
)