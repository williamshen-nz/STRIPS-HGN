(define (problem BW-6-4532-3)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 - block)
    (:init
        (handempty)
        (on b1 b4)
        (on-table b2)
        (on-table b3)
        (on b4 b2)
        (on b5 b3)
        (on b6 b1)
        (clear b5)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b4)
            (on-table b3)
            (on b4 b3)
            (on-table b5)
            (on-table b6)
        )
    )
)