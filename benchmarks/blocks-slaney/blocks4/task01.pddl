(define (problem BW-4-5774-1)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 - block)
    (:init
        (handempty)
        (on b1 b2)
        (on-table b2)
        (on-table b3)
        (on b4 b1)
        (clear b3)
        (clear b4)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b3)
            (on-table b3)
            (on-table b4)
        )
    )
)