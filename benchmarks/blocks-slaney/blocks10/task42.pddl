(define (problem BW-10-7268-42)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block)
    (:init
        (handempty)
        (on-table b1)
        (on-table b2)
        (on b3 b5)
        (on b4 b10)
        (on-table b5)
        (on-table b6)
        (on b7 b8)
        (on b8 b6)
        (on b9 b3)
        (on b10 b7)
        (clear b1)
        (clear b2)
        (clear b4)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b10)
            (on b3 b4)
            (on b4 b6)
            (on b5 b9)
            (on-table b6)
            (on b7 b1)
            (on b8 b2)
            (on b9 b3)
            (on-table b10)
        )
    )
)