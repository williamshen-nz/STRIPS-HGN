(define (problem BW-10-7268-52)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b8)
        (on-table b3)
        (on b4 b5)
        (on b5 b10)
        (on b6 b4)
        (on-table b7)
        (on-table b8)
        (on b9 b1)
        (on b10 b3)
        (clear b2)
        (clear b6)
        (clear b7)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b10)
            (on b3 b9)
            (on b4 b1)
            (on b5 b7)
            (on b6 b3)
            (on-table b7)
            (on b8 b4)
            (on b9 b5)
            (on b10 b8)
        )
    )
)