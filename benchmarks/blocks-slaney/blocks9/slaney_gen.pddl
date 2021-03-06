

(define (problem BW-9-7235-1)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b9)
        (on-table b3)
        (on b4 b7)
        (on b5 b6)
        (on-table b6)
        (on b7 b3)
        (on-table b8)
        (on b9 b8)
        (clear b1)
        (clear b2)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b2)
            (on b2 b4)
            (on b3 b6)
            (on b4 b5)
            (on b5 b8)
            (on-table b6)
            (on-table b7)
            (on b8 b7)
            (on b9 b3)
        )
    )
)


(define (problem BW-9-7235-2)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b9)
        (on b3 b7)
        (on-table b4)
        (on b5 b4)
        (on-table b6)
        (on b7 b6)
        (on b8 b3)
        (on-table b9)
        (clear b1)
        (clear b2)
        (clear b5)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b9)
            (on b3 b5)
            (on-table b4)
            (on b5 b2)
            (on-table b6)
            (on-table b7)
            (on b8 b4)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-3)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b3)
        (on b2 b1)
        (on-table b3)
        (on b4 b5)
        (on b5 b7)
        (on-table b6)
        (on b7 b2)
        (on-table b8)
        (on b9 b8)
        (clear b4)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on-table b3)
            (on b4 b8)
            (on b5 b7)
            (on b6 b9)
            (on b7 b4)
            (on b8 b1)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-4)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on-table b2)
        (on b3 b6)
        (on b4 b2)
        (on-table b5)
        (on b6 b1)
        (on-table b7)
        (on b8 b7)
        (on-table b9)
        (clear b3)
        (clear b4)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on b3 b8)
            (on b4 b1)
            (on b5 b7)
            (on-table b6)
            (on b7 b6)
            (on b8 b5)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-5)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b9)
        (on b3 b8)
        (on b4 b1)
        (on b5 b3)
        (on-table b6)
        (on b7 b5)
        (on b8 b4)
        (on b9 b6)
        (clear b2)
        (clear b7)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on-table b3)
            (on b4 b1)
            (on b5 b7)
            (on b6 b8)
            (on b7 b2)
            (on b8 b4)
            (on b9 b5)
        )
    )
)


(define (problem BW-9-7235-6)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b2)
        (on b2 b5)
        (on b3 b1)
        (on-table b4)
        (on b5 b8)
        (on b6 b4)
        (on-table b7)
        (on-table b8)
        (on b9 b7)
        (clear b3)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b7)
            (on b2 b1)
            (on b3 b9)
            (on b4 b5)
            (on b5 b2)
            (on b6 b3)
            (on-table b7)
            (on-table b8)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-7)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b6)
        (on-table b2)
        (on b3 b2)
        (on b4 b9)
        (on b5 b4)
        (on-table b6)
        (on b7 b8)
        (on b8 b5)
        (on-table b9)
        (clear b1)
        (clear b3)
        (clear b7)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b4)
            (on b3 b9)
            (on b4 b5)
            (on b5 b7)
            (on b6 b3)
            (on b7 b6)
            (on-table b8)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-8)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b7)
        (on b3 b2)
        (on b4 b6)
        (on b5 b4)
        (on b6 b3)
        (on-table b7)
        (on b8 b9)
        (on-table b9)
        (clear b1)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b9)
            (on b2 b6)
            (on b3 b8)
            (on b4 b1)
            (on-table b5)
            (on b6 b4)
            (on-table b7)
            (on b8 b2)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-9)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b4)
        (on b2 b1)
        (on b3 b9)
        (on b4 b8)
        (on b5 b3)
        (on b6 b2)
        (on b7 b6)
        (on-table b8)
        (on b9 b7)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b9)
            (on b2 b3)
            (on b3 b6)
            (on-table b4)
            (on-table b5)
            (on b6 b7)
            (on-table b7)
            (on b8 b2)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-10)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b7)
        (on b2 b1)
        (on-table b3)
        (on b4 b9)
        (on b5 b6)
        (on b6 b8)
        (on b7 b3)
        (on-table b8)
        (on-table b9)
        (clear b2)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b8)
            (on b3 b2)
            (on-table b4)
            (on b5 b7)
            (on b6 b1)
            (on-table b7)
            (on b8 b4)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-11)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b1)
        (on b3 b8)
        (on b4 b2)
        (on-table b5)
        (on b6 b7)
        (on b7 b4)
        (on b8 b6)
        (on b9 b3)
        (clear b5)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b5)
            (on-table b2)
            (on b3 b9)
            (on-table b4)
            (on b5 b4)
            (on b6 b7)
            (on b7 b1)
            (on b8 b6)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-12)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b7)
        (on b3 b9)
        (on-table b4)
        (on b5 b8)
        (on-table b6)
        (on b7 b6)
        (on b8 b3)
        (on b9 b2)
        (clear b1)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b3)
            (on b2 b6)
            (on-table b3)
            (on-table b4)
            (on b5 b2)
            (on b6 b1)
            (on b7 b5)
            (on b8 b4)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-13)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b3)
        (on b3 b5)
        (on b4 b6)
        (on b5 b1)
        (on-table b6)
        (on b7 b4)
        (on b8 b7)
        (on b9 b2)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b5)
            (on-table b4)
            (on b5 b8)
            (on b6 b3)
            (on b7 b6)
            (on-table b8)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-14)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b2)
        (on b2 b4)
        (on b3 b9)
        (on-table b4)
        (on b5 b8)
        (on b6 b3)
        (on-table b7)
        (on b8 b6)
        (on b9 b7)
        (clear b1)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b9)
            (on b3 b2)
            (on b4 b3)
            (on-table b5)
            (on b6 b7)
            (on b7 b8)
            (on-table b8)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-15)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b6)
        (on b2 b7)
        (on b3 b8)
        (on-table b4)
        (on b5 b2)
        (on b6 b5)
        (on b7 b3)
        (on b8 b4)
        (on b9 b1)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b3)
            (on b3 b7)
            (on b4 b5)
            (on-table b5)
            (on-table b6)
            (on b7 b8)
            (on b8 b4)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-16)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b6)
        (on b3 b4)
        (on b4 b9)
        (on b5 b1)
        (on-table b6)
        (on b7 b5)
        (on b8 b7)
        (on b9 b2)
        (clear b3)
        (clear b8)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b1)
            (on b3 b7)
            (on b4 b5)
            (on b5 b3)
            (on b6 b2)
            (on b7 b6)
            (on-table b8)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-17)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on b2 b8)
        (on-table b3)
        (on b4 b7)
        (on b5 b3)
        (on-table b6)
        (on b7 b1)
        (on b8 b4)
        (on b9 b2)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on b3 b1)
            (on-table b4)
            (on-table b5)
            (on-table b6)
            (on b7 b4)
            (on-table b8)
            (on b9 b6)
        )
    )
)


(define (problem BW-9-7235-18)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b6)
        (on b2 b8)
        (on b3 b2)
        (on-table b4)
        (on b5 b4)
        (on b6 b7)
        (on-table b7)
        (on b8 b1)
        (on-table b9)
        (clear b3)
        (clear b5)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b6)
            (on b4 b1)
            (on b5 b9)
            (on b6 b5)
            (on b7 b3)
            (on-table b8)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-19)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b3)
        (on b2 b4)
        (on b3 b7)
        (on-table b4)
        (on b5 b9)
        (on-table b6)
        (on-table b7)
        (on-table b8)
        (on b9 b1)
        (clear b2)
        (clear b5)
        (clear b6)
        (clear b8)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b5)
            (on b3 b6)
            (on b4 b3)
            (on b5 b1)
            (on b6 b2)
            (on-table b7)
            (on b8 b7)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-20)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b9)
        (on b2 b7)
        (on-table b3)
        (on b4 b2)
        (on b5 b8)
        (on-table b6)
        (on-table b7)
        (on b8 b4)
        (on b9 b3)
        (clear b1)
        (clear b5)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b2)
            (on b2 b5)
            (on b3 b1)
            (on-table b4)
            (on b5 b6)
            (on-table b6)
            (on b7 b3)
            (on b8 b9)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-21)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b9)
        (on b2 b4)
        (on b3 b2)
        (on b4 b8)
        (on b5 b7)
        (on b6 b1)
        (on b7 b6)
        (on-table b8)
        (on-table b9)
        (clear b3)
        (clear b5)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b3)
            (on b3 b9)
            (on b4 b7)
            (on-table b5)
            (on b6 b1)
            (on b7 b5)
            (on b8 b6)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-22)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b2)
        (on b2 b6)
        (on-table b3)
        (on b4 b7)
        (on-table b5)
        (on b6 b5)
        (on b7 b1)
        (on b8 b9)
        (on b9 b4)
        (clear b3)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b8)
            (on-table b2)
            (on-table b3)
            (on b4 b9)
            (on b5 b2)
            (on b6 b5)
            (on-table b7)
            (on b8 b3)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-23)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b4)
        (on b2 b1)
        (on-table b3)
        (on-table b4)
        (on b5 b7)
        (on b6 b8)
        (on b7 b6)
        (on b8 b9)
        (on b9 b2)
        (clear b3)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b4)
            (on b2 b7)
            (on b3 b6)
            (on b4 b5)
            (on b5 b3)
            (on b6 b9)
            (on-table b7)
            (on b8 b2)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-24)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on b3 b6)
        (on b4 b1)
        (on b5 b9)
        (on-table b6)
        (on b7 b8)
        (on b8 b5)
        (on b9 b3)
        (clear b2)
        (clear b4)
        (clear b7)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b1)
            (on b3 b6)
            (on b4 b2)
            (on b5 b8)
            (on-table b6)
            (on b7 b9)
            (on b8 b3)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-25)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b7)
        (on-table b3)
        (on b4 b2)
        (on-table b5)
        (on b6 b9)
        (on b7 b6)
        (on-table b8)
        (on-table b9)
        (clear b1)
        (clear b3)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b9)
            (on-table b3)
            (on-table b4)
            (on-table b5)
            (on b6 b3)
            (on b7 b4)
            (on b8 b2)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-26)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b7)
        (on b2 b4)
        (on-table b3)
        (on b4 b9)
        (on-table b5)
        (on b6 b2)
        (on-table b7)
        (on b8 b6)
        (on b9 b1)
        (clear b3)
        (clear b5)
        (clear b8)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b4)
            (on-table b3)
            (on b4 b3)
            (on b5 b8)
            (on b6 b9)
            (on b7 b2)
            (on b8 b6)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-27)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b3)
        (on b2 b1)
        (on-table b3)
        (on-table b4)
        (on-table b5)
        (on-table b6)
        (on-table b7)
        (on b8 b4)
        (on b9 b7)
        (clear b2)
        (clear b5)
        (clear b6)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on-table b2)
            (on b3 b1)
            (on b4 b5)
            (on-table b5)
            (on b6 b9)
            (on b7 b6)
            (on b8 b2)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-28)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b6)
        (on-table b3)
        (on b4 b7)
        (on-table b5)
        (on-table b6)
        (on b7 b3)
        (on b8 b1)
        (on-table b9)
        (clear b2)
        (clear b4)
        (clear b5)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b3)
            (on-table b2)
            (on b3 b7)
            (on-table b4)
            (on b5 b6)
            (on-table b6)
            (on b7 b5)
            (on-table b8)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-29)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on b3 b6)
        (on b4 b3)
        (on-table b5)
        (on b6 b7)
        (on b7 b1)
        (on b8 b5)
        (on b9 b8)
        (clear b2)
        (clear b4)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b3)
            (on b2 b4)
            (on b3 b2)
            (on-table b4)
            (on-table b5)
            (on b6 b8)
            (on b7 b5)
            (on b8 b9)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-30)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on-table b2)
        (on-table b3)
        (on b4 b1)
        (on b5 b4)
        (on-table b6)
        (on b7 b3)
        (on b8 b6)
        (on b9 b5)
        (clear b2)
        (clear b7)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b7)
            (on b2 b6)
            (on b3 b1)
            (on b4 b3)
            (on b5 b9)
            (on b6 b5)
            (on-table b7)
            (on b8 b2)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-31)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b4)
        (on b3 b1)
        (on b4 b9)
        (on b5 b3)
        (on-table b6)
        (on-table b7)
        (on b8 b7)
        (on b9 b5)
        (clear b2)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b3)
            (on-table b3)
            (on b4 b2)
            (on b5 b4)
            (on b6 b7)
            (on b7 b8)
            (on b8 b1)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-32)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b7)
        (on b2 b1)
        (on-table b3)
        (on-table b4)
        (on b5 b8)
        (on-table b6)
        (on-table b7)
        (on b8 b2)
        (on b9 b6)
        (clear b3)
        (clear b4)
        (clear b5)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b7)
            (on-table b2)
            (on b3 b6)
            (on-table b4)
            (on b5 b1)
            (on-table b6)
            (on-table b7)
            (on b8 b3)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-33)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b7)
        (on b2 b9)
        (on-table b3)
        (on b4 b1)
        (on b5 b2)
        (on b6 b4)
        (on b7 b5)
        (on-table b8)
        (on-table b9)
        (clear b3)
        (clear b6)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b8)
            (on b2 b7)
            (on b3 b6)
            (on-table b4)
            (on-table b5)
            (on-table b6)
            (on b7 b5)
            (on b8 b2)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-34)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b9)
        (on-table b2)
        (on b3 b4)
        (on b4 b5)
        (on b5 b7)
        (on-table b6)
        (on b7 b8)
        (on b8 b1)
        (on-table b9)
        (clear b2)
        (clear b3)
        (clear b6)
    )
    (:goal
        (and
            (on b1 b4)
            (on-table b2)
            (on-table b3)
            (on-table b4)
            (on b5 b6)
            (on-table b6)
            (on b7 b3)
            (on b8 b9)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-35)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b2)
        (on b2 b3)
        (on b3 b7)
        (on b4 b6)
        (on-table b5)
        (on b6 b5)
        (on-table b7)
        (on-table b8)
        (on-table b9)
        (clear b1)
        (clear b4)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b2)
            (on b2 b3)
            (on-table b3)
            (on b4 b9)
            (on b5 b6)
            (on b6 b7)
            (on-table b7)
            (on b8 b4)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-36)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b3)
        (on b2 b1)
        (on b3 b7)
        (on-table b4)
        (on b5 b4)
        (on b6 b8)
        (on b7 b6)
        (on-table b8)
        (on b9 b2)
        (clear b5)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b6)
            (on-table b2)
            (on-table b3)
            (on b4 b2)
            (on b5 b3)
            (on b6 b8)
            (on b7 b1)
            (on-table b8)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-37)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b7)
        (on b3 b9)
        (on b4 b6)
        (on b5 b8)
        (on-table b6)
        (on-table b7)
        (on b8 b3)
        (on-table b9)
        (clear b1)
        (clear b2)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b9)
            (on b3 b6)
            (on b4 b2)
            (on-table b5)
            (on-table b6)
            (on b7 b8)
            (on b8 b1)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-38)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on-table b3)
        (on b4 b8)
        (on b5 b4)
        (on b6 b1)
        (on b7 b6)
        (on-table b8)
        (on b9 b5)
        (clear b2)
        (clear b3)
        (clear b7)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b6)
            (on-table b2)
            (on-table b3)
            (on b4 b9)
            (on b5 b8)
            (on b6 b5)
            (on b7 b3)
            (on-table b8)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-39)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b1)
        (on-table b3)
        (on-table b4)
        (on-table b5)
        (on-table b6)
        (on b7 b2)
        (on b8 b4)
        (on b9 b7)
        (clear b3)
        (clear b5)
        (clear b6)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b6)
            (on b3 b4)
            (on-table b4)
            (on b5 b1)
            (on b6 b5)
            (on b7 b9)
            (on-table b8)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-40)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b1)
        (on-table b3)
        (on-table b4)
        (on-table b5)
        (on b6 b4)
        (on b7 b2)
        (on b8 b5)
        (on b9 b8)
        (clear b3)
        (clear b6)
        (clear b7)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b9)
            (on b2 b3)
            (on b3 b6)
            (on-table b4)
            (on b5 b7)
            (on b6 b4)
            (on b7 b8)
            (on b8 b2)
            (on b9 b5)
        )
    )
)


(define (problem BW-9-7235-41)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b7)
        (on-table b3)
        (on b4 b5)
        (on b5 b1)
        (on-table b6)
        (on b7 b6)
        (on b8 b3)
        (on b9 b4)
        (clear b2)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b3)
            (on-table b2)
            (on b3 b5)
            (on b4 b8)
            (on-table b5)
            (on b6 b2)
            (on b7 b9)
            (on b8 b7)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-42)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b3)
        (on-table b3)
        (on-table b4)
        (on-table b5)
        (on b6 b5)
        (on b7 b1)
        (on b8 b4)
        (on-table b9)
        (clear b2)
        (clear b6)
        (clear b7)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b9)
            (on-table b2)
            (on-table b3)
            (on b4 b7)
            (on b5 b6)
            (on b6 b3)
            (on-table b7)
            (on b8 b5)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-43)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b2)
        (on-table b2)
        (on-table b3)
        (on b4 b1)
        (on-table b5)
        (on b6 b3)
        (on b7 b4)
        (on b8 b7)
        (on b9 b8)
        (clear b5)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b9)
            (on-table b2)
            (on b3 b7)
            (on-table b4)
            (on b5 b6)
            (on b6 b3)
            (on-table b7)
            (on-table b8)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-44)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b9)
        (on-table b2)
        (on-table b3)
        (on b4 b2)
        (on b5 b6)
        (on-table b6)
        (on b7 b5)
        (on b8 b3)
        (on-table b9)
        (clear b1)
        (clear b4)
        (clear b7)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b6)
            (on b2 b7)
            (on b3 b5)
            (on b4 b1)
            (on-table b5)
            (on b6 b9)
            (on b7 b8)
            (on-table b8)
            (on b9 b3)
        )
    )
)


(define (problem BW-9-7235-45)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on-table b2)
        (on b3 b4)
        (on b4 b2)
        (on b5 b6)
        (on b6 b7)
        (on b7 b9)
        (on-table b8)
        (on-table b9)
        (clear b1)
        (clear b3)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b8)
            (on-table b2)
            (on-table b3)
            (on b4 b3)
            (on b5 b9)
            (on b6 b4)
            (on b7 b1)
            (on b8 b2)
            (on b9 b6)
        )
    )
)


(define (problem BW-9-7235-46)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b4)
        (on b3 b9)
        (on-table b4)
        (on b5 b6)
        (on b6 b7)
        (on b7 b3)
        (on b8 b1)
        (on-table b9)
        (clear b2)
        (clear b5)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b7)
            (on b3 b1)
            (on b4 b3)
            (on-table b5)
            (on b6 b4)
            (on b7 b6)
            (on b8 b9)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-47)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on-table b3)
        (on b4 b8)
        (on b5 b4)
        (on b6 b1)
        (on-table b7)
        (on b8 b2)
        (on b9 b7)
        (clear b3)
        (clear b5)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b6)
            (on-table b3)
            (on b4 b3)
            (on b5 b4)
            (on-table b6)
            (on b7 b1)
            (on-table b8)
            (on b9 b8)
        )
    )
)


(define (problem BW-9-7235-48)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b3)
        (on b2 b9)
        (on b3 b5)
        (on b4 b1)
        (on b5 b2)
        (on b6 b4)
        (on b7 b8)
        (on-table b8)
        (on-table b9)
        (clear b6)
        (clear b7)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b6)
            (on-table b3)
            (on-table b4)
            (on b5 b9)
            (on b6 b4)
            (on b7 b1)
            (on b8 b7)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-49)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on b2 b8)
        (on b3 b1)
        (on b4 b6)
        (on b5 b9)
        (on b6 b3)
        (on-table b7)
        (on b8 b4)
        (on b9 b7)
        (clear b2)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b4)
            (on b3 b6)
            (on-table b4)
            (on b5 b9)
            (on b6 b1)
            (on-table b7)
            (on-table b8)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-50)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b8)
        (on b2 b7)
        (on b3 b9)
        (on b4 b3)
        (on b5 b6)
        (on-table b6)
        (on-table b7)
        (on-table b8)
        (on-table b9)
        (clear b1)
        (clear b2)
        (clear b4)
        (clear b5)
    )
    (:goal
        (and
            (on b1 b2)
            (on b2 b7)
            (on-table b3)
            (on b4 b8)
            (on-table b5)
            (on-table b6)
            (on-table b7)
            (on b8 b5)
            (on b9 b4)
        )
    )
)


(define (problem BW-9-7235-51)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b4)
        (on b2 b5)
        (on b3 b1)
        (on b4 b9)
        (on b5 b8)
        (on-table b6)
        (on b7 b2)
        (on-table b8)
        (on-table b9)
        (clear b3)
        (clear b6)
        (clear b7)
    )
    (:goal
        (and
            (on b1 b8)
            (on-table b2)
            (on b3 b4)
            (on-table b4)
            (on b5 b2)
            (on-table b6)
            (on b7 b5)
            (on b8 b7)
            (on b9 b1)
        )
    )
)


(define (problem BW-9-7235-52)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b9)
        (on b2 b6)
        (on b3 b5)
        (on-table b4)
        (on-table b5)
        (on b6 b3)
        (on-table b7)
        (on-table b8)
        (on b9 b4)
        (clear b1)
        (clear b2)
        (clear b7)
        (clear b8)
    )
    (:goal
        (and
            (on-table b1)
            (on b2 b5)
            (on-table b3)
            (on-table b4)
            (on b5 b4)
            (on b6 b2)
            (on b7 b3)
            (on b8 b1)
            (on b9 b7)
        )
    )
)


(define (problem BW-9-7235-53)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on b3 b1)
        (on b4 b5)
        (on b5 b6)
        (on b6 b3)
        (on b7 b2)
        (on-table b8)
        (on b9 b4)
        (clear b7)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b8)
            (on b2 b9)
            (on-table b3)
            (on b4 b6)
            (on b5 b7)
            (on b6 b2)
            (on-table b7)
            (on b8 b5)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-54)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b7)
        (on b2 b3)
        (on-table b3)
        (on-table b4)
        (on b5 b1)
        (on-table b6)
        (on b7 b4)
        (on-table b8)
        (on b9 b5)
        (clear b2)
        (clear b6)
        (clear b8)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on b3 b5)
            (on-table b4)
            (on-table b5)
            (on b6 b1)
            (on b7 b6)
            (on b8 b9)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-55)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b3)
        (on-table b3)
        (on b4 b2)
        (on b5 b1)
        (on b6 b4)
        (on b7 b5)
        (on-table b8)
        (on b9 b8)
        (clear b6)
        (clear b7)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b5)
            (on b2 b4)
            (on b3 b8)
            (on b4 b7)
            (on-table b5)
            (on b6 b9)
            (on b7 b1)
            (on b8 b6)
            (on b9 b2)
        )
    )
)


(define (problem BW-9-7235-56)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on b2 b6)
        (on b3 b1)
        (on b4 b3)
        (on-table b5)
        (on b6 b9)
        (on b7 b8)
        (on b8 b5)
        (on b9 b7)
        (clear b2)
        (clear b4)
    )
    (:goal
        (and
            (on b1 b3)
            (on b2 b9)
            (on b3 b8)
            (on-table b4)
            (on b5 b7)
            (on b6 b5)
            (on-table b7)
            (on b8 b2)
            (on b9 b6)
        )
    )
)


(define (problem BW-9-7235-57)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on b2 b4)
        (on-table b3)
        (on b4 b9)
        (on b5 b2)
        (on-table b6)
        (on b7 b6)
        (on b8 b7)
        (on-table b9)
        (clear b1)
        (clear b3)
        (clear b8)
    )
    (:goal
        (and
            (on b1 b3)
            (on-table b2)
            (on b3 b6)
            (on b4 b8)
            (on-table b5)
            (on b6 b7)
            (on b7 b9)
            (on b8 b5)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-58)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b4)
        (on b2 b8)
        (on b3 b1)
        (on-table b4)
        (on b5 b2)
        (on-table b6)
        (on b7 b3)
        (on b8 b7)
        (on b9 b5)
        (clear b6)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b8)
            (on b2 b3)
            (on b3 b7)
            (on b4 b6)
            (on b5 b2)
            (on b6 b9)
            (on b7 b4)
            (on b8 b5)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-59)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on b1 b5)
        (on b2 b8)
        (on b3 b2)
        (on-table b4)
        (on-table b5)
        (on-table b6)
        (on b7 b9)
        (on-table b8)
        (on b9 b6)
        (clear b1)
        (clear b3)
        (clear b4)
        (clear b7)
    )
    (:goal
        (and
            (on b1 b2)
            (on-table b2)
            (on b3 b4)
            (on b4 b1)
            (on b5 b6)
            (on-table b6)
            (on-table b7)
            (on b8 b9)
            (on-table b9)
        )
    )
)


(define (problem BW-9-7235-60)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init
        (on-table b1)
        (on-table b2)
        (on b3 b6)
        (on b4 b3)
        (on-table b5)
        (on-table b6)
        (on b7 b4)
        (on b8 b7)
        (on b9 b8)
        (clear b1)
        (clear b2)
        (clear b5)
        (clear b9)
    )
    (:goal
        (and
            (on b1 b4)
            (on b2 b9)
            (on-table b3)
            (on b4 b2)
            (on-table b5)
            (on b6 b8)
            (on-table b7)
            (on-table b8)
            (on b9 b5)
        )
    )
)