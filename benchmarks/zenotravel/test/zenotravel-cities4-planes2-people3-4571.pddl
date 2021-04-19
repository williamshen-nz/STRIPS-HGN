(define (problem ZTRAVEL-cities4-planes2-people3)
(:domain zeno-travel)
(:objects
	plane1 - aircraft
	plane2 - aircraft
	person1 - person
	person2 - person
	person3 - person
	city0 - city
	city1 - city
	city2 - city
	city3 - city
	fl0 - flevel
	fl1 - flevel
	fl2 - flevel
	fl3 - flevel
	fl4 - flevel
	fl5 - flevel
	fl6 - flevel
	)
(:init
	(at plane1 city0)
	(fuel-level plane1 fl0)
	(at plane2 city2)
	(fuel-level plane2 fl0)
	(at person1 city3)
	(at person2 city0)
	(at person3 city0)
	(next fl0 fl1)
	(next fl1 fl2)
	(next fl2 fl3)
	(next fl3 fl4)
	(next fl4 fl5)
	(next fl5 fl6)
)
(:goal (and
	(at person1 city0)
	(at person2 city3)
	(at person3 city1)
	))

)
