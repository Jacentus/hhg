package io.fullstackdeveloper.lambda

// Pure functions
// - no side effects
// - referential transparency e.g. abs(-3) = 3
// - defined and valid for all possible input values (total vs. partial function)
fun abs(value: Int) = if (value < 0) -value else value

// Currying - transformation of a function with multiple arguments into a sequence of single-argument functions
fun add(value: Int, otherValue: Int) = value + otherValue
fun add(value: Int): (Int) -> Int = { otherValue -> value + otherValue }

/*
    val add3 = add(3)
    println(add3(2))
    println(add3(3))
 */

// Recursion
fun factorial(number: Int): Int {
    tailrec fun loop(currentNumber: Int, accumulator: Int): Int =
        if (currentNumber <= 0) accumulator
        else loop(currentNumber - 1, currentNumber * accumulator)
    return loop(number, 1)
}

fun fibonacci(elementIndex: Int): Int {
    tailrec fun loop(elementsLeft: Int, current: Int, next: Int): Int =
        if (elementsLeft == 0) current
        else loop(elementsLeft - 1, next, current + next)
    return loop(elementIndex, 0, 1)
}

// Higher-order functions
fun formatResult(n: Int, f: (Int) -> Int) = "Result: ${f(n)}"

/*
    println(formatResult(3, ::factorial))
    println(formatResult(3, ::fibonacci))
*/

// Polymorphic functions
typealias Predicate<T> = (T) -> Boolean

fun <E> findFirst(xs: Array<E>, predicate: Predicate<E>): Int {
    tailrec fun loop(index: Int): Int = when {
        index == xs.size -> -1
        predicate(xs[index]) -> index
        else -> loop(index + 1)
    }
    return loop(0)
}

fun isEven(value: Int) = value % 2 == 0

/*
    val numbers = arrayOf(1, 2, 3, 4, 5, 6)
    println(findFirst(numbers) { it > 2 })
    println(findFirst(numbers, ::isEven))
 */

fun <A, B, C> partial(a: A, fn: (A, B) -> C): (B) -> C = { b -> fn(a, b) }
fun <A, B, C> curry(fn: (A, B) -> C): (A) -> (B) -> C = { a: A -> { b: B -> fn(a, b) } }
fun <A, B, C> uncurry(fn: (A) -> (B) -> C): (A, B) -> C = { a: A, b: B -> fn(a)(b) }
fun <A, B, C> compose(f: (B) -> C, g: (A) -> B): (A) -> C = { a: A -> f(g(a)) }

/*
    val findInNumbers = partial(arrayOf(1, 2, 3, 4), ::findFirst)
    println(findInNumbers { it == 2 })
    println(findInNumbers { it == 3 })

    val curriedAdd = curry(::add)
    val add3 = curriedAdd(3)
    println(add3(3))

    val uncurriedAdd = uncurry(curriedAdd)
    println(uncurriedAdd(3, 4))

    val absAdd3 = compose(add3, ::abs) //add3(abs(-3))
    println(formatResult(3, absAdd3))
 */

fun main() {



}
