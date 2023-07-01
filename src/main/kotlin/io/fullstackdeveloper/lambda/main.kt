package io.fullstackdeveloper.lambda

import java.time.LocalDateTime

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

// Functional data structures

sealed class List<out A> {

    companion object {

        fun <A> of(vararg xs: A): List<A> {
            val tail = xs.sliceArray(1 until xs.size)
            return if (xs.isEmpty()) Nil else Cons(xs[0], of(*tail))
        }

        fun <A> empty(): List<A> = Nil

    }

}

object Nil : List<Nothing>()
data class Cons<out A>(val head: A, val tail: List<A>) : List<A>()

/*
    val emptyList = Nil
    val messages = Cons("java", Cons("kotlin", Nil))
 */

fun sum(xs: List<Int>): Int = when (xs) {
    is Nil -> 0
    is Cons -> xs.head + sum(xs.tail)
}

fun product(xs: List<Double>): Double = when (xs) {
    is Nil -> 1.0
    is Cons -> xs.head * product(xs.tail)
}

fun <A> tail(xs: List<A>) = when (xs) {
    is Cons -> xs.tail
    else -> Nil
}

fun <A> setHead(xs: List<A>, x: A): List<A> = when (xs) {
    is Nil -> Nil
    is Cons -> Cons(x, xs.tail)
}

fun <A> prepend(xs: List<A>, x: A) = when (xs) {
    is Cons -> Cons(x, xs)
    else -> Nil
}

fun <A> append(xs1: List<A>, xs2: List<A>): List<A> = when (xs1) {
    is Nil -> xs2
    is Cons -> Cons(xs1.head, append(xs1.tail, xs2))
}

tailrec fun <A> drop(xs: List<A>, n: Int): List<A> =
    if (n <= 0) xs else when (xs) {
        is Cons -> drop(xs.tail, n - 1)
        else -> Nil
    }

tailrec fun <A> dropWhile(xs: List<A>, predicate: Predicate<A>): List<A> = when (xs) {
    is Cons -> if (predicate(xs.head)) dropWhile(xs.tail, predicate) else xs
    else -> xs
}

fun <A> init(l: List<A>): List<A> = when (l) {
    is Cons -> if (l.tail == Nil) Nil else Cons(l.head, init(l.tail))
    is Nil -> Nil
}

fun <A, B> foldRight(xs: List<A>, value: B, f: (A, B) -> B): B = when (xs) {
    is Nil -> value
    is Cons -> f(xs.head, foldRight(xs.tail, value, f))
}

fun sumFr(xs: List<Int>) = foldRight(xs, 0) { a, b -> a + b }
fun productFr(xs: List<Int>) = foldRight(xs, 1.0) { a, b -> a * b }
fun lengthFr(xs: List<Int>) = foldRight(xs, 0) { _, len -> 1 + len }
fun <A> appendFr(a1: List<A>, a2: List<A>): List<A> = foldRight(a1, a2) { x, y -> Cons(x, y) }
fun <A> concatFr(xxs: List<List<A>>): List<A> =
    foldRight(xxs, List.empty()) { xs1: List<A>, xs2: List<A> ->
        foldRight(xs1, xs2) { a, ls -> Cons(a, ls) }
    }


tailrec fun <A, B> foldLeft(xs: List<A>, value: B, f: (B, A) -> B): B = when (xs) {
    is Nil -> value
    is Cons -> foldLeft(xs.tail, f(value, xs.head), f)
}

fun sumFl(xs: List<Int>) = foldLeft(xs, 0) { a, b -> a + b }
fun productFl(xs: List<Int>) = foldLeft(xs, 1.0) { a, b -> a * b }
fun lengthFl(xs: List<Int>) = foldLeft(xs, 0) { len, _ -> 1 + len }

fun <A> reverseFl(xs: List<A>): List<A> = foldLeft(xs, List.empty()) { t: List<A>, h: A -> Cons(h, t) }

fun <A, B> foldLeftR(xs: List<A>, z: B, f: (B, A) -> B): B =
    foldRight(xs, { b: B -> b }, { a, g -> { b -> g(f(b, a)) } })(z)

fun <A, B> foldRightL(xs: List<A>, z: B, f: (A, B) -> B): B =
    foldLeft(xs, { b: B -> b }, { g, a -> { b -> g(f(a, b)) } })(z)

fun <A, B> map(xs: List<A>, f: (A) -> B): List<B> = foldRightL(xs, List.empty()) { a: A, xa: List<B> ->
    Cons(f(a), xa)
}

fun <A> filter(xs: List<A>, f: (A) -> Boolean): List<A> = foldRight(xs, List.empty()) { a, ls ->
    if (f(a)) Cons(a, ls)
    else ls
}

fun <A, B> flatMap(xa: List<A>, f: (A) -> List<B>): List<B> =
    foldRight(xa, List.empty()) { a, lb ->
        append(f(a), lb)
    }

fun <A> zipWith(xa: List<A>, xb: List<A>, f: (A, A) -> A): List<A> = when (xa) {
    is Nil -> Nil
    is Cons -> when (xb) {
        is Nil -> Nil
        is Cons -> Cons(f(xa.head, xb.head), zipWith(xa.tail, xb.tail, f))
    }
}

/*
    val values = List.of(1, 2, 3, 4, 5)
    println(sum(values))
    println(product(List.of(1.0, 2.0, 3.0)))
 */

sealed class Tree<out A>
data class Leaf<A>(val value: A) : Tree<A>()
data class Branch<A>(val left: Tree<A>, val right: Tree<A>) : Tree<A>()

fun <A> numberOfNodes(tree: Tree<A>): Int = when (tree) {
    is Leaf -> 1
    is Branch -> 1 + numberOfNodes(tree.left) + numberOfNodes(tree.right)
}

fun <A> maxDepth(tree: Tree<A>): Int = when (tree) {
    is Leaf -> 0
    is Branch -> 1 + maxOf(maxDepth(tree.left), maxDepth(tree.right))
}

fun <A, B> map(tree: Tree<A>, f: (A) -> B): Tree<B> = when (tree) {
    is Leaf -> Leaf(f(tree.value))
    is Branch -> Branch(map(tree.left, f), map(tree.right, f))
}

fun <A, B> fold(tree: Tree<A>, f: (A) -> B, b: (B, B) -> B): B = when (tree) {
    is Leaf -> f(tree.value)
    is Branch -> b(fold(tree.left, f, b), fold(tree.right, f, b))
}

fun <A> numberOfNodeF(tree: Tree<A>) = fold(tree, { 1 }, { b1, b2 -> 1 + b1 + b2 })

fun <A> maxDepthF(tree: Tree<A>) = fold(tree, { 0 }, { b1, b2 -> 1 + maxOf(b1, b2) })

fun <A, B> mapF(tree: Tree<A>, f: (A) -> B) =
    fold(tree, { a: A -> Leaf(f(a)) }, { b1: Tree<B>, b2: Tree<B> -> Branch(b1, b2) })

sealed class Option<out A>
class Some<A>(val value: A) : Option<A>()
object None : Option<Nothing>()

fun <A, B> Option<A>.map(f: (A) -> B) = when (this) {
    is None -> None
    is Some -> Some(f(value))
}

fun <A> Option<A>.getOrElse(default: () -> A): A = when (this) {
    is None -> default()
    is Some -> value
}

fun <A, B> Option<A>.flatMap(f: (A) -> Option<B>): Option<B> = map(f).getOrElse { None }

fun <A> Option<A>.orElse(ob: () -> Option<A>): Option<A> = map { Some(it) }.getOrElse { ob() }

fun <A> Option<A>.filter(f: (A) -> Boolean): Option<A> = flatMap { a -> if (f(a)) Some(a) else None }

fun getTimestamp(): Option<LocalDateTime> = None // Some(LocalDateTime.now())

sealed class Either<out E, out A>
data class Left<out E>(val value: E) : Either<E, Nothing>()
data class Right<out A>(val value: A) : Either<Nothing, A>()

fun <E, A, B> Either<E, A>.map(f: (A) -> B): Either<E, B> = when (this) {
    is Left -> this
    is Right -> Right(f(value))
}

fun <E, A> Either<E, A>.orElse(f: () -> Either<E, A>): Either<E, A> = when (this) {
    is Left -> f()
    is Right -> this
}

fun safeDiv(x: Int, y: Int): Either<String, Int> =
    try {
        Right(x / y)
    } catch (e: Exception) {
        Left("Division by zero")
    }

/*
 val datetime = getTimestamp()
        .map { it.year }
        .getOrElse { LocalDateTime.MIN }
    println(datetime)

    val division = safeDiv(2, 0)
        .map(add2)
        .map(::abs)
        .orElse { Right(0) }
 */


fun main() {

}
