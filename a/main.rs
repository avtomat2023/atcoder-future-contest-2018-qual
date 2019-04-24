pub trait Readable {
    type Output;
    fn words_count() -> usize;
    fn read_words(words: &[&str]) -> Result<Self::Output, String>;
}
#[macro_export]
macro_rules! readable {
    ( $ t : ty , $ words_count : expr , |$ words : ident | $ read_words : expr ) => {
        impl Readable for $t {
            type Output = $t;
            fn words_count() -> usize {
                $words_count
            }
            fn read_words($words: &[&str]) -> Result<$t, String> {
                Ok($read_words)
            }
        }
    };
}
readable!((), 1, |_ss| ());
readable!(String, 1, |ss| ss[0].to_string());
impl Readable for char {
    type Output = char;
    fn words_count() -> usize {
        1
    }
    fn read_words(words: &[&str]) -> Result<char, String> {
        let chars: Vec<char> = words[0].chars().collect();
        if chars.len() == 1 {
            Ok(chars[0])
        } else {
            Err(format!("cannot parse \"{}\" as a char", words[0]))
        }
    }
}
pub struct Chars();
impl Readable for Chars {
    type Output = Vec<char>;
    fn words_count() -> usize {
        1
    }
    fn read_words(words: &[&str]) -> Result<Vec<char>, String> {
        Ok(words[0].chars().collect())
    }
}
macro_rules ! impl_readable_for_ints { ( $ ( $ t : ty ) * ) => { $ ( impl Readable for $ t { type Output = Self ; fn words_count ( ) -> usize { 1 } fn read_words ( words : & [ & str ] ) -> Result <$ t , String > { use std :: str :: FromStr ; <$ t >:: from_str ( words [ 0 ] ) . map_err ( | _ | { format ! ( "cannot parse \"{}\" as {}" , words [ 0 ] , stringify ! ( $ t ) ) } ) } } ) * } ; }
impl_readable_for_ints ! ( i8 u8 i16 u16 i32 u32 i64 u64 isize usize f32 f64 ) ;
macro_rules ! define_one_origin_int_types { ( $ new_t : ident $ int_t : ty ) => { # [ doc = " Converts 1-origin integer into 0-origin when read from stdin." ] # [ doc = "" ] # [ doc = " # Example" ] # [ doc = "" ] # [ doc = " ```no_run" ] # [ doc = " # #[macro_use] extern crate atcoder_snippets;" ] # [ doc = " # use atcoder_snippets::read::*;" ] # [ doc = " // Stdin: \"1\"" ] # [ doc = " read!(a = usize_);" ] # [ doc = " assert_eq!(a, 0);" ] # [ doc = " ```" ] # [ allow ( non_camel_case_types ) ] pub struct $ new_t ( ) ; impl Readable for $ new_t { type Output = $ int_t ; fn words_count ( ) -> usize { 1 } fn read_words ( words : & [ & str ] ) -> Result < Self :: Output , String > { <$ int_t >:: read_words ( words ) . map ( | n | n - 1 ) } } } ; ( $ new_t : ident $ int_t : ty ; $ ( $ inner_new_t : ident $ inner_int_t : ty ) ;* ) => { define_one_origin_int_types ! ( $ new_t $ int_t ) ; define_one_origin_int_types ! ( $ ( $ inner_new_t $ inner_int_t ) ;* ) ; } ; }
define_one_origin_int_types ! ( u8_ u8 ; u16_ u16 ; u32_ u32 ; u64_ u64 ; usize_ usize ) ;
macro_rules ! impl_readable_for_tuples { ( $ t : ident $ var : ident ) => ( ) ; ( $ t : ident $ var : ident ; $ ( $ inner_t : ident $ inner_var : ident ) ;* ) => { impl_readable_for_tuples ! ( $ ( $ inner_t $ inner_var ) ;* ) ; impl <$ t : Readable , $ ( $ inner_t : Readable ) ,*> Readable for ( $ t , $ ( $ inner_t ) ,* ) { type Output = ( <$ t >:: Output , $ ( <$ inner_t >:: Output ) ,* ) ; fn words_count ( ) -> usize { let mut n = <$ t >:: words_count ( ) ; $ ( n += <$ inner_t >:: words_count ( ) ; ) * n } # [ allow ( unused_assignments ) ] fn read_words ( words : & [ & str ] ) -> Result < Self :: Output , String > { let mut start = 0 ; let $ var = <$ t >:: read_words ( & words [ start .. start +<$ t >:: words_count ( ) ] ) ?; start += <$ t >:: words_count ( ) ; $ ( let $ inner_var = <$ inner_t >:: read_words ( & words [ start .. start +<$ inner_t >:: words_count ( ) ] ) ?; start += <$ inner_t >:: words_count ( ) ; ) * Ok ( ( $ var , $ ( $ inner_var ) ,* ) ) } } } ; }
impl_readable_for_tuples ! ( T4 x4 ; T3 x3 ; T2 x2 ; T1 x1 ) ;
pub trait ReadableFromLine {
    type Output;
    fn read_line(line: &str) -> Result<Self::Output, String>;
}
fn split_into_words(line: &str) -> Vec<&str> {
    #[allow(deprecated)]
    line.trim_right_matches('\n').split_whitespace().collect()
}
impl<T: Readable> ReadableFromLine for T {
    type Output = T::Output;
    fn read_line(line: &str) -> Result<T::Output, String> {
        let words = split_into_words(line);
        if words.len() != T::words_count() {
            return Err(format!(
                "line \"{}\" has {} words, expected {}",
                line,
                words.len(),
                T::words_count()
            ));
        }
        T::read_words(&words)
    }
}
impl<T: Readable> ReadableFromLine for Vec<T> {
    type Output = Vec<T::Output>;
    fn read_line(line: &str) -> Result<Self::Output, String> {
        let n = T::words_count();
        let words = split_into_words(line);
        if words.len() % n != 0 {
            return Err(format!(
                "line \"{}\" has {} words, expected multiple of {}",
                line,
                words.len(),
                n
            ));
        }
        let mut result = Vec::new();
        for chunk in words.chunks(n) {
            match T::read_words(chunk) {
                Ok(v) => result.push(v),
                Err(msg) => {
                    let flagment_msg = if n == 1 {
                        format!("word {}", result.len())
                    } else {
                        let l = result.len();
                        format!("words {}-{}", n * l + 1, (n + 1) * l)
                    };
                    return Err(format!("{} of line \"{}\": {}", flagment_msg, line, msg));
                }
            }
        }
        Ok(result)
    }
}
impl<T: Readable, U: Readable> ReadableFromLine for (T, Vec<U>) {
    type Output = (T::Output, <Vec<U> as ReadableFromLine>::Output);
    fn read_line(line: &str) -> Result<Self::Output, String> {
        let n = T::words_count();
        #[allow(deprecated)]
        let trimmed = line.trim_right_matches('\n');
        let words_and_rest: Vec<&str> = trimmed.splitn(n + 1, ' ').collect();
        if words_and_rest.len() < n {
            return Err(format!(
                "line \"{}\" has {} words, expected at least {}",
                line,
                words_and_rest.len(),
                n
            ));
        }
        let words = &words_and_rest[..n];
        let empty_str = "";
        let rest = words_and_rest.get(n).unwrap_or(&empty_str);
        Ok((T::read_words(words)?, Vec::<U>::read_line(rest)?))
    }
}
macro_rules ! impl_readable_from_line_for_tuples_with_vec { ( $ t : ident $ var : ident ) => ( ) ; ( $ t : ident $ var : ident ; $ ( $ inner_t : ident $ inner_var : ident ) ;+ ) => { impl_readable_from_line_for_tuples_with_vec ! ( $ ( $ inner_t $ inner_var ) ;+ ) ; impl <$ t : Readable , $ ( $ inner_t : Readable ) ,+ , U : Readable > ReadableFromLine for ( $ t , $ ( $ inner_t ) ,+ , Vec < U > ) { type Output = ( $ t :: Output , $ ( $ inner_t :: Output ) ,+ , Vec < U :: Output > ) ; fn read_line ( line : & str ) -> Result < Self :: Output , String > { let mut n = $ t :: words_count ( ) ; $ ( n += $ inner_t :: words_count ( ) ; ) + # [ allow ( deprecated ) ] let trimmed = line . trim_right_matches ( '\n' ) ; let words_and_rest : Vec <& str > = trimmed . splitn ( n + 1 , ' ' ) . collect ( ) ; if words_and_rest . len ( ) < n { return Err ( format ! ( "line \"{}\" has {} words, expected at least {}" , line , words_and_rest . len ( ) , n ) ) ; } let words = & words_and_rest [ .. n ] ; let empty_str = "" ; let rest = words_and_rest . get ( n ) . unwrap_or ( & empty_str ) ; let ( $ var , $ ( $ inner_var ) ,* ) = < ( $ t , $ ( $ inner_t ) ,+ ) >:: read_words ( words ) ?; Ok ( ( $ var , $ ( $ inner_var ) ,* , Vec ::< U >:: read_line ( rest ) ? ) ) } } } ; }
impl_readable_from_line_for_tuples_with_vec ! ( T4 t4 ; T3 t3 ; T2 t2 ; T1 t1 ) ;
pub fn read<T: ReadableFromLine>() -> T::Output {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    T::read_line(&line).unwrap()
}
#[macro_export]
macro_rules ! read { ( ) => { let mut line = String :: new ( ) ; std :: io :: stdin ( ) . read_line ( & mut line ) . unwrap ( ) ; } ; ( $ pat : pat = $ t : ty ) => { let $ pat = read ::<$ t > ( ) ; } ; ( $ ( $ pat : pat = $ t : ty ) ,+ ) => { read ! ( ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) ) ; } ; }
#[macro_export]
macro_rules ! readls { ( $ ( $ pat : pat = $ t : ty ) ,+ ) => { $ ( read ! ( $ pat = $ t ) ; ) * } ; }
pub fn readx<T: ReadableFromLine>() -> Vec<T::Output> {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let result = stdin
        .lock()
        .lines()
        .map(|line_result| {
            let line = line_result.expect("read from stdin failed");
            T::read_line(&line).unwrap()
        })
        .collect();
    result
}
#[macro_export]
macro_rules ! readx_loop { ( |$ pat : pat = $ t : ty | $ body : expr ) => { use std :: io :: BufRead ; let stdin = std :: io :: stdin ( ) ; for line in stdin . lock ( ) . lines ( ) { let line = line . expect ( "read from stdin failed" ) ; let $ pat = <$ t >:: read_line ( & line ) . unwrap ( ) ; $ body } } ; ( |$ ( $ pat : pat = $ t : ty ) ,*| $ body : expr ) => { readx_loop ! ( | ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) | $ body ) ; } ; }
pub fn readn<T: ReadableFromLine>(n: usize) -> Vec<T::Output> {
    use std::io::{self, BufRead};
    let stdin = io::stdin();
    let result: Vec<T::Output> = stdin
        .lock()
        .lines()
        .take(n)
        .map(|line_result| {
            let line = line_result.expect("read from stdin failed");
            T::read_line(&line).unwrap()
        })
        .collect();
    if result.len() < n {
        panic!(
            "expected reading {} lines, but only {} lines are read",
            n,
            result.len()
        );
    }
    result
}
#[macro_export]
macro_rules ! readn_loop { ( $ n : expr , |$ pat : pat = $ t : ty | $ body : expr ) => { use std :: io :: BufRead ; let stdin = std :: io :: stdin ( ) ; { let mut lock = stdin . lock ( ) ; for _ in 0 ..$ n { let mut line = String :: new ( ) ; lock . read_line ( & mut line ) . expect ( "read from stdin failed" ) ; let $ pat = <$ t >:: read_line ( & line ) . unwrap ( ) ; $ body } } } ; ( $ n : expr , |$ ( $ pat : pat = $ t : ty ) ,*| $ body : expr ) => { readn_loop ! ( $ n , | ( $ ( $ pat ) ,* ) = ( $ ( $ t ) ,* ) | $ body ) ; } ; }
pub trait Words {
    fn read<T: Readable>(&self) -> T::Output;
}
impl<'a> Words for [&'a str] {
    fn read<T: Readable>(&self) -> T::Output {
        T::read_words(self).unwrap()
    }
}
impl<'a> Words for &'a str {
    fn read<T: Readable>(&self) -> T::Output {
        T::read_words(&[self]).unwrap()
    }
}

#[allow(unused_imports)]
use std::io::{self, BufWriter, StdoutLock, Write};
pub fn with_stdout<F: FnOnce(BufWriter<StdoutLock>)>(f: F) {
    let stdout = io::stdout();
    let writer = BufWriter::new(stdout.lock());
    f(writer);
}

#[derive(Clone)]
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first_take: bool,
}
impl<I: Iterator> Iterator for StepBy<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            self.iter.next()
        } else {
            self.iter.nth(self.step)
        }
    }
}
#[derive(Clone)]
pub struct LScan<I: Iterator, S: Clone, F: FnMut(&S, I::Item) -> S> {
    iter: I,
    state: Option<S>,
    f: F,
}
impl<I: Iterator, S: Clone, F> Iterator for LScan<I, S, F>
where
    F: FnMut(&S, I::Item) -> S,
{
    type Item = S;
    fn next(&mut self) -> Option<S> {
        if self.state.is_none() {
            return None;
        }
        let state_inner = self.state.take().unwrap();
        if let Some(item) = self.iter.next() {
            self.state = Some((self.f)(&state_inner, item));
        }
        Some(state_inner)
    }
}
pub struct Flatten<I: Iterator>
where
    I::Item: IntoIterator,
{
    outer_iter: I,
    inner_iter: Option<<<I as Iterator>::Item as IntoIterator>::IntoIter>,
}
impl<I, J> Iterator for Flatten<I>
where
    I: Iterator<Item = J>,
    J: IntoIterator,
{
    type Item = <<J as IntoIterator>::IntoIter as Iterator>::Item;
    fn next(&mut self) -> Option<J::Item> {
        loop {
            if let Some(inner_iter) = self.inner_iter.as_mut() {
                if let item @ Some(_) = inner_iter.next() {
                    return item;
                }
            }
            match self.outer_iter.next() {
                None => return None,
                Some(inner) => self.inner_iter = Some(inner.into_iter()),
            }
        }
    }
}
pub struct GroupBy<K: Eq, I: Iterator, F: FnMut(&I::Item) -> K> {
    cur: Option<(I::Item, K)>,
    iter: I,
    key_fn: F,
}
impl<K: Eq, I: Iterator, F: FnMut(&I::Item) -> K> Iterator for GroupBy<K, I, F> {
    type Item = (K, Vec<I::Item>);
    fn next(&mut self) -> Option<(K, Vec<I::Item>)> {
        let cur = self.cur.take();
        cur.map(|(item, key)| {
            let mut group = vec![item];
            loop {
                let next = self.iter.next();
                match next {
                    Some(next_item) => {
                        let next_key = (self.key_fn)(&next_item);
                        if key == next_key {
                            group.push(next_item);
                        } else {
                            self.cur = Some((next_item, next_key));
                            break;
                        }
                    }
                    None => {
                        self.cur = None;
                        break;
                    }
                }
            }
            (key, group)
        })
    }
}
pub trait IteratorExt: Iterator {
    fn step_by_(self, step: usize) -> StepBy<Self>
    where
        Self: Sized,
    {
        assert_ne!(step, 0);
        StepBy {
            iter: self,
            step: step - 1,
            first_take: true,
        }
    }
    fn for_each<F: FnMut(Self::Item)>(self, mut f: F)
    where
        Self: Sized,
    {
        for item in self {
            f(item);
        }
    }
    fn lscan<S: Clone, F>(self, state: S, f: F) -> LScan<Self, S, F>
    where
        Self: Sized,
        F: FnMut(&S, Self::Item) -> S,
    {
        LScan {
            iter: self,
            state: Some(state),
            f: f,
        }
    }
    fn get_unique(mut self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Eq,
    {
        let first_opt = self.next();
        first_opt.and_then(|first| {
            if self.all(|item| item == first) {
                Some(first)
            } else {
                None
            }
        })
    }
    fn flatten(mut self) -> Flatten<Self>
    where
        Self: Sized,
        Self::Item: IntoIterator,
    {
        let inner_opt = self.next();
        Flatten {
            outer_iter: self,
            inner_iter: inner_opt.map(|inner| inner.into_iter()),
        }
    }
    fn group_by<K: Eq, F: FnMut(&Self::Item) -> K>(mut self, mut f: F) -> GroupBy<K, Self, F>
    where
        Self: Sized,
    {
        let next = self.next();
        GroupBy {
            cur: next.map(|item| {
                let key = f(&item);
                (item, key)
            }),
            iter: self,
            key_fn: f,
        }
    }
    fn join(mut self, sep: &str) -> String
    where
        Self: Sized,
        Self::Item: std::fmt::Display,
    {
        let mut result = String::new();
        if let Some(first) = self.next() {
            result.push_str(&format!("{}", first));
        }
        for s in self {
            result.push_str(&format!("{}{}", sep, s));
        }
        result
    }
    fn cat(self) -> String
    where
        Self: Sized,
        Self::Item: std::fmt::Display,
    {
        self.join("")
    }
}
impl<I: Iterator> IteratorExt for I {}
pub struct Unfold<T, F>
where
    F: FnMut(&T) -> Option<T>,
{
    state: Option<T>,
    f: F,
}
impl<T, F> Iterator for Unfold<T, F>
where
    F: FnMut(&T) -> Option<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.state.is_none() {
            return None;
        }
        let state_inner = self.state.take().unwrap();
        self.state = (self.f)(&state_inner);
        Some(state_inner)
    }
}
pub fn unfold<T, F>(init: T, f: F) -> Unfold<T, F>
where
    F: FnMut(&T) -> Option<T>,
{
    Unfold {
        state: Some(init),
        f: f,
    }
}
pub struct Iterate<T, F>
where
    F: FnMut(&T) -> T,
{
    state: T,
    f: F,
}
impl<T, F> Iterator for Iterate<T, F>
where
    F: FnMut(&T) -> T,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        use std::mem::swap;
        let mut state = (self.f)(&self.state);
        swap(&mut state, &mut self.state);
        Some(state)
    }
}
pub fn iterate<T, F>(init: T, f: F) -> Iterate<T, F>
where
    F: FnMut(&T) -> T,
{
    Iterate { state: init, f: f }
}

pub trait PrimitiveInteger {
    fn abs_diff(self, rhs: Self) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
}
macro_rules ! impl_primitive_integer { ( $ ( $ t : ty ) * ) => { $ ( impl PrimitiveInteger for $ t { fn abs_diff ( self , rhs : $ t ) -> $ t { if self < rhs { rhs - self } else { self - rhs } } # [ allow ( unused_comparisons ) ] fn rem_euclid ( self , rhs : $ t ) -> $ t { let r = self % rhs ; if r < 0 { if rhs < 0 { r - rhs } else { r + rhs } } else { r } } } ) * } }
impl_primitive_integer ! ( u8 u16 u32 u64 usize i8 i16 i32 i64 isize ) ;
pub trait PrimitiveUnsigned {
    fn ceil_div(self, rhs: Self) -> Self;
    fn round_div(self, rhs: Self) -> Self;
}
macro_rules ! impl_primitive_unsigned { ( $ ( $ t : ty ) * ) => { $ ( impl PrimitiveUnsigned for $ t { fn ceil_div ( self , rhs : $ t ) -> $ t { ( self + rhs - 1 ) / rhs } fn round_div ( self , rhs : $ t ) -> $ t { ( self + rhs / 2 ) / rhs } } ) * } }
impl_primitive_unsigned ! ( u8 u16 u32 u64 usize ) ;

use std::cmp::{Ord, Ordering};
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Reverse<T>(pub T);
impl<T: PartialOrd> PartialOrd for Reverse<T> {
    fn partial_cmp(&self, other: &Reverse<T>) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}
impl<T: Ord> Ord for Reverse<T> {
    fn cmp(&self, other: &Reverse<T>) -> Ordering {
        other.0.cmp(&self.0)
    }
}
pub trait SortDesc<T> {
    fn sort_desc(&mut self)
    where
        T: Ord;
    fn sort_desc_by_key<K: Ord, F: FnMut(&T) -> K>(&mut self, f: F);
}
impl<T> SortDesc<T> for [T] {
    fn sort_desc(&mut self)
    where
        T: Ord,
    {
        self.sort();
        self.reverse();
    }
    fn sort_desc_by_key<K: Ord, F: FnMut(&T) -> K>(&mut self, mut f: F) {
        self.sort_by_key(|x| Reverse(f(x)));
    }
}
#[derive(PartialEq, PartialOrd)]
pub struct Total<T: PartialOrd + PartialEq>(pub T);
impl<T: PartialOrd + PartialEq> Eq for Total<T> {}
impl<T: PartialOrd + PartialEq> Ord for Total<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[macro_export]
macro_rules! dbg {
    ( $ e : expr ) => {{
        use std::io::{self, Write};
        let result = $e;
        writeln!(
            io::stderr(),
            "{}: {} = {:?}",
            line!(),
            stringify!($e),
            result
        )
            .unwrap();
        result
    }};
}

#[derive(Debug)]
pub struct Xorshift {
    seed: u64,
}

impl Xorshift {
    pub fn new() -> Xorshift {
        Xorshift {
            seed: 0xf0fb588ca2196dac,
        }
    }

    pub fn with_seed(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }

    pub fn next(&mut self) -> u64 {
        self.seed = self.seed ^ (self.seed << 13);
        self.seed = self.seed ^ (self.seed >> 7);
        self.seed = self.seed ^ (self.seed << 17);
        self.seed
    }

    pub fn rand(&mut self) -> u64 {
        self.next()
    }

    // 0.0 ~ 1.0
    pub fn randf(&mut self) -> f64 {
        use std::mem;
        const UPPER_MASK: u64 = 0x3FF0000000000000;
        const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
        let tmp = UPPER_MASK | (self.next() & LOWER_MASK);
        let result: f64 = unsafe { mem::transmute(tmp) };
        result - 1.0
    }
}

use std::fmt::{self, Display, Formatter};
use std::time::{Instant, Duration};
use std::cmp::min;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};
use std::vec;
use std::slice;

const BOARD_WIDTH: usize = 100;
const MAX_HEIGHT: usize = 100;
const MOUNTAIN_MAX_COUNT: usize = 1000;

// Parameters
const POPULATION_SIZE: usize = 20;
const ELITE_SIZE: usize = 5;
const CROSSOVER_SIZE: usize = 15;
const MUTATION_SIZE: usize = 0;

#[derive(Clone, PartialEq, Eq, Debug)]
struct Board<T> {
    cells: Vec<T>
}

impl<T> Index<usize> for Board<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &[T] {
        debug_assert!(index < BOARD_WIDTH);
        &self.cells[BOARD_WIDTH * index .. BOARD_WIDTH * (index+1)]
    }
}

impl<T> IndexMut<usize> for Board<T> {
    fn index_mut(&mut self, index: usize) -> &mut [T] {
        &mut self.cells[BOARD_WIDTH * index .. BOARD_WIDTH * (index+1)]
    }
}

impl<T> IntoIterator for Board<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> vec::IntoIter<T> {
        self.cells.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Board<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> slice::Iter<'a, T> {
        self.iter()
    }
}

impl<T> FromIterator<T> for Board<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Board<T> {
        Board::from_vec(iter.into_iter().collect())
    }
}

impl<T> Board<T> {
    fn len() -> usize {
        BOARD_WIDTH * BOARD_WIDTH
    }

    fn new() -> Board<T> where T: Default + Clone {
        Board { cells: vec![T::default(); Board::<T>::len()] }
    }

    fn from_vec(v: Vec<T>) -> Board<T> {
        debug_assert!(v.len() == Board::<T>::len());
        Board { cells: v }
    }

    fn iter(&self) -> slice::Iter<T> {
        self.cells.iter()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Mountain {
    x: usize,
    y: usize,
    h: usize,
}

impl Display for Mountain {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.h)
    }
}

impl Mountain {
    fn random(rng: &mut Xorshift) -> Mountain {
        Mountain {
            x: rng.rand() as usize % BOARD_WIDTH,
            y: rng.rand() as usize % BOARD_WIDTH,
            h: rng.rand() as usize % MAX_HEIGHT,
        }
    }

    fn for_each_cell<F>(&self, mut f: F)
    where
        F: FnMut((usize, usize), usize)
    {
        if self.h == 0 {
            return;
        }

        let ty = triangle(self.h, self.y, 0, BOARD_WIDTH-1);
        let iter_y = (self.y.saturating_sub(self.h-1) ..)
            .zip(ty.into_iter());
        for (y, hy) in iter_y {
            let tx = triangle(hy, self.x, 0, BOARD_WIDTH-1);
            let iter_x = (self.x.saturating_sub(hy-1)..)
                .zip(tx.into_iter());
            for (x, h) in iter_x {
                f((x, y), h);
            }
        }
    }
}

fn triangle(h: usize, center: usize, left: usize, right: usize) -> Vec<usize> {
    if h == 0 {
        return Vec::new();
    }

    let mut res = Vec::with_capacity(2 * h as usize - 1);

    let l_width = min(h - 1, center - left);
    for i in (1..h).skip((h-1) - l_width) {
        res.push(i);
    }

    res.push(h);

    let r_width = min(h - 1, right - center);
    for i in (1..h).skip((h-1) - r_width).rev() {
        res.push(i);
    }

    res
}

#[cfg(test)]
#[test]
fn test_triangle() {
    assert_eq!(triangle(2, 1, 0, 10), vec![1, 2, 1]);
    assert_eq!(triangle(3, 1, 0, 10), vec![2, 3, 2, 1]);

    assert_eq!(triangle(2, 9, 0, 10), vec![1, 2, 1]);
    assert_eq!(triangle(3, 9, 0, 10), vec![1, 2, 3, 2]);
}

fn calc_penalty(input: &Board<usize>, ms: &[Mountain]) -> usize {
    let mut diff: Board<isize> = input.iter().map(|&h| h as isize).collect();

    for m in ms {
        m.for_each_cell(|(x,y), h| {
            diff[y][x] -= h as isize;
        });
    }
    diff.into_iter().map(|d| d.abs() as usize).sum()
}

fn select_elites(mut population: Vec<Vec<Mountain>>, input: &Board<usize>) -> Vec<Vec<Mountain>> {
    population.sort_by_key(|ms| calc_penalty(input, ms));
    population[..ELITE_SIZE].to_vec()
}

fn make_crossover(elites: &[Vec<Mountain>], rng: &mut Xorshift) -> Vec<Vec<Mountain>> {
    (0..CROSSOVER_SIZE).map(|_| {
        let e1 = &elites[rng.rand() as usize % elites.len()];
        let e2 = &elites[rng.rand() as usize % elites.len()];
        let mut mountains = Vec::new();
        for _ in 0..MOUNTAIN_MAX_COUNT/2 {
            mountains.push(e1[rng.rand() as usize % e1.len()].clone());
        }
        for _ in MOUNTAIN_MAX_COUNT/2..MOUNTAIN_MAX_COUNT {
            mountains.push(e2[rng.rand() as usize % e2.len()].clone());
        }
        mountains
    }).collect()
}

fn solve(input: &Board<usize>, time_limit: Instant) -> Vec<Mountain> {
    let time_limit_tightend: Instant = time_limit - Duration::from_millis(100);
    let mut rng = Xorshift::new();
    let mut loop_count = 0;
    let mut population = Vec::new();
    for _ in 0..POPULATION_SIZE {
        let mut ms = Vec::new();
        for _ in 0..MOUNTAIN_MAX_COUNT {
            ms.push(Mountain::random(&mut rng));
        }
        population.push(ms);
    }

    loop {
        if Instant::now() > time_limit_tightend {
            break;
        }

        let elites = select_elites(population, input);
        population = elites.clone();
        population.extend(make_crossover(&elites, &mut rng));
        // TODO: make mutation

        loop_count += 1;
    }

    dbg!(loop_count);
    let ms = population.into_iter().min_by_key(|ms| calc_penalty(input, ms)).unwrap();
    ms.into_iter().filter(|m| m.h > 0).collect()
}

fn main() {
    let start = Instant::now();

    let input: Board<usize> = readx::<Vec<usize>>().into_iter().flatten().collect();
    let ans = solve(&input, start + Duration::from_millis(20_000));
    with_stdout(|mut f| {
        writeln!(f, "{}", ans.len()).unwrap();
        for m in ans {
            writeln!(f, "{}", m).unwrap();
        }
    });
}
