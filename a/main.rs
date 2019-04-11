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

struct Mountain {
    x: usize,
    y: usize,
    h: u32
}

impl Display for Mountain {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.h)
    }
}

fn gen_random(rng: &mut Xorshift) -> Mountain {
    Mountain {
        x: rng.rand() as usize % 100,
        y: rng.rand() as usize % 100,
        h: rng.rand() as u32 % 100 + 1
    }
}

fn main() {
    println!("1000");
    let mut rng = Xorshift::new();
    for _ in 0..1000 {
        println!("{}", gen_random(&mut rng));
    }
}
