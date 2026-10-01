//! Exact unsigned layout expressions, bounded over the complete launch domain.
use super::{Result, unsupported};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Index {
    Lane,
    Constant(u64),
    Add(Rc<Index>, Rc<Index>),
    Sub(Rc<Index>, Rc<Index>),
    Mul(Rc<Index>, Rc<Index>),
    Div(Rc<Index>, u64),
    Mod(Rc<Index>, u64),
}

#[cfg(test)]
mod tests {
    use super::*;
    fn op(kind:char,a:Rc<Index>,b:Rc<Index>,n:u64)->Rc<Index>{Index::binary(kind,a,b,n,u32::MAX as u64).unwrap()}
    fn c(n:u64)->Rc<Index>{Rc::new(Index::Constant(n))}
    #[test]
    fn layout_offsets_match_independent_coordinate_reference() {
        for rows in [1,2,5,17] {for cols in [1,3,13,65] {let n=rows*cols;
            for (sy,sx) in [(0,1),(1,rows),(cols+3,1)] {
                let lane=Rc::new(Index::Lane);
                let y=op('/',lane.clone(),c(cols),n);
                let x=op('-',lane.clone(),op('*',y.clone(),c(cols),n),n);
                let index=op('+',op('*',y,c(sy),n),op('*',x,c(sx),n),n);
                for i in 0..n {assert_eq!(index.eval(i),i/cols*sy+i%cols*sx);}
                let (_,hi)=index.bounds(n).unwrap();
                assert!((0..n).all(|i|index.eval(i)<=hi));
            }
        }}
    }
    #[test]
    fn integer_faults_are_not_silently_wrapped() {
        assert!(Index::binary('/',Rc::new(Index::Lane),c(0),65,u32::MAX as u64).is_err());
        assert!(Index::binary('+',Rc::new(Index::Lane),c(u32::MAX as u64),65,u32::MAX as u64).is_err());
        assert!(Index::binary('-',Rc::new(Index::Lane),c(1),65,u32::MAX as u64).is_err());
    }
}
impl Index {
    pub fn binary(op: char, a: Rc<Self>, b: Rc<Self>, elements: u64, max: u64) -> Result<Rc<Self>> {
        let result = match (op, a.as_ref(), b.as_ref()) {
            ('+' | '-', _, Self::Constant(0)) | ('*' | '/', _, Self::Constant(1)) => a,
            ('+', Self::Constant(0), _) | ('*', Self::Constant(1), _) => b,
            ('*', _, Self::Constant(0)) | ('*', Self::Constant(0), _) => Rc::new(Self::Constant(0)),
            ('+', _, _) => Rc::new(Self::Add(a, b)),
            ('*', _, _) => Rc::new(Self::Mul(a, b)),
            ('/' | '%', _, Self::Constant(d)) if *d != 0 => {
                let d = *d;
                Rc::new(if op == '/' { Self::Div(a, d) } else { Self::Mod(a, d) })
            }
            ('-', _, Self::Mul(q, d)) => {
                if let (Self::Div(x, divisor), Self::Constant(factor)) = (q.as_ref(), d.as_ref()) {
                    if x == &a && divisor == factor {
                        Rc::new(Self::Mod(a, *divisor))
                    } else { Rc::new(Self::Sub(a, b)) }
                } else { Rc::new(Self::Sub(a, b)) }
            }
            ('-', _, _) => Rc::new(Self::Sub(a, b)),
            _ => return Err(unsupported("layout division requires a nonzero constant divisor")),
        };
        let (lo, hi) = result.bounds(elements)?;
        if hi > max { return Err(unsupported("layout index arithmetic may overflow its unsigned type")); }
        Ok(if lo == hi { Rc::new(Self::Constant(lo)) } else { result })
    }
    pub fn bounds(&self, elements: u64) -> Result<(u64, u64)> {
        let overflow = || unsupported("layout index arithmetic is not provably non-wrapping");
        Ok(match self {
            Self::Lane => (0, elements.saturating_sub(1)),
            Self::Constant(n) => (*n, *n),
            Self::Div(a, d) => { let (lo, hi) = a.bounds(elements)?; (lo / d, hi / d) }
            Self::Mod(a, d) => { let (lo, hi) = a.bounds(elements)?; if lo / d == hi / d { (lo % d, hi % d) } else { (0, d - 1) } }
            Self::Add(a, b) => { let (al, ah) = a.bounds(elements)?; let (bl, bh) = b.bounds(elements)?; (al.checked_add(bl).ok_or_else(overflow)?, ah.checked_add(bh).ok_or_else(overflow)?) }
            Self::Mul(a, b) => { let (al, ah) = a.bounds(elements)?; let (bl, bh) = b.bounds(elements)?; (al.checked_mul(bl).ok_or_else(overflow)?, ah.checked_mul(bh).ok_or_else(overflow)?) }
            Self::Sub(a, b) => { let (al, ah) = a.bounds(elements)?; let (bl, bh) = b.bounds(elements)?; (al.checked_sub(bh).ok_or_else(overflow)?, ah.checked_sub(bl).ok_or_else(overflow)?) }
        })
    }
    pub fn cce(&self) -> String {
        match self {
            Self::Lane => "(offset + lane)".into(),
            Self::Constant(n) => format!("{n}ULL"),
            Self::Add(a,b) => format!("({} + {})", a.cce(), b.cce()),
            Self::Sub(a,b) => format!("({} - {})", a.cce(), b.cce()),
            Self::Mul(a,b) => format!("({} * {})", a.cce(), b.cce()),
            Self::Div(a,d) => format!("({} / {d}ULL)", a.cce()),
            Self::Mod(a,d) => format!("({} % {d}ULL)", a.cce()),
        }
    }
    #[cfg(test)]
    pub fn eval(&self, lane: u64) -> u64 {
        match self { Self::Lane => lane, Self::Constant(n) => *n, Self::Add(a,b) => a.eval(lane)+b.eval(lane), Self::Sub(a,b) => a.eval(lane)-b.eval(lane), Self::Mul(a,b) => a.eval(lane)*b.eval(lane), Self::Div(a,d) => a.eval(lane)/d, Self::Mod(a,d) => a.eval(lane)%d }
    }
}
