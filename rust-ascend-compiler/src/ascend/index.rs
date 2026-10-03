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

/// On each span-aligned interval, index(base+lane)=index(base)+slope*lane.
/// Span zero denotes a globally affine expression, without interval boundaries.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) struct IndexRun {pub span:u64,pub slope:i128}
fn common_span(mut a:u64,mut b:u64)->u64 {
    while b!=0 {let remainder=a%b;a=b;b=remainder;}a
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
    #[test]
    fn proven_runs_match_exact_addresses_padding_and_special_value_bits() {
        for columns in [8u64,32,96,4096] {
            let n=7*columns-3;let width=columns+32;let lane=Rc::new(Index::Lane);
            let row=op('/',lane.clone(),c(columns),n);let column=op('%',lane.clone(),c(columns),n);
            let full=op('+',op('+',op('*',row.clone(),c(width),n),column.clone(),n),c(8),n);
            let shared=op('+',column,c(8),n);
            for (index,span,slope) in [(op('+',lane,c(3),n),0,1),(row,columns,0),(shared,columns,1),(full,columns,1)] {
                assert_eq!(index.aligned_run(),Some(IndexRun {span,slope}));
                let source:Vec<u32>=(0..=index.bounds(n).unwrap().1).map(|i|
                    [0u32,0x80000000,0x7f800000,0xff800000,0x7fc01234,0x3f800000][i as usize%6]).collect();
                for tile in [8u64,256,4096] {for offset in (0..n).step_by(tile as usize) {
                    let count=(n-offset).min(tile);let aligned=(count+7)/8*8;let mut local=vec![0u32;aligned as usize];let mut at=0;
                    while at<count {
                        assert_eq!(at%8,0);let run=if span==0 {count-at} else {(span-(offset+at)%span).min(count-at)};
                        let base=index.eval(offset+at);
                        for j in 0..run {
                            let address=base as i128+slope*j as i128;
                            assert_eq!(index.eval(offset+at+j) as i128,address);
                            local[(at+j) as usize]=source[address as usize];
                        }
                        at+=run;
                    }
                    for j in 0..count {assert_eq!(local[j as usize],source[index.eval(offset+j) as usize]);}
                    assert!(local[count as usize..].iter().all(|&bits|bits==0));
                }}
            }
        }
    }
    #[test]
    fn run_proof_retains_scalar_paths_for_unaligned_and_nonlinear_indices() {
        let lane=Rc::new(Index::Lane);let n=1024;
        assert!(op('%',lane.clone(),c(13),n).aligned_run().is_none());
        assert!(op('*',lane.clone(),c(2),n).aligned_run().is_none());
        assert!(op('*',lane.clone(),lane.clone(),n).run().is_none());
        assert!(op('/',op('+',lane.clone(),c(1),n),c(8),n).run().is_none());
        let combined=op('+',op('*',op('/',lane.clone(),c(16),n),c(16),n),op('%',lane,c(24),n),n);
        assert_eq!(combined.aligned_run(),Some(IndexRun {span:8,slope:1}));
        for base in (0..n).step_by(8) {for j in 0..8 {assert_eq!(combined.eval(base+j),combined.eval(base)+j);}}
    }
    #[test]
    fn injective_patch_proofs_match_independent_coordinate_sets() {
        for rows in [1,2,7] {for cols in [32,96,4096] {for width in [cols,cols+32,cols*2] {
            let n=rows*cols;let lane=Rc::new(Index::Lane);
            let index=op('+',op('*',op('/',lane.clone(),c(cols),n),c(width),n),op('%',lane,c(cols),n),n);
            assert!(index.injective());
            let values:std::collections::HashSet<_>=(0..n).map(|i|index.eval(i)).collect();
            assert_eq!(values.len(),n as usize);
            assert!((0..n).all(|i|index.eval(i)==i/cols*width+i%cols));
        }}}
        let lane=Rc::new(Index::Lane);
        assert!(!op('%',lane.clone(),c(32),64).injective());
        let overlapping=op('+',op('*',op('/',lane.clone(),c(32),64),c(16),64),op('%',lane,c(32),64),64);
        assert!(!overlapping.injective());
        assert!(!Index::Constant(0).injective());
    }
}
impl Index {
    /// Sufficient symbolic affine-run proof; never samples the launch domain.
    pub fn run(&self)->Option<IndexRun> {
        let make=|span,slope|Some(IndexRun {span,slope});
        match self {
            Self::Lane=>make(0,1),Self::Constant(_)=>make(0,0),
            Self::Add(a,b)|Self::Sub(a,b)=>{
                let x=a.run()?;let y=b.run()?;
                let slope=if matches!(self,Self::Add(..)) {x.slope.checked_add(y.slope)?} else {x.slope.checked_sub(y.slope)?};
                make(common_span(x.span,y.span),slope)
            },
            Self::Mul(a,b)=>{
                if let Self::Constant(value)=a.as_ref() {let run=b.run()?;return make(run.span,run.slope.checked_mul(*value as i128)?);}
                if let Self::Constant(value)=b.as_ref() {let run=a.run()?;return make(run.span,run.slope.checked_mul(*value as i128)?);}
                let x=a.run()?;let y=b.run()?;
                if x.slope==0 && y.slope==0 {make(common_span(x.span,y.span),0)} else {None}
            },
            Self::Div(a,d)=>{
                if matches!(a.as_ref(),Self::Lane) {return make(*d,0);}
                let run=a.run()?;
                if run.slope%(*d as i128)==0 {make(run.span,run.slope/(*d as i128))} else {None}
            },
            Self::Mod(a,d)=>{
                if matches!(a.as_ref(),Self::Lane) {return make(*d,1);}
                let run=a.run()?;
                if run.slope%(*d as i128)==0 {make(run.span,0)} else {None}
            },
        }
    }
    /// DMA/local vector starts must be 32-byte aligned: eight FP32 elements.
    pub fn aligned_run(&self)->Option<IndexRun> {
        self.run().filter(|run|(run.span==0 || run.span%8==0) && matches!(run.slope,0|1))
    }
    pub fn binary(op: char, a: Rc<Self>, b: Rc<Self>, elements: u64, max: u64) -> Result<Rc<Self>> {
        let result = match (op, a.as_ref(), b.as_ref()) {
            ('-', x, y) if x == y => Rc::new(Self::Constant(0)),
            ('+' | '-', _, Self::Constant(0)) | ('*' | '/', _, Self::Constant(1)) => a,
            ('+', Self::Constant(0), _) | ('*', Self::Constant(1), _) => b,
            ('*', _, Self::Constant(0)) | ('*', Self::Constant(0), _) => Rc::new(Self::Constant(0)),
            ('+', _, _) => Rc::new(Self::Add(a, b)),
            ('*', _, _) => Rc::new(Self::Mul(a, b)),
            ('/' | '%', _, Self::Constant(d)) if *d != 0 => {
                let d = *d;
                if op=='%' && a.bounds(elements)?.1<d {a}
                else {Rc::new(if op == '/' { Self::Div(a, d) } else { Self::Mod(a, d) })}
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
    /// Sufficient symbolic proofs only: no enumeration or assumption about thread scheduling.
    pub fn injective(&self)->bool {
        match self {
            Self::Lane=>true,
            Self::Add(a,b)=>match (a.as_ref(),b.as_ref()) {
                (_,Self::Constant(_))=>a.injective(),(Self::Constant(_),_)=>b.injective(),
                (Self::Mul(row,stride),Self::Mod(lane,cols))=>match (row.as_ref(),stride.as_ref(),lane.as_ref()) {
                    (Self::Div(origin,divisor),Self::Constant(width),Self::Lane)=>
                        matches!(origin.as_ref(),Self::Lane) && divisor==cols && width>=cols,
                    _=>false,
                },
                _=>false,
            },
            Self::Mul(a,b)=>match (a.as_ref(),b.as_ref()) {
                (_,Self::Constant(value)) if *value>0=>a.injective(),
                (Self::Constant(value),_) if *value>0=>b.injective(),_=>false,
            },
            _=>false,
        }
    }
    #[cfg(test)]
    pub fn eval(&self, lane: u64) -> u64 {
        match self { Self::Lane => lane, Self::Constant(n) => *n, Self::Add(a,b) => a.eval(lane)+b.eval(lane), Self::Sub(a,b) => a.eval(lane)-b.eval(lane), Self::Mul(a,b) => a.eval(lane)*b.eval(lane), Self::Div(a,d) => a.eval(lane)/d, Self::Mod(a,d) => a.eval(lane)%d }
    }
}
