mod tuple {
    use std::cmp::PartialEq;
    use std::ops::{Add, Sub, Neg, Mul, Div};

    const EPSILON: f64 = 0.00001;

    fn equal(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Tuple {
        pub x: f64,
        pub y: f64,
        pub z: f64,
        pub w: f64,
    }

    impl PartialEq<Tuple> for Tuple {
        fn eq(&self, other: &Self) -> bool {
            equal(self.x, other.x) &&
            equal(self.y, other.y) &&
            equal(self.z, other.z) &&
            equal(self.w, other.w)
        }
    }

    impl Add<Tuple> for Tuple {
        type Output = Tuple;
        fn add(self, rhs: Self) -> Self::Output {
            Tuple {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
                w: self.w + rhs.w,
            }
        }
    }

    impl Sub<Tuple> for Tuple {
        type Output = Tuple;
        fn sub(self, rhs: Tuple) -> Self::Output {
            Tuple {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
                w: self.w - rhs.w,
            }
        }
    }

    impl Mul<f64> for Tuple {
        type Output = Tuple;
        fn mul(self, rhs: f64) -> Self::Output {
            Tuple {
                x: self.x * rhs,
                y: self.y * rhs,
                z: self.z * rhs,
                w: self.w * rhs,
            }
        }
    }

    impl Div<f64> for Tuple {
        type Output = Tuple;
        fn div(self, rhs: f64) -> Self::Output {
            Tuple {
                x: self.x / rhs,
                y: self.y / rhs,
                z: self.z / rhs,
                w: self.w / rhs,
            }
        }
    }

    impl Neg for Tuple {
        type Output = Tuple;
        fn neg(self) -> Self::Output {
            Tuple {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        }
    }

    impl Tuple {
        pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
            Self { x, y, z, w }
        }

        pub fn point(x: f64, y: f64, z: f64) -> Self {
            Self::new(x, y, z, 1.)
        }

        pub fn vector(x: f64, y: f64, z: f64) -> Self {
            Self::new(x, y, z, 0.)
        }

        pub fn is_point(&self) -> bool {
            equal(self.w, 1.)
        }

        pub fn is_vector(&self) -> bool {
            equal(self.w, 0.)
        }

        pub fn magnitude(&self) -> f64 {
            (self.x*self.x + self.y*self.y + self.z*self.z).sqrt()
        }

        pub fn normalized(&self) -> Tuple {
            let normal_scale = 1. / self.magnitude();
            Tuple::new(
                self.x * normal_scale,
                self.y * normal_scale,
                self.z * normal_scale,
                self.w * normal_scale,
            )
        }

        pub fn normalize(&mut self) {
            let normal_scale = 1. / self.magnitude();
            self.x *= normal_scale;
            self.y *= normal_scale;
            self.z *= normal_scale;
            self.w *= normal_scale;
        }

        pub fn dot(&self, other: &Tuple) -> f64 {
            self.x * other.x +
            self.y * other.y +
            self.z * other.z +
            self.w * other.w
        }

        pub fn cross(&self, other: &Tuple) -> Tuple {
            Tuple::vector(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
            )
        }
    }
}
#[cfg(test)]
mod tests_tuple {
    use super::tuple::*;

    #[test]
    fn test_point() {
        let a = Tuple::new(4.3, -4.2, 3.1, 1.0);
        assert_eq!(a.x, 4.3);
        assert_eq!(a.y, -4.2);
        assert_eq!(a.z, 3.1);
        assert_eq!(a.w, 1.0);
        assert!(a.is_point());
        assert!(!a.is_vector());
    }

    #[test]
    fn test_vector() {
        let a = Tuple::new(4.3, -4.2, 3.1, 0.0);
        assert_eq!(a.x, 4.3);
        assert_eq!(a.y, -4.2);
        assert_eq!(a.z, 3.1);
        assert_eq!(a.w, 0.0);
        assert!(!a.is_point());
        assert!(a.is_vector());
    }

    #[test]
    fn test_point_to_tuple() {
        let p: Tuple = Tuple::point(4., -4., 3.).into();
        assert_eq!(p, Tuple::new(4., -4., 3., 1.));
    }

    #[test]
    fn test_vector_to_tuple() {
        let v: Tuple = Tuple::vector(4., -4., 3.).into();
        assert_eq!(v, Tuple::new(4., -4., 3., 0.));
    }

    #[test]
    fn test_add_tuples() {
        let a1 = Tuple::new(3., -2., 5., 1.);
        let a2 = Tuple::new(-2., 3., 1., 0.);
        assert_eq!(a1 + a2, Tuple::new(1., 1., 6., 1.));
    }

    #[test]
    fn test_subtract_points() {
        let p1 = Tuple::point(3., 2., 1.);
        let p2 = Tuple::point(5., 6., 7.);
        assert_eq!(p1 - p2, Tuple::vector(-2., -4., -6.));
    }

    #[test]
    fn test_subtract_vector_from_point() {
        let p = Tuple::point(3., 2., 1.);
        let v = Tuple::vector(5., 6., 7.);
        assert_eq!(p - v, Tuple::point(-2., -4., -6.));
    }

    #[test]
    fn test_subtract_vectors() {
        let v1 = Tuple::vector(3., 2., 1.);
        let v2 = Tuple::vector(5., 6., 7.);
        assert_eq!(v1 - v2, Tuple::vector(-2., -4., -6.));
    }

    #[test]
    fn test_subtract_vector_from_zero() {
        let zero = Tuple::vector(0., 0., 0.);
        let v = Tuple::vector(1., -2., 3.);
        assert_eq!(zero - v, Tuple::vector(-1., 2., -3.));
    }

    #[test]
    fn test_negate_tuple() {
        let a = Tuple::new(1., -2., 3., -4.);
        assert_eq!(-a, Tuple::new(-1., 2., -3., 4.));
    }

    #[test]
    fn test_multiply_tuple_by_scalar() {
        let a = Tuple::new(1., -2., 3., -4.);
        assert_eq!(a * 3.5, Tuple::new(3.5, -7., 10.5, -14.));
    }

    #[test]
    fn test_multiply_tuple_by_fraction() {
        let a = Tuple::new(1., -2., 3., -4.);
        assert_eq!(a * 0.5, Tuple::new(0.5, -1., 1.5, -2.));
    }

    #[test]
    fn test_divide_tuple_by_scalar() {
        let a = Tuple::new(1., -2., 3., -4.);
        assert_eq!(a / 2., Tuple::new(0.5, -1., 1.5, -2.));
    }

    #[test]
    fn test_magnitude_unit_vector_x() {
        let v = Tuple::vector(1., 0., 0.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn test_magnitude_unit_vector_y() {
        let v = Tuple::vector(0., 1., 0.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn test_magnitude_unit_vector_z() {
        let v = Tuple::vector(0., 0., 1.);
        assert_eq!(v.magnitude(), 1.);
    }

    #[test]
    fn test_magnitude_vector_123() {
        let v = Tuple::vector(1., 2., 3.);
        assert_eq!(v.magnitude(), 14_f64.sqrt());
    }

    #[test]
    fn test_magnitude_vector_negative_123() {
        let v = Tuple::vector(-1., -2., -3.);
        assert_eq!(v.magnitude(), 14_f64.sqrt());
    }

    #[test]
    fn test_normalize_vector_400() {
        let v = Tuple::vector(4., 0., 0.);
        assert_eq!(v.normalized(), Tuple::vector(1., 0., 0.));
    }

    #[test]
    fn test_normalize_vector_123() {
        let v = Tuple::vector(1., 2., 3.);
        let denom = 14_f64.sqrt();
        let expected_normal = Tuple::vector(1./denom, 2./denom, 3./denom);
        assert_eq!(v.normalized(), expected_normal);
    }

    #[test]
    fn test_vector_dot_product() {
        let a = Tuple::vector(1., 2., 3.);
        let b = Tuple::vector(2., 3., 4.);
        assert_eq!(a.dot(&b), 20.);
    }

    #[test]
    fn test_vector_cross_product() {
        let a = Tuple::vector(1., 2., 3.);
        let b = Tuple::vector(2., 3., 4.);
        assert_eq!(a.cross(&b), Tuple::vector(-1., 2., -1.));
        assert_eq!(b.cross(&a), Tuple::vector(1., -2., 1.));
    }
}