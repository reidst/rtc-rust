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
    #[cfg(test)]
    mod tests {
        use super::*;
    
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
}

mod canvas {
    use std::cmp::PartialEq;
    use std::ops::{Add, Sub, Mul};

    const EPSILON: f64 = 0.00001;
    const CANVAS_BLANK_COLOR: Color = Color { red: 0., green: 0., blue: 0. };

    fn equal(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[derive(Debug, Clone, Copy)]
    struct Color {
        pub red: f64,
        pub green: f64,
        pub blue: f64,
    }

    impl PartialEq<Color> for Color {
        fn eq(&self, other: &Self) -> bool {
            equal(self.red, other.red) &&
            equal(self.green, other.green) &&
            equal(self.blue, other.blue)
        }
    }

    impl Add<Color> for Color {
        type Output = Color;

        fn add(self, rhs: Color) -> Self::Output {
            Color {
                red:    self.red    + rhs.red,
                green:  self.green  + rhs.green,
                blue:   self.blue   + rhs.blue,
            }
        }
    }

    impl Sub<Color> for Color {
        type Output = Color;

        fn sub(self, rhs: Color) -> Self::Output {
            Color {
                red:    self.red    - rhs.red,
                green:  self.green  - rhs.green,
                blue:   self.blue   - rhs.blue,
            }
        }
    }

    impl Mul<f64> for Color {
        type Output = Color;

        fn mul(self, rhs: f64) -> Self::Output {
            Color {
                red:    self.red    * rhs,
                green:  self.green  * rhs,
                blue:   self.blue   * rhs,
            }
        }
    }

    impl Mul<Color> for Color {
        type Output = Color;

        fn mul(self, rhs: Color) -> Self::Output {
            Color {
                red:    self.red    * rhs.red,
                green:  self.green  * rhs.green,
                blue:   self.blue   * rhs.blue,
            }
        }
    }

    impl Color {
        fn new(r: f64, g: f64, b: f64) -> Self {
            Color {
                red: r,
                green: g,
                blue: b,
            }
        }
    }


    struct Canvas {
        pub width: usize,
        pub height: usize,
        grid: Vec<Vec<Color>>,
    }

    impl Canvas {
        fn new(width: usize, height: usize) -> Self {
            let mut grid: Vec<Vec<Color>> = Vec::with_capacity(height);
            for _rows in 0..height {
                let mut row: Vec<Color> = Vec::with_capacity(width);
                for _cols in 0..width {
                    row.push(CANVAS_BLANK_COLOR);
                }
                grid.push(row);
            }

            Canvas { width, height, grid }
        }

        fn pixel_at(&self, x: usize, y: usize) -> Option<Color> {
            let row = self.grid.get(y);
            if let Some(row) = row {
                let item = row.get(x);
                if let Some(item) = item {
                    Some(*item)
                } else {
                    None
                }
            } else {
                None
            }
        }

        fn write_pixel(&mut self, x: usize, y: usize, c: Color) {
            todo!()
        }

        fn to_ppm(&self) -> String {
            todo!()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_color_getters() {
            let c = Color::new(-0.5, 0.4, 1.7);
            assert_eq!(c.red, -0.5);
            assert_eq!(c.green, 0.4);
            assert_eq!(c.blue, 1.7);
        }

        #[test]
        fn test_add_colors() {
            let c1 = Color::new(0.9, 0.6, 0.75);
            let c2 = Color::new(0.7, 0.1, 0.25);
            assert_eq!(c1 + c2, Color::new(1.6, 0.7, 1.0));
        }

        #[test]
        fn test_subtract_colors() {
            let c1 = Color::new(0.9, 0.6, 0.75);
            let c2 = Color::new(0.7, 0.1, 0.25);
            assert_eq!(c1 - c2, Color::new(0.2, 0.5, 0.5));
        }

        #[test]
        fn test_multiply_color_by_scalar() {
            let c = Color::new(0.2, 0.3, 0.4);
            assert_eq!(c * 2., Color::new(0.4, 0.6, 0.8));
        }

        #[test]
        fn test_multiply_colors() {
            let c1 = Color::new(1., 0.2, 0.4);
            let c2 = Color::new(0.9, 1., 0.1);
            assert_eq!(c1 * c2, Color::new(0.9, 0.2, 0.04));
        }

        #[test]
        fn test_create_canvas() {
            let canvas = Canvas::new(10, 20);
            assert_eq!(canvas.width, 10);
            assert_eq!(canvas.height, 20);
            for i in 0..canvas.width {
                for j in 0..canvas.height {
                    assert_eq!(canvas.pixel_at(i, j), Some(Color::new(0.,0.,0.)));
                }
            }
        }

        #[test]
        fn test_write_pixels_to_canvas() {
            let mut canvas = Canvas::new(10, 20);
            let red = Color::new(1., 0., 0.);
            canvas.write_pixel(2, 3, red);
            assert_eq!(canvas.pixel_at(2, 3), Some(red));
        }

        #[test]
        fn test_ppm_header() {
            let canvas = Canvas::new(5, 3);
            let ppm: String = canvas.to_ppm();
            let mut lines = ppm.lines();
            assert_eq!(lines.next(), Some("P3"));
            assert_eq!(lines.next(), Some("5 3"));
            assert_eq!(lines.next(), Some("255"));
        }

        #[test]
        fn test_ppm_pixel_data() {
            let mut canvas = Canvas::new(5, 3);
            let c1 = Color::new(1.5, 0., 0.);
            let c2 = Color::new(0., 0.5, 0.);
            let c3 = Color::new(-0.5, 0., 1.);
            canvas.write_pixel(0, 0, c1);
            canvas.write_pixel(2, 1, c2);
            canvas.write_pixel(4, 2, c3);
            let ppm = canvas.to_ppm();
            let mut lines = ppm.lines().skip(3);
            assert_eq!(lines.next(), Some("255 0 0 0 0 0 0 0 0 0 0 0 0 0 0"));
            assert_eq!(lines.next(), Some("0 0 0 0 0 0 0 128 0 0 0 0 0 0 0"));
            assert_eq!(lines.next(), Some("0 0 0 0 0 0 0 0 0 0 0 0 0 0 255"));
        }

        #[test]
        fn test_ppm_split_long_lines() {
            let mut canvas = Canvas::new(10, 2);
            for i in 0..canvas.width {
                for j in 0..canvas.height {
                    canvas.write_pixel(i, j, Color::new(1., 0.8, 0.6));
                }
            }
            let ppm = canvas.to_ppm();
            let mut lines = ppm.lines()
                .skip(3)
                .map(|line| line.trim());
            assert_eq!(lines.next(), Some("255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204"));
            assert_eq!(lines.next(), Some("153 255 204 153 255 204 153 255 204 153 255 204 153"));
            assert_eq!(lines.next(), Some("255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204"));
            assert_eq!(lines.next(), Some("153 255 204 153 255 204 153 255 204 153 255 204 153"));
        }

        #[test]
        fn test_ppm_ends_with_newline() {
            let canvas = Canvas::new(5, 3);
            let ppm = canvas.to_ppm();
            assert!(ppm.ends_with("\n"));
        }
    }
}
