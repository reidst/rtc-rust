const EPSILON: f64 = 0.00001;

fn equal(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

pub mod tuple {
    use std::cmp::PartialEq;
    use std::ops::{Add, Sub, Neg, Mul, Div};
    use crate::equal;

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

        pub fn normalize(&self) -> Tuple {
            let normal_scale = 1. / self.magnitude();
            Tuple::new(
                self.x * normal_scale,
                self.y * normal_scale,
                self.z * normal_scale,
                self.w * normal_scale,
            )
        }

        pub fn normalize_ip(&mut self) {
            let normal_scale = 1. / self.magnitude();
            self.x *= normal_scale;
            self.y *= normal_scale;
            self.z *= normal_scale;
            self.w *= normal_scale;
        }

        pub fn dot(&self, other: Tuple) -> f64 {
            self.x * other.x +
            self.y * other.y +
            self.z * other.z +
            self.w * other.w
        }

        pub fn cross(&self, other: Tuple) -> Tuple {
            Tuple::vector(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x,
            )
        }

        pub fn reflect(&self, normal: Tuple) -> Tuple {
            assert!(self.is_vector(), "only vectors can be reflected");
            assert!(normal.is_vector(), "a valid normal must be a vector");
            self.to_owned() - normal * 2. * self.dot(normal)
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
            assert_eq!(v.normalize(), Tuple::vector(1., 0., 0.));
        }
    
        #[test]
        fn test_normalize_vector_123() {
            let v = Tuple::vector(1., 2., 3.);
            let denom = 14_f64.sqrt();
            let expected_normal = Tuple::vector(1./denom, 2./denom, 3./denom);
            assert_eq!(v.normalize(), expected_normal);
        }
    
        #[test]
        fn test_vector_dot_product() {
            let a = Tuple::vector(1., 2., 3.);
            let b = Tuple::vector(2., 3., 4.);
            assert_eq!(a.dot(b), 20.);
        }
    
        #[test]
        fn test_vector_cross_product() {
            let a = Tuple::vector(1., 2., 3.);
            let b = Tuple::vector(2., 3., 4.);
            assert_eq!(a.cross(b), Tuple::vector(-1., 2., -1.));
            assert_eq!(b.cross(a), Tuple::vector(1., -2., 1.));
        }
    }
}

pub mod canvas {
    use std::cmp::PartialEq;
    use std::ops::{Add, Sub, Mul};
    use crate::equal;

    const CANVAS_BLANK_COLOR: Color = Color { red: 0., green: 0., blue: 0. };

    #[derive(Debug, Clone, Copy)]
    pub struct Color {
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
        pub fn new(r: f64, g: f64, b: f64) -> Self {
            Color {
                red: r,
                green: g,
                blue: b,
            }
        }
    }


    pub struct Canvas {
        pub width: usize,
        pub height: usize,
        grid: Vec<Vec<Color>>,
    }

    impl Canvas {
        pub fn new(width: usize, height: usize) -> Self {
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

        fn bounds_check(&self, x: usize, y: usize) {
            if x >= self.width {
                panic!(
                    "Illegal canvas x coordinate {}: the canvas width is {}",
                    x,
                    self.width,
                )
            }
            if y >= self.height {
                panic!(
                    "Illegal canvas y coordinate {}: the canvas height is {}",
                    y,
                    self.height,
                )
            }
        }

        pub fn pixel_at(&self, x: usize, y: usize) -> Color {
            self.bounds_check(x, y);
            self.grid[y][x]
        }

        pub fn write_pixel(&mut self, x: usize, y: usize, c: Color) {
            if (0..self.width).contains(&x) && (0..self.height).contains(&y) {
                self.grid[y][x] = c;
            }
        }

        pub fn to_ppm(&self) -> String {
            let mut data = String::new();
            data.push_str("P3\n");
            data.push_str(format!("{} {}\n", self.width, self.height).as_str());
            data.push_str("255\n");
            for row in self.grid.iter() {
                let mut line = String::new();
                for color in row.iter() {
                    for component in [color.red, color.green, color.blue] {
                        let num = (component * 256.) as usize;
                        let num = num.clamp(0, 255);
                        let num_str = num.to_string() + " ";
                        if line.len() + num_str.len() > 70 { // current line too long
                            data.push_str(line.as_str().trim());
                            data.push('\n');
                            line = String::new();
                        }
                        line.push_str(num_str.as_str());
                    }
                }
                data.push_str(line.as_str().trim());
                data.push('\n');
            }
            
            data
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
                    assert_eq!(canvas.pixel_at(i, j), Color::new(0.,0.,0.));
                }
            }
        }

        #[test]
        fn test_write_pixels_to_canvas() {
            let mut canvas = Canvas::new(10, 20);
            let red = Color::new(1., 0., 0.);
            canvas.write_pixel(2, 3, red);
            assert_eq!(canvas.pixel_at(2, 3), red);
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

pub mod matrix {
    use std::cmp::PartialEq;
    use std::ops::Mul;

    use crate::tuple::Tuple;
    use crate::equal;

    fn bounds_check(r: usize, c: usize, size: usize) {
        if r >= size || c >= size {
            panic!("a Matrix{size}x{size} was indexed at [{r},{c}]");
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Matrix2x2 {
        data: [[f64; 2]; 2],
    }

    impl PartialEq<Matrix2x2> for Matrix2x2 {
        fn eq(&self, other: &Matrix2x2) -> bool {
            equal(self.data[0][0], other.data[0][0]) &&
            equal(self.data[0][1], other.data[0][1]) &&
            equal(self.data[1][0], other.data[1][0]) &&
            equal(self.data[1][1], other.data[1][1])
        }
    }

    impl Matrix2x2 {
        pub fn new(contents: [f64; 4]) -> Self {
            Matrix2x2 { data: [
                [contents[0], contents[1]],
                [contents[2], contents[3]],
            ] }
        }

        pub fn get(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 2);
            self.data[r][c]
        }

        pub fn determinant(&self) -> f64 {
            self.data[0][0] * self.data[1][1] -
            self.data[0][1] * self.data[1][0]
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Matrix3x3 {
        data: [[f64; 3]; 3],
    }

    impl PartialEq<Matrix3x3> for Matrix3x3 {
        fn eq(&self, other: &Matrix3x3) -> bool {
            for r in 0..3 {
                for c in 0..3 {
                    if !equal(self.data[r][c], other.data[r][c]) {
                        return false
                    }
                }
            }
            true
        }
    }

    impl Matrix3x3 {
        pub fn new(contents: [f64; 9]) -> Self {
            let mut data = [[0f64; 3]; 3];
            for (i, &v) in contents.iter().enumerate() {
                data[i / 3][i % 3] = v;
            }
            Matrix3x3 { data }
        }

        pub fn get(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 3);
            self.data[r][c]
        }

        pub fn submatrix(&self, r: usize, c: usize) -> Matrix2x2 {
            bounds_check(r, c, 3);
            let mut out = Matrix2x2::new([0.; 4]);
            for row in 0..3 {
                if row == r { continue }
                let target_row = if row < r { row } else { row - 1 };
                for col in 0..3 {
                    if col == c { continue }
                    let target_col = if col < c { col } else { col - 1 };
                    out.data[target_row][target_col] = self.data[row][col];
                }
            }
            out
        }

        pub fn minor(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 3);
            self.submatrix(r, c).determinant()
        }

        pub fn cofactor(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 3);
            if (r + c) % 2 == 1 {
                -self.minor(r, c)
            } else {
                self.minor(r, c)
            }
        }

        pub fn determinant(&self) -> f64 {
            self.data[0][0] * self.cofactor(0, 0) +
            self.data[0][1] * self.cofactor(0, 1) +
            self.data[0][2] * self.cofactor(0, 2)
        }
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Matrix4x4 {
        data: [[f64; 4]; 4],
    }

    impl PartialEq<Matrix4x4> for Matrix4x4 {
        fn eq(&self, other: &Matrix4x4) -> bool {
            for r in 0..4 {
                for c in 0..4 {
                    if !equal(self.data[r][c], other.data[r][c]) {
                        return false
                    }
                }
            }
            true
        }
    }

    impl Mul<Matrix4x4> for Matrix4x4 {
        type Output = Matrix4x4;

        fn mul(self, rhs: Matrix4x4) -> Self::Output {
            let mut out = Matrix4x4::new([0.; 16]);
            for row in 0..4 {
                for col in 0..4 {
                    out.data[row][col] = 
                        self.data[row][0] * rhs.data[0][col] +
                        self.data[row][1] * rhs.data[1][col] +
                        self.data[row][2] * rhs.data[2][col] +
                        self.data[row][3] * rhs.data[3][col];
                }
            }
            out
        }
    }

    impl Mul<Tuple> for Matrix4x4 {
        type Output = Tuple;

        fn mul(self, rhs: Tuple) -> Self::Output {
            let mut out = [0f64; 4];
            let rhs = [rhs.x, rhs.y, rhs.z, rhs.w];
            for row in 0..4 {
                out[row] = 
                    self.data[row][0] * rhs[0] +
                    self.data[row][1] * rhs[1] +
                    self.data[row][2] * rhs[2] +
                    self.data[row][3] * rhs[3];
            }
            Tuple::new(out[0], out[1], out[2], out[3])
        }
    }

    impl Matrix4x4 {
        pub fn new(contents: [f64; 16]) -> Self {
            let mut data = [[0f64; 4]; 4];
            for (i, &v) in contents.iter().enumerate() {
                data[i / 4][i % 4] = v;
            }
            Matrix4x4 { data }
        }

        pub fn identity() -> Self {
            Matrix4x4 { data: [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ]}
        }

        pub fn get(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 4);
            self.data[r][c]
        }

        pub fn transpose(&self) -> Matrix4x4 {
            let mut data = [[0f64; 4]; 4];
            for row in 0..4 {
                for col in 0..4 {
                    data[col][row] = self.data[row][col];
                }
            }
            Matrix4x4 { data }
        }

        pub fn submatrix(&self, r: usize, c: usize) -> Matrix3x3 {
            bounds_check(r, c, 4);
            let mut out = Matrix3x3::new([0.; 9]);
            for row in 0..4 {
                if row == r { continue }
                let target_row = if row < r { row } else { row - 1 };
                for col in 0..4 {
                    if col == c { continue }
                    let target_col = if col < c { col } else { col - 1 };
                    out.data[target_row][target_col] = self.data[row][col];
                }
            }
            out
        }

        pub fn minor(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 4);
            self.submatrix(r, c).determinant()
        }

        pub fn cofactor(&self, r: usize, c: usize) -> f64 {
            bounds_check(r, c, 4);
            if (r + c) % 2 == 1 {
                -self.minor(r, c)
            } else {
                self.minor(r, c)
            }
        }

        pub fn determinant(&self) -> f64 {
            self.data[0][0] * self.cofactor(0, 0) +
            self.data[0][1] * self.cofactor(0, 1) +
            self.data[0][2] * self.cofactor(0, 2) +
            self.data[0][3] * self.cofactor(0, 3)
        }

        pub fn invertible(&self) -> bool {
            self.determinant() != 0.
        }

        pub fn inverse(&self) -> Matrix4x4 {
            if !self.invertible() { panic!("Tried to invert a non-invertible matrix") }
            let det = self.determinant();
            let mut out = [0.; 16];
            for row in 0..4 {
                for col in 0..4 {
                    out[col * 4 + row] = self.cofactor(row, col) / det;
                }
            }
            Matrix4x4::new(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::tuple::Tuple;

        use super::*;

        #[test]
        fn test_construct_matrix4() {
            let m = Matrix4x4::new([
                 1.,   2.,   3.,   4.,
                 5.5,  6.5,  7.5,  8.5,
                 9.,  10.,  11.,  12.,
                13.5, 14.5, 15.5, 16.5,
            ]);
            assert_eq!(m.get(0, 0), 1.);
            assert_eq!(m.get(0, 3), 4.);
            assert_eq!(m.get(1, 0), 5.5);
            assert_eq!(m.get(1, 2), 7.5);
            assert_eq!(m.get(2, 2), 11.);
            assert_eq!(m.get(3, 0), 13.5);
            assert_eq!(m.get(3, 2), 15.5);
        }

        #[test]
        fn test_construct_matrix2() {
            let m = Matrix2x2::new([
                -3.,  5.,
                 1., -2.,
            ]);
            assert_eq!(m.get(0, 0), -3.);
            assert_eq!(m.get(0, 1), 5.);
            assert_eq!(m.get(1, 0), 1.);
            assert_eq!(m.get(1, 1), -2.);
        }

        #[test]
        fn test_construct_matrix3() {
            let m = Matrix3x3::new([
                -3.,  5.,  0.,
                 1., -2., -7.,
                 0.,  1.,  1.,
            ]);
            assert_eq!(m.get(0, 0), -3.);
            assert_eq!(m.get(1, 1), -2.);
            assert_eq!(m.get(2, 2), 1.);
        }

        #[test]
        fn test_matrix_equality() {
            let a = Matrix4x4::new([
                1., 2., 3., 4.,
                5., 6., 7., 8.,
                9., 8., 7., 6.,
                5., 4., 3., 2.,
            ]);
            let b = Matrix4x4::new([
                1., 2., 3., 4.,
                5., 6., 7., 8.,
                9., 8., 7., 6.,
                5., 4., 3., 2.,
            ]);
            assert_eq!(a, b);
        }

        #[test]
        fn test_matrix_inequality() {
            let a = Matrix4x4::new([
                1., 2., 3., 4.,
                5., 6., 7., 8.,
                9., 8., 7., 6.,
                5., 4., 3., 2.,
            ]);
            let b = Matrix4x4::new([
                2., 3., 4., 5.,
                6., 7., 8., 9.,
                8., 7., 6., 5.,
                4., 3., 2., 1.,
            ]);
            assert_ne!(a, b);
        }

        #[test]
        fn test_matrix_multiplication() {
            let a = Matrix4x4::new([
                1., 2., 3., 4.,
                5., 6., 7., 8.,
                9., 8., 7., 6.,
                5., 4., 3., 2.,
            ]);
            let b = Matrix4x4::new([
                -2., 1., 2.,  3.,
                 3., 2., 1., -1.,
                 4., 3., 6.,  5.,
                 1., 2., 7.,  8.,
            ]);
            assert_eq!(a * b, Matrix4x4::new([
                20., 22.,  50.,  48.,
                44., 54., 114., 108.,
                40., 58., 110., 102.,
                16., 26.,  46.,  42.,
            ]));
        }

        #[test]
        fn test_matrix_tuple_multiplication() {
            let a = Matrix4x4::new([
                1., 2., 3., 4.,
                2., 4., 4., 2.,
                8., 6., 4., 1.,
                0., 0., 0., 1.,
            ]);
            let b = Tuple::new(1., 2., 3., 1.);
            assert_eq!(a * b, Tuple::new(18., 24., 33., 1.));
        }

        #[test]
        fn test_matrix_identity_multiplication() {
            let a = Matrix4x4::new([
                0., 1.,  2.,  4.,
                1., 2.,  4.,  8.,
                2., 4.,  8., 16.,
                4., 8., 16., 32.,
            ]);
            assert_eq!(a * Matrix4x4::identity(), a);
        }

        #[test]
        fn test_transpose_matrix() {
            let a = Matrix4x4::new([
                0., 9., 3., 0.,
                9., 8., 0., 8.,
                1., 8., 5., 3.,
                0., 0., 5., 8.,
            ]);
            assert_eq!(a.transpose(), Matrix4x4::new([
                0., 9., 1., 0.,
                9., 8., 8., 0.,
                3., 0., 5., 5.,
                0., 8., 3., 8.,
            ]));
        }

        #[test]
        fn test_transpose_identity_matrix() {
            let a = Matrix4x4::identity().transpose();
            assert_eq!(a, Matrix4x4::identity());
        }

        #[test]
        fn test_determinant2() {
            let a = Matrix2x2::new([
                 1., 5.,
                -3., 2.,
            ]);
            assert_eq!(a.determinant(), 17.);
        }

        #[test]
        fn test_submatrix3() {
            let a = Matrix3x3::new([
                 1., 5.,  0.,
                -3., 2.,  7.,
                 0., 6., -3.,
            ]);
            assert_eq!(a.submatrix(0, 2), Matrix2x2::new([
                -3., 2.,
                 0., 6.,
            ]));
        }

        #[test]
        fn test_submatrix4() {
            let a = Matrix4x4::new([
                -6., 1.,  1., 6.,
                -8., 5.,  8., 6.,
                -1., 0.,  8., 2.,
                -7., 1., -1., 1.,
            ]);
            assert_eq!(a.submatrix(2, 1), Matrix3x3::new([
                -6.,  1., 6.,
                -8.,  8., 6.,
                -7., -1., 1.,
            ]));
        }

        #[test]
        fn test_minor3() {
            let a = Matrix3x3::new([
                3.,  5.,  0.,
                2., -1., -7.,
                6., -1.,  5.,
            ]);
            let b = a.submatrix(1, 0);
            assert_eq!(b.determinant(), 25.);
            assert_eq!(a.minor(1, 0), 25.);
        }

        #[test]
        fn test_cofactor3() {
            let a = Matrix3x3::new([
                3.,  5.,  0.,
                2., -1., -7.,
                6., -1.,  5.,
            ]);
            assert_eq!(a.minor(0, 0), -12.);
            assert_eq!(a.cofactor(0, 0), -12.);
            assert_eq!(a.minor(1, 0), 25.);
            assert_eq!(a.cofactor(1, 0), -25.);
        }

        #[test]
        fn test_determinant3() {
            let a = Matrix3x3::new([
                 1., 2.,  6.,
                -5., 8., -4.,
                 2., 6.,  4.,
            ]);
            assert_eq!(a.cofactor(0, 0), 56.);
            assert_eq!(a.cofactor(0, 1), 12.);
            assert_eq!(a.cofactor(0, 2), -46.);
            assert_eq!(a.determinant(), -196.);
        }

        #[test]
        fn test_determinant4() {
            let a = Matrix4x4::new([
                -2., -8.,  3.,  5.,
                -3.,  1.,  7.,  3.,
                 1.,  2., -9.,  6.,
                -6.,  7.,  7., -9.,
            ]);
            assert_eq!(a.cofactor(0, 0), 690.);
            assert_eq!(a.cofactor(0, 1), 447.);
            assert_eq!(a.cofactor(0, 2), 210.);
            assert_eq!(a.cofactor(0, 3), 51.);
            assert_eq!(a.determinant(), -4071.);
        }

        #[test]
        fn test_invertible() {
            let a = Matrix4x4::new([
                6.,  4., 4.,  4.,
                5.,  5., 7.,  6.,
                4., -9., 3., -7.,
                9.,  1., 7., -6.,
            ]);
            assert_eq!(a.determinant(), -2120.);
            assert!(a.invertible());
        }

        #[test]
        fn test_noninvertible() {
            let a = Matrix4x4::new([
                -4.,  2., -2., -3.,
                 9.,  6.,  2.,  6.,
                 0., -5.,  1., -5.,
                 0.,  0.,  0.,  0.,
            ]);
            assert_eq!(a.determinant(), 0.);
            assert!(!a.invertible());
        }

        #[test]
        fn test_invert_matrix1() {
            let a = Matrix4x4::new([
                -5.,  2.,  6., -8.,
                 1., -5.,  1.,  8.,
                 7.,  7., -6., -7.,
                 1., -3.,  7.,  4.,
            ]);
            let b = a.inverse();
            assert_eq!(a.determinant(), 532.);
            assert_eq!(a.cofactor(2, 3), -160.);
            assert_eq!(b.get(3, 2), -160. / 532.);
            assert_eq!(a.cofactor(3, 2), 105.);
            assert_eq!(b.get(2, 3), 105. / 532.);
            assert_eq!(b, Matrix4x4::new([
                 0.21805,  0.45113,  0.24060, -0.04511,
                -0.80827, -1.45677, -0.44361,  0.52068,
                -0.07895, -0.22368, -0.05263,  0.19737,
                -0.52256, -0.81391, -0.30075,  0.30639,
            ]));
        }

        #[test]
        fn test_invert_matrix2() {
            let a = Matrix4x4::new([
                 8., -5.,  9.,  2.,
                 7.,  5.,  6.,  1.,
                -6.,  0.,  9.,  6.,
                -3.,  0., -9., -4.,
            ]);
            assert_eq!(a.inverse(), Matrix4x4::new([
                -0.15385, -0.15385, -0.28205, -0.53846,
                -0.07692,  0.12308,  0.02564,  0.03077,
                 0.35897,  0.35897,  0.43590,  0.92308,
                 -0.69231, -0.69231, -0.76923, -1.92308,
            ]));
        }

        #[test]
        fn test_invert_matrix3() {
            let a = Matrix4x4::new([
                 9.,  3.,  0.,  9.,
                -5., -2., -6., -3.,
                -4.,  9.,  6.,  4.,
                -7.,  6.,  6.,  2.,
            ]);
            assert_eq!(a.inverse(), Matrix4x4::new([
                -0.04074, -0.07778,  0.14444, -0.22222,
                -0.07778,  0.03333,  0.36667, -0.33333,
                -0.02901, -0.14630, -0.10926,  0.12963,
                 0.17778,  0.06667, -0.26667,  0.33333,
            ]));
        }

        #[test]
        fn multiply_matrix_product_by_inverse() {
            let a = Matrix4x4::new([
                 3., -9.,  7.,  3.,
                 3., -8.,  2., -9.,
                -4.,  4.,  4.,  1.,
                -6.,  5., -1.,  1.,
            ]);
            let b = Matrix4x4::new([
                8.,  2.,  2.,  2., 
                3., -1.,  7.,  0., 
                7.,  0.,  5.,  4., 
                6., -2.,  0.,  5., 
            ]);
            let c = a * b;
            assert_eq!(c * b.inverse(), a);
        }
    }
}

pub mod transformation {
    use std::cmp::PartialEq;
    use std::ops::Mul;
    use crate::tuple::Tuple;
    use crate::matrix::Matrix4x4;

    #[derive(Debug, Copy, Clone)]
    pub struct Transform {
        mat: Matrix4x4,
    }

    impl PartialEq<Transform> for Transform {
        fn eq(&self, other: &Transform) -> bool {
            self.mat == other.mat
        }
    }

    impl Mul<Tuple> for Transform {
        type Output = Tuple;

        fn mul(self, rhs: Tuple) -> Self::Output {
            self.mat * rhs
        }
    }

    impl Mul<Transform> for Transform {
        type Output = Transform;

        fn mul(self, rhs: Transform) -> Self::Output {
            Transform { mat: self.mat * rhs.mat }
        }
    }

    impl Transform {
        pub fn from(mat: Matrix4x4) -> Self {
            Transform { mat }
        }

        pub fn identity() -> Self {
            Transform { mat: Matrix4x4::identity() }
        }

        pub fn translation(x: f64, y: f64, z: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                1., 0., 0.,  x,
                0., 1., 0.,  y,
                0., 0., 1.,  z,
                0., 0., 0., 1.,
            ]) }
        }

        pub fn scaling(x: f64, y: f64, z: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                 x, 0., 0., 0.,
                0.,  y, 0., 0.,
                0., 0.,  z, 0.,
                0., 0., 0., 1.,
            ]) }
        }

        pub fn rotation_x(rad: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                1.,        0.,         0., 0.,
                0., rad.cos(), -rad.sin(), 0.,
                0., rad.sin(),  rad.cos(), 0.,
                0.,        0.,         0., 1.,
            ]) }
        }

        pub fn rotation_y(rad: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                 rad.cos(), 0., rad.sin(), 0.,
                        0., 1.,        0., 0.,
                -rad.sin(), 0., rad.cos(), 0.,
                        0., 0.,        0., 1.,
            ]) }
        }

        pub fn rotation_z(rad: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                rad.cos(), -rad.sin(), 0., 0.,
                rad.sin(),  rad.cos(), 0., 0.,
                       0.,         0., 1., 0.,
                       0.,         0., 0., 1.,
            ]) }
        }

        pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
            Transform { mat: Matrix4x4::new([
                1., xy, xz, 0.,
                yx, 1., yz, 0.,
                zx, zy, 1., 0.,
                0., 0., 0., 1.,
            ]) }
        }

        pub fn inverse(&self) -> Self {
            Transform { mat: self.mat.inverse() }
        }

        pub fn transpose(&self) -> Self {
            Transform { mat: self.mat.transpose() }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::tuple::Tuple;

        const PI: f64 = std::f64::consts::PI;
        const SQRT_2: f64 = std::f64::consts::SQRT_2;

        #[test]
        fn test_translate_point() {
            let transform = Transform::translation(5., -3., 2.);
            let p = Tuple::point(-3., 4., 5.);
            assert_eq!(transform * p, Tuple::point(2., 1., 7.));
        }

        #[test]
        fn test_inverse_translation() {
            let transform = Transform::translation(5., -3., 2.);
            let inv = transform.inverse();
            let p = Tuple::point(-3., 4., 5.);
            assert_eq!(inv * p, Tuple::point(-8., 7., 3.));
        }

        #[test]
        fn test_translation_not_affect_vectors() {
            let transform = Transform::translation(5., -3., 2.);
            let v = Tuple::vector(-3., 4., 5.);
            assert_eq!(transform * v, v);
        }

        #[test]
        fn test_scale_point() {
            let transform = Transform::scaling(2., 3., 4.);
            let p = Tuple::point(-4., 6., 8.);
            assert_eq!(transform * p, Tuple::point(-8., 18., 32.));
        }

        #[test]
        fn test_scale_vector() {
            let transform = Transform::scaling(2., 3., 4.);
            let v = Tuple::vector(-4., 6., 8.);
            assert_eq!(transform * v, Tuple::vector(-8., 18., 32.));
        }

        #[test]
        fn test_multiply_by_scaling_inverse() {
            let transform = Transform::scaling(2., 3., 4.);
            let inv = transform.inverse();
            let v = Tuple::vector(-4., 6., 8.);
            assert_eq!(inv * v, Tuple::vector(-2., 2., 2.));
        }

        #[test]
        fn test_reflection_as_negative_scaling() {
            let transform = Transform::scaling(-1., 1., 1.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(-2., 3., 4.));
        }

        #[test]
        fn test_rotation_x() {
            let p = Tuple::point(0., 1., 0.);
            let half_quarter = Transform::rotation_x(PI / 4.);
            let full_quarter = Transform::rotation_x(PI / 2.);
            assert_eq!(half_quarter * p, Tuple::point(0., SQRT_2 / 2., SQRT_2 / 2.));
            assert_eq!(full_quarter * p, Tuple::point(0., 0., 1.));
        }

        #[test]
        fn test_inverse_rotation_x() {
            let p = Tuple::point(0., 1., 0.);
            let half_quarter = Transform::rotation_x(PI / 4.);
            let inv = half_quarter.inverse();
            assert_eq!(inv * p, Tuple::point(0., SQRT_2 / 2., -SQRT_2 / 2.));
        }

        #[test]
        fn test_rotation_y() {
            let p = Tuple::point(0., 0., 1.);
            let half_quarter = Transform::rotation_y(PI / 4.);
            let full_quarter = Transform::rotation_y(PI / 2.);
            assert_eq!(half_quarter * p, Tuple::point(SQRT_2 / 2., 0., SQRT_2 / 2.));
            assert_eq!(full_quarter * p, Tuple::point(1., 0., 0.));
        }

        #[test]
        fn test_rotation_z() {
            let p = Tuple::point(0., 1., 0.);
            let half_quarter = Transform::rotation_z(PI / 4.);
            let full_quarter = Transform::rotation_z(PI / 2.);
            assert_eq!(half_quarter * p, Tuple::point(-SQRT_2 / 2., SQRT_2 / 2., 0.));
            assert_eq!(full_quarter * p, Tuple::point(-1., 0., 0.));
        }

        #[test]
        fn test_shear_xy() {
            let transform = Transform::shearing(1., 0., 0., 0., 0., 0.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(5., 3., 4.));
        }

        #[test]
        fn test_shear_xz() {
            let transform = Transform::shearing(0., 1., 0., 0., 0., 0.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(6., 3., 4.));
        }

        #[test]
        fn test_shear_yx() {
            let transform = Transform::shearing(0., 0., 1., 0., 0., 0.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(2., 5., 4.));
        }

        #[test]
        fn test_shear_yz() {
            let transform = Transform::shearing(0., 0., 0., 1., 0., 0.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(2., 7., 4.));
        }

        #[test]
        fn test_shear_zx() {
            let transform = Transform::shearing(0., 0., 0., 0., 1., 0.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(2., 3., 6.));
        }

        #[test]
        fn test_shear_zy() {
            let transform = Transform::shearing(0., 0., 0., 0., 0., 1.);
            let p = Tuple::point(2., 3., 4.);
            assert_eq!(transform * p, Tuple::point(2., 3., 7.));
        }

        #[test]
        fn test_apply_transforms_in_sequence() {
            let p = Tuple::point(1., 0., 1.);
            let a = Transform::rotation_x(PI / 2.);
            let b = Transform::scaling(5., 5., 5.);
            let c = Transform::translation(10., 5., 7.);
            // apply rotation first
            let p2 = a * p;
            assert_eq!(p2, Tuple::point(1., -1., 0.));
            // then apply scaling
            let p3 = b * p2;
            assert_eq!(p3, Tuple::point(5., -5., 0.));
            // then apply translation
            let p4 = c * p3;
            assert_eq!(p4, Tuple::point(15., 0., 7.));
        }

        #[test]
        fn test_apply_transforms_in_reversed_chain() {
            let p = Tuple::point(1., 0., 1.);
            let a = Transform::rotation_x(PI / 2.);
            let b = Transform::scaling(5., 5., 5.);
            let c = Transform::translation(10., 5., 7.);
            let t = c * b * a;
            assert_eq!(t * p, Tuple::point(15., 0., 7.));
        }
    }
}

pub mod intersection {
    use std::cmp::PartialEq;
    use std::ptr;

    use crate::shading::Material;
    use crate::tuple::Tuple;
    use crate::transformation::Transform;

    #[derive(Debug, Copy, Clone)]
    pub struct Ray {
        pub origin: Tuple,
        pub direction: Tuple,
    }

    impl Ray {
        pub fn new(origin: Tuple, direction: Tuple) -> Self {
            assert!(origin.is_point(), "A ray was made with a non-point origin: {origin:?}");
            assert!(direction.is_vector(), "A ray was made with a non-vector direction: {direction:?}");
            Ray { origin, direction }
        }

        pub fn position(&self, t: f64) -> Tuple {
            self.origin + self.direction * t
        }

        pub fn transform(&self, trans: Transform) -> Self {
            Ray {
                origin: trans * self.origin,
                direction: trans * self.direction,
            }
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Sphere {
        pub transform: Transform,
        pub material: Material,
    }

    impl Sphere {
        pub fn new() -> Self {
            Sphere {
                transform: Transform::identity(),
                material: Material::new(),
            }
        }

        pub fn intersect(&self, ray: Ray) -> Vec<Intersection> {
            let ray = ray.transform(self.transform.inverse());
            let mut xs = Vec::new();
            let sphere_to_ray = ray.origin - Tuple::point(0., 0., 0.);
            let a = ray.direction.dot(ray.direction);
            let b = ray.direction.dot(sphere_to_ray) * 2.;
            let c = sphere_to_ray.dot(sphere_to_ray) - 1.;
            let discriminant = b * b - 4. * a * c;
            if discriminant >= 0. {
                xs.push(Intersection::new(
                    (-b - discriminant.sqrt()) / (2. * a),
                    &self,
                ));
                xs.push(Intersection::new(
                    (-b + discriminant.sqrt()) / (2. * a),
                    &self,
                ));
            }
            xs
        }

        pub fn set_transform(&mut self, trans: Transform) {
            self.transform = trans;
        }

        pub fn chain_transform(&mut self, trans: Transform) {
            self.set_transform(self.transform * trans);
        }

        pub fn normal_at(&self, world_point: Tuple) -> Tuple {
            let object_point = self.transform.inverse() * world_point;
            let object_normal = object_point - Tuple::point(0., 0., 0.);
            let mut world_normal = self.transform.inverse().transpose() * object_normal;
            world_normal.w = 0.;
            world_normal.normalize()
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Intersection<'a> {
        pub t: f64,
        pub object: &'a Sphere,
    }

    impl<'a> PartialEq<Intersection<'a>> for Intersection<'a> {
        fn eq(&self, other: &Intersection<'a>) -> bool {
            self.t == other.t &&
            ptr::eq(self.object, other.object)
        }
    }

    impl<'a> Intersection<'a> {
        fn new(t: f64, object: &'a Sphere) -> Self {
            Intersection { t, object }
        }
    }
    
    pub fn hit(intersections: Vec<Intersection>) -> Option<Intersection> {
        let mut hit = None;
        for i in intersections {
            if i.t < 0. { continue }
            match hit {
                None => hit = Some(i),
                Some(h) if i.t < h.t => hit = Some(i),
                _ => {},
            }
        }
        hit
    }

    #[cfg(test)]
    mod tests {
        use std::ptr;
        use super::*;
        use crate::tuple::Tuple;
        use crate::transformation::Transform;

        #[test]
        fn test_create_ray() {
            let origin = Tuple::point(1., 2., 3.);
            let direction = Tuple::vector(4., 5., 6.);
            let r = Ray::new(origin, direction);
            assert_eq!(r.origin, origin);
            assert_eq!(r.direction, direction);
        }

        #[test]
        fn test_compute_point_from_distance() {
            let r = Ray::new(Tuple::point(2., 3., 4.), Tuple::vector(1., 0., 0.));
            assert_eq!(r.position(0.), Tuple::point(2., 3., 4.));
            assert_eq!(r.position(1.), Tuple::point(3., 3., 4.));
            assert_eq!(r.position(-1.), Tuple::point(1., 3., 4.));
            assert_eq!(r.position(2.5), Tuple::point(4.5, 3., 4.));
        }

        #[test]
        fn test_ray_intersects_sphere_twice() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, 4.);
            assert_eq!(xs[1].t, 6.);
        }

        #[test]
        fn test_ray_intersects_sphere_at_tangent() {
            let r = Ray::new(Tuple::point(0., 1., -5.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, 5.);
            assert_eq!(xs[1].t, 5.);
        }

        #[test]
        fn test_ray_misses_sphere() {
            let r = Ray::new(Tuple::point(0., 2., -5.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 0);
        }

        #[test]
        fn test_ray_originates_inside_sphere() {
            let r = Ray::new(Tuple::point(0., 0., 0.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, -1.);
            assert_eq!(xs[1].t, 1.);
        }

        #[test]
        fn test_sphere_behind_ray() {
            let r = Ray::new(Tuple::point(0., 0., 5.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, -6.);
            assert_eq!(xs[1].t, -4.);
        }

        #[test]
        fn test_create_intersection() {
            let s = Sphere::new();
            let i = Intersection::new(3.5, &s);
            assert_eq!(i.t, 3.5);
            assert!(ptr::eq(i.object, &s))
        }

        #[test]
        fn test_aggregate_intersections() {
            let s = Sphere::new();
            let i1 = Intersection::new(1., &s);
            let i2 = Intersection::new(2., &s);
            let xs = vec![i1, i2];
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, 1.);
            assert_eq!(xs[1].t, 2.);
        }

        #[test]
        fn test_intersect_sets_object_for_intersection() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let s = Sphere::new();
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert!(ptr::eq(xs[0].object, &s));
            assert!(ptr::eq(xs[1].object, &s));
        }

        #[test]
        fn test_hit_all_positive_intersections() {
            let s = Sphere::new();
            let i1 = Intersection::new(1., &s);
            let i2 = Intersection::new(2., &s);
            let xs = vec![i2, i1];
            let i = hit(xs);
            assert_eq!(i, Some(i1));
        }

        #[test]
        fn test_hit_some_negative_intersections() {
            let s = Sphere::new();
            let i1 = Intersection::new(-1., &s);
            let i2 = Intersection::new(1., &s);
            let xs = vec![i2, i1];
            let i = hit(xs);
            assert_eq!(i, Some(i2));
        }

        #[test]
        fn test_hit_all_negative_intersections() {
            let s = Sphere::new();
            let i1 = Intersection::new(-2., &s);
            let i2 = Intersection::new(-1., &s);
            let xs = vec![i2, i1];
            let i = hit(xs);
            assert_eq!(i, None);
        }

        #[test]
        fn test_hit_is_lowest_nonnegative_intersection() {
            let s = Sphere::new();
            let i1 = Intersection::new(5., &s);
            let i2 = Intersection::new(7., &s);
            let i3 = Intersection::new(-3., &s);
            let i4 = Intersection::new(2., &s);
            let xs = vec![i1, i2, i3, i4];
            let i = hit(xs);
            assert_eq!(i, Some(i4));
        }

        #[test]
        fn test_translate_ray() {
            let r = Ray::new(Tuple::point(1., 2., 3.), Tuple::vector(0., 1., 0.));
            let m = Transform::translation(3., 4., 5.);
            let r2 = r.transform(m);
            assert_eq!(r2.origin, Tuple::point(4., 6., 8.));
            assert_eq!(r2.direction, Tuple::vector(0., 1., 0.));
        }

        #[test]
        fn test_scale_ray() {
            let r = Ray::new(Tuple::point(1., 2., 3.), Tuple::vector(0., 1., 0.));
            let m = Transform::scaling(2., 3., 4.);
            let r2 = r.transform(m);
            assert_eq!(r2.origin, Tuple::point(2., 6., 12.));
            assert_eq!(r2.direction, Tuple::vector(0., 3., 0.));
        }

        #[test]
        fn test_sphere_default_transformation() {
            let s = Sphere::new();
            assert_eq!(s.transform, Transform::identity());
        }

        #[test]
        fn test_change_sphere_transform() {
            let mut s = Sphere::new();
            let t = Transform::translation(2., 3., 4.);
            s.set_transform(t);
            assert_eq!(s.transform, t);
        }

        #[test]
        fn test_intersect_scaled_sphere_with_ray() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let mut s = Sphere::new();
            s.set_transform(Transform::scaling(2., 2., 2.));
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 2);
            assert_eq!(xs[0].t, 3.);
            assert_eq!(xs[1].t, 7.);
        }

        #[test]
        fn test_intersect_translated_sphere_with_ray() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let mut s = Sphere::new();
            s.set_transform(Transform::translation(5., 0., 0.));
            let xs = s.intersect(r);
            assert_eq!(xs.len(), 0);
        }
    }
}

pub mod shading {
    use crate::canvas::Color;
    use crate::tuple::Tuple;

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct PointLight {
        pub position: Tuple,
        pub intensity: Color,
    }

    impl PointLight {
        pub fn new(position: Tuple, intensity: Color) -> Self {
            assert!(position.is_point(), "a PointLight must be given a position, not a vector");
            PointLight { position, intensity }
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Material {
        pub color: Color,
        pub ambient: f64,
        pub diffuse: f64,
        pub specular: f64,
        pub shininess: f64,
    }

    impl Material {
        pub fn new() -> Self {
            Material {
                color: Color::new(1., 1., 1.),
                ambient: 0.1,
                diffuse: 0.9,
                specular: 0.9,
                shininess: 200.,
            }
        }
    }

    pub fn lighting(
        material: Material,
        light: Option<PointLight>,
        point: Tuple,
        eyev: Tuple,
        normalv: Tuple
    ) -> Color {
        // if no point light exists, then return black
        let light = match light {
            Some(l) => l,
            None => return Color::new(0., 0., 0.),
        };
        // calculate ambient light, which is independent of the eye or normal vectors
        let effective_color = material.color * light.intensity;
        let lightv = (light.position - point).normalize();
        let ambient = effective_color * material.ambient;

        let light_dot_normal = lightv.dot(normalv);
        let (diffuse, specular) = if light_dot_normal < 0. {
            // light is behind the surface, so no diffuse or specular light
            (Color::new(0., 0., 0.), Color::new(0., 0., 0.))
        } else {
            // diffuse is solely dependant on the angle between the light and normal vectors
            let diffuse = effective_color * material.diffuse * light_dot_normal;
            let reflectv = (-lightv).reflect(normalv);
            let reflect_dot_eye = reflectv.dot(eyev);
            let specular = if reflect_dot_eye <= 0. {
                // light reflects away from the eye, so no specular light
                Color::new(0., 0., 0.)
            } else {
                // specular light formula
                let factor = reflect_dot_eye.powf(material.shininess);
                light.intensity * material.specular * factor
            };
            (diffuse, specular)
        };
        ambient + diffuse + specular
    }

    #[cfg(test)]
    mod tests {
        use crate::canvas::Color;
        use crate::intersection::Sphere;
        use crate::transformation::Transform;
        use crate::tuple::Tuple;
        use super::*;

        const PI: f64 = std::f64::consts::PI;
        const SQRT_2: f64 = std::f64::consts::SQRT_2;

        fn sqrt3() -> f64 { 3f64.sqrt() }

        #[test]
        fn test_sphere_normal_x() {
            let s = Sphere::new();
            let n = s.normal_at(Tuple::point(1., 0., 0.));
            assert_eq!(n, Tuple::vector(1., 0., 0.));
        }

        #[test]
        fn test_sphere_normal_y() {
            let s = Sphere::new();
            let n = s.normal_at(Tuple::point(0., 1., 0.));
            assert_eq!(n, Tuple::vector(0., 1., 0.));
        }

        #[test]
        fn test_sphere_normal_z() {
            let s = Sphere::new();
            let n = s.normal_at(Tuple::point(0., 0., 1.));
            assert_eq!(n, Tuple::vector(0., 0., 1.));
        }

        #[test]
        fn test_sphere_normal_nonaxial() {
            let s = Sphere::new();
            let n = s.normal_at(Tuple::point(sqrt3() / 3., sqrt3() / 3., sqrt3() / 3.));
            assert_eq!(n, Tuple::vector(sqrt3() / 3., sqrt3() / 3., sqrt3() / 3.));
        }

        #[test]
        fn test_normal_vector_is_normalized() {
            let s = Sphere::new();
            let n = s.normal_at(Tuple::point(sqrt3() / 3., sqrt3() / 3., sqrt3() / 3.));
            assert_eq!(n, n.normalize());
        }

        #[test]
        fn test_normal_on_translated_sphere() {
            let mut s = Sphere::new();
            s.set_transform(Transform::translation(0., 1., 0.));
            let n = s.normal_at(Tuple::point(0., 1.70711, -0.70711));
            assert_eq!(n, Tuple::vector(0., 0.70711, -0.70711));
        }

        #[test]
        fn test_normal_on_transformed_sphere() {
            let mut s = Sphere::new();
            let m = Transform::scaling(1., 0.5, 1.) * Transform::rotation_z(PI / 5.);
            s.set_transform(m);
            let n = s.normal_at(Tuple::point(0., SQRT_2 / 2., -SQRT_2 / 2.));
            assert_eq!(n, Tuple::vector(0., 0.97014, -0.24254));
        }

        #[test]
        fn test_reflect_45_degree_vector() {
            let v = Tuple::vector(1., -1., 0.);
            let n = Tuple::vector(0., 1., 0.);
            let r = v.reflect(n);
            assert_eq!(r, Tuple::vector(1., 1., 0.));
        }

        #[test]
        fn test_reflect_vector_off_slant_surface() {
            let v = Tuple::vector(0., -1., 0.);
            let n = Tuple::vector(SQRT_2 / 2., SQRT_2 / 2., 0.);
            let r = v.reflect(n);
            assert_eq!(r, Tuple::vector(1., 0., 0.));
        }

        #[test]
        fn test_create_point_light() {
            let intensity = Color::new(1., 1., 1.);
            let position = Tuple::point(0., 0., 0.);
            let light = PointLight::new(position, intensity);
            assert_eq!(light.position, position);
            assert_eq!(light.intensity, intensity);
        }

        #[test]
        fn test_default_material() {
            let m = Material::new();
            assert_eq!(m.color, Color::new(1., 1., 1.));
            assert_eq!(m.ambient, 0.1);
            assert_eq!(m.diffuse, 0.9);
            assert_eq!(m.specular, 0.9);
            assert_eq!(m.shininess, 200.);
        }

        #[test]
        fn test_sphere_default_material() {
            let s = Sphere::new();
            let m = s.material;
            assert_eq!(m, Material::new());
        }

        #[test]
        fn test_assign_material_to_sphere() {
            let mut s = Sphere::new();
            let mut m = Material::new();
            m.ambient = 1.;
            s.material = m;
            assert_eq!(s.material, m);
        }

        #[test]
        fn test_eye_between_light_and_surface() {
            let m = Material::new();
            let position = Tuple::point(0., 0., 0.);
            let eyev = Tuple::vector(0., 0., -1.);
            let normalv = Tuple::vector(0., 0., -1.);
            let light = PointLight::new(Tuple::point(0., 0., -10.), Color::new(1., 1., 1.));
            let result = lighting(m, Some(light), position, eyev, normalv);
            assert_eq!(result, Color::new(1.9, 1.9, 1.9));
        }

        #[test]
        fn test_eye_between_light_and_surface_eye_offset_45() {
            let m = Material::new();
            let position = Tuple::point(0., 0., 0.);
            let eyev = Tuple::vector(0., SQRT_2 / 2., SQRT_2 / 2.);
            let normalv = Tuple::vector(0., 0., -1.);
            let light = PointLight::new(Tuple::point(0., 0., -10.), Color::new(1., 1., 1.));
            let result = lighting(m, Some(light), position, eyev, normalv);
            assert_eq!(result, Color::new(1., 1., 1.));
        }

        #[test]
        fn test_eye_opposite_surface_light_offset_45() {
            let m = Material::new();
            let position = Tuple::point(0., 0., 0.);
            let eyev = Tuple::vector(0., 0., -1.);
            let normalv = Tuple::vector(0., 0., -1.);
            let light = PointLight::new(Tuple::point(0., 10., -10.), Color::new(1., 1., 1.));
            let result = lighting(m, Some(light), position, eyev, normalv);
            assert_eq!(result, Color::new(0.7364, 0.7364, 0.7364));
        }

        #[test]
        fn test_eye_in_path_of_reflection_vector() {
            let m = Material::new();
            let position = Tuple::point(0., 0., 0.);
            let eyev = Tuple::vector(0., -SQRT_2 / 2., -SQRT_2 / 2.);
            let normalv = Tuple::vector(0., 0., -1.);
            let light = PointLight::new(Tuple::point(0., 10., -10.), Color::new(1., 1., 1.));
            let result = lighting(m, Some(light), position, eyev, normalv);
            assert_eq!(result, Color::new(1.6364, 1.6364, 1.6364));
        }

        #[test]
        fn test_light_behind_surface() {
            let m = Material::new();
            let position = Tuple::point(0., 0., 0.);
            let eyev = Tuple::vector(0., 0., -1.);
            let normalv = Tuple::vector(0., 0., -1.);
            let light = PointLight::new(Tuple::point(0., 0., 10.), Color::new(1., 1., 1.));
            let result = lighting(m, Some(light), position, eyev, normalv);
            assert_eq!(result, Color::new(0.1, 0.1, 0.1));
        }
    }
}

pub mod scene {
    use crate::matrix::Matrix4x4;
    use crate::shading::{PointLight, lighting};
    use crate::intersection::{Sphere, Intersection, Ray, hit};
    use crate::transformation::Transform;
    use crate::tuple::Tuple;
    use crate::canvas::{Color, Canvas};

    #[derive(Debug)]
    pub struct World {
        pub light: Option<PointLight>,
        pub objects: Vec<Sphere>,
    }

    impl World {
        pub fn empty() -> Self {
            World { light: None, objects: vec![] }
        }

        pub fn default() -> Self {
            let light = PointLight {
                position: Tuple::point(-10., 10., -10.),
                intensity: Color::new(1., 1., 1.),
            };
            let mut s1 = Sphere::new();
            s1.material.color = Color::new(0.8, 1., 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;
            let mut s2 = Sphere::new();
            s2.transform = Transform::scaling(0.5, 0.5, 0.5);
            World {
                light: Some(light),
                objects: vec![s1, s2],
            }
        }

        pub fn intersect(&self, ray: Ray) -> Vec<Intersection> {
            let mut intersections = vec![];
            for obj in self.objects.iter() {
                let xs = obj.intersect(ray);
                intersections.extend(xs);
            }
            // sort_unstable_by used for optimization purposes
            intersections.sort_unstable_by(|a, b| 
                a.t.total_cmp(&b.t));
            intersections
        }

        pub fn shade_hit(&self, comps: Computations) -> Color {
            lighting(
                comps.object.material,
                self.light,
                comps.point,
                comps.eyev,
                comps.normalv,
            )
        }

        pub fn color_at(&self, ray: Ray) -> Color {
            let ints = self.intersect(ray);
            match hit(ints) {
                // the ray hit something; use its color
                Some(hit) => {
                    let comps = Computations::prepare(hit, ray);
                    self.shade_hit(comps)
                },
                // the ray did not hit anything, or objects were behind the ray
                None => Color::new(0., 0., 0.),
            }
        }
    }

    pub struct Computations<'a> {
        pub t: f64,
        pub object: &'a Sphere,
        pub point: Tuple,
        pub eyev: Tuple,
        pub normalv: Tuple,
        pub inside: bool,
    }

    impl<'a> Computations<'a> {
        pub fn prepare(int: Intersection<'a>, ray: Ray) -> Self {
            let t = int.t;
            let object = int.object;
            let point = ray.position(t);
            let eyev = -ray.direction;
            let normalv = object.normal_at(point);
            let inside = normalv.dot(eyev) < 0.;
            let normalv = if inside { -normalv } else { normalv };
            Computations { t, object, point, eyev, normalv, inside }
        }
    }

    impl Transform {
        pub fn view(from: Tuple, to: Tuple, up: Tuple) -> Transform {
            assert!(from.is_point(), "a ViewTransform's `from` field must be a point");
            assert!(to.is_point(), "a ViewTransform's `to` field must be a point");
            assert!(up.is_vector(), "a ViewTransform's `up` field must be a vector");
            let forward = (to - from).normalize();
            let upn = up.normalize();
            let left = forward.cross(upn);
            let true_up = left.cross(forward);
            let orientation = Transform::from(Matrix4x4::new([
                    left.x,     left.y,     left.z, 0.,
                 true_up.x,  true_up.y,  true_up.z, 0.,
                -forward.x, -forward.y, -forward.z, 0.,
                        0.,         0.,         0., 1.,
            ]));
            orientation * Transform::translation(-from.x, -from.y, -from.z)
        }
    }

    pub struct Camera {
        pub hsize: usize,
        pub vsize: usize,
        pub fov: f64,
        pub transform: Transform,
        pixel_size: f64,
        half_width: f64,
        half_height: f64,
    }

    impl Camera {
        pub fn new(hsize: usize, vsize: usize, fov: f64) -> Self {
            let transform = Transform::identity();
            let half_view = (fov / 2.).tan();
            let aspect = hsize as f64 / vsize as f64;
            let (half_width, half_height) = if aspect >= 1. {
                (half_view, half_view / aspect)
            } else {
                (half_view * aspect, half_view)
            };
            let pixel_size = (half_width * 2.) / hsize as f64;
            Camera { hsize, vsize, fov, transform, pixel_size, half_width, half_height }
        }

        pub fn pixel_size(&self) -> f64 {
            self.pixel_size
        }

        pub fn half_width(&self) -> f64 {
            self.half_width
        }

        pub fn half_height(&self) -> f64 {
            self.half_height
        }

        pub fn ray_for_pixel(&self, px: usize, py: usize) -> Ray {
            // offset from the edge of the canvas to the pixel's center
            let xoffset = (px as f64 + 0.5) * self.pixel_size;
            let yoffset = (py as f64 + 0.5) * self.pixel_size;
            // untransformed coordinates of the pixel in world space
            let world_x = self.half_width - xoffset;
            let world_y = self.half_height - yoffset;
            // use the camera matrix to transform the canvas
            // use this to compute the ray's direction
            let inv_trans = self.transform.inverse();
            let pixel = inv_trans * Tuple::point(world_x, world_y, -1.);
            let origin = inv_trans * Tuple::point(0., 0., 0.);
            let direction = (pixel - origin).normalize();
            Ray::new(origin, direction)
        }

        pub fn render(&self, w: World) -> Canvas {
            let mut image = Canvas::new(self.hsize, self.vsize);
            for y in 0..self.vsize {
                for x in 0..self.hsize {
                    let ray = self.ray_for_pixel(x, y);
                    let color = w.color_at(ray);
                    image.write_pixel(x, y, color);
                }
            }
            image
        }
    }

    #[cfg(test)]
    mod tests {
        use std::ptr;
        use crate::equal;
        use crate::transformation::Transform;
        use crate::intersection::Ray;
        use crate::matrix::Matrix4x4;
        use super::*;

        const PI: f64 = std::f64::consts::PI;
        const SQRT_2: f64 = std::f64::consts::SQRT_2;

        #[test]
        fn test_create_world() {
            let w = World::empty();
            assert_eq!(w.objects.len(), 0);
            assert_eq!(w.light, None);
        }

        #[test]
        fn test_default_world() {
            let light = PointLight::new(Tuple::point(-10., 10., -10.), Color::new(1., 1., 1.));
            let mut s1 = Sphere::new();
            s1.material.color = Color::new(0.8, 1., 0.6);
            s1.material.diffuse = 0.7;
            s1.material.specular = 0.2;
            let mut s2 = Sphere::new();
            s2.transform = Transform::scaling(0.5, 0.5, 0.5);
            let w = World::default();
            assert_eq!(w.light, Some(light));
            assert!(w.objects.contains(&s1));
            assert!(w.objects.contains(&s2));
        }

        #[test]
        fn test_intersect_world_with_ray() {
            let w = World::default();
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let xs = w.intersect(r);
            assert_eq!(xs.len(), 4);
            assert_eq!(xs[0].t, 4.);
            assert_eq!(xs[1].t, 4.5);
            assert_eq!(xs[2].t, 5.5);
            assert_eq!(xs[3].t, 6.);
        }

        #[test]
        fn test_precompute_intersection_state() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let shape = Sphere::new();
            let i = Intersection { t: 4., object: &shape };
            let comps = Computations::prepare(i, r);
            assert_eq!(comps.t, i.t);
            assert!(ptr::eq(comps.object, i.object));
            assert_eq!(comps.point, Tuple::point(0., 0., -1.));
            assert_eq!(comps.eyev, Tuple::vector(0., 0., -1.));
            assert_eq!(comps.normalv, Tuple::vector(0., 0., -1.));
        }

        #[test]
        fn test_ray_hit_outside_object() {
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let shape = Sphere::new();
            let i = Intersection { t: 4., object: &shape };
            let comps = Computations::prepare(i, r);
            assert_eq!(comps.inside, false);
        }

        #[test]
        fn test_ray_hit_inside_object() {
            let r = Ray::new(Tuple::point(0., 0., 0.), Tuple::vector(0., 0., 1.));
            let shape = Sphere::new();
            let i = Intersection { t: 1., object: &shape };
            let comps = Computations::prepare(i, r);
            assert_eq!(comps.point, Tuple::point(0., 0., 1.));
            assert_eq!(comps.eyev, Tuple::vector(0., 0., -1.));
            assert_eq!(comps.inside, true);
            assert_eq!(comps.normalv, Tuple::vector(0., 0., -1.));
        }

        #[test]
        fn test_shade_intersection() {
            let w = World::default();
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let shape = &w.objects[0];
            let i = Intersection { t: 4., object: shape };
            let comps = Computations::prepare(i, r);
            let c = w.shade_hit(comps);
            assert_eq!(c, Color::new(0.38066, 0.47583, 0.2855));
        }

        #[test]
        fn test_shade_intersection_from_inside() {
            let mut w = World::default();
            w.light = Some(PointLight::new(Tuple::point(0., 0.25, 0.), Color::new(1., 1., 1.)));
            let r = Ray::new(Tuple::point(0., 0., 0.), Tuple::vector(0., 0., 1.));
            let shape = &w.objects[1];
            let i = Intersection { t: 0.5, object: shape };
            let comps = Computations::prepare(i, r);
            let c = w.shade_hit(comps);
            assert_eq!(c, Color::new(0.90498, 0.90498, 0.90498));
        }

        #[test]
        fn test_color_when_ray_misses() {
            let w = World::default();
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 1., 0.));
            let c = w.color_at(r);
            assert_eq!(c, Color::new(0., 0., 0.));
        }

        #[test]
        fn test_color_when_ray_hits() {
            let w = World::default();
            let r = Ray::new(Tuple::point(0., 0., -5.), Tuple::vector(0., 0., 1.));
            let c = w.color_at(r);
            assert_eq!(c, Color::new(0.38066, 0.47583, 0.2855));
        }

        #[test]
        fn test_color_with_intersection_behind_ray() {
            let mut w = World::default();
            let inner_material_color = {    // bodgy mutable-pointer-management workaround
                let outer = &mut w.objects[0];
                outer.material.ambient = 1.;
                let inner = &mut w.objects[1];
                inner.material.ambient = 1.;

                inner.material.color
            };
            let r = Ray::new(Tuple::point(0., 0., 0.75), Tuple::vector(0., 0., -1.));
            let c = w.color_at(r);
            assert_eq!(c, inner_material_color);
        }

        #[test]
        fn test_transform_matrix_default_orientation() {
            let from = Tuple::point(0., 0., 0.);
            let to = Tuple::point(0., 0., -1.);
            let up = Tuple::vector(0., 1., 0.);
            let t = Transform::view(from, to, up);
            assert_eq!(t, Transform::identity());
        }

        #[test]
        fn test_view_transform_looking_pos_z() {
            let from = Tuple::point(0., 0., 0.);
            let to = Tuple::point(0., 0., 1.);
            let up = Tuple::vector(0., 1., 0.);
            let t = Transform::view(from, to, up);
            assert_eq!(t, Transform::scaling(-1., 1., -1.));
        }

        #[test]
        fn test_view_transform_moves_world() {
            let from = Tuple::point(0., 0., 8.);
            let to = Tuple::point(0., 0., 0.);
            let up = Tuple::vector(0., 1., 0.);
            let t = Transform::view(from, to, up);
            assert_eq!(t, Transform::translation(0., 0., -8.));
        }

        #[test]
        fn test_arbitrary_view_transform() {
            let from = Tuple::point(1., 3., 2.);
            let to = Tuple::point(4., -2., 8.);
            let up = Tuple::vector(1., 1., 0.);
            let t = Transform::view(from, to, up);
            assert_eq!(t, Transform::from(Matrix4x4::new([
                -0.50709, 0.50709,  0.67612, -2.36643,
                 0.76772, 0.60609,  0.12122, -2.82843,
                -0.35857, 0.59761, -0.71714,  0.00000,
                 0.00000, 0.00000,  0.00000,  1.00000,
            ])));
        }

        #[test]
        fn test_create_camera() {
            let hsize = 160;
            let vsize = 120;
            let fov = PI / 2.;
            let c = Camera::new(hsize, vsize, fov);
            assert_eq!(c.hsize, 160);
            assert_eq!(c.vsize, 120);
            assert_eq!(c.fov, PI / 2.);
            assert_eq!(c.transform, Transform::identity());
        }

        #[test]
        fn test_pixel_size_horizontal_canvas() {
            let c = Camera::new(200, 125, PI / 2.);
            assert!(equal(c.pixel_size(), 0.01));
        }

        #[test]
        fn test_pixel_size_vertical_canvas() {
            let c = Camera::new(125, 200, PI / 2.);
            assert!(equal(c.pixel_size(), 0.01));
        }

        #[test]
        fn test_ray_through_canvas_center() {
            let c = Camera::new(201, 101, PI / 2.);
            let r = c.ray_for_pixel(100, 50);
            assert_eq!(r.origin, Tuple::point(0., 0., 0.));
            assert_eq!(r.direction, Tuple::vector(0., 0., -1.));
        }

        #[test]
        fn test_ray_through_canvas_corner() {
            let c = Camera::new(201, 101, PI / 2.);
            let r = c.ray_for_pixel(0, 0);
            assert_eq!(r.origin, Tuple::point(0., 0., 0.));
            assert_eq!(r.direction, Tuple::vector(0.66519, 0.33259, -0.66851));
        }

        #[test]
        fn test_ray_through_transformed_canvas() {
            let mut c = Camera::new(201, 101, PI / 2.);
            c.transform = Transform::rotation_y(PI / 4.) * Transform::translation(0., -2., 5.);
            let r = c.ray_for_pixel(100, 50);
            assert_eq!(r.origin, Tuple::point(0., 2., -5.));
            assert_eq!(r.direction, Tuple::vector(SQRT_2 / 2., 0., -SQRT_2 / 2.));
        }

        #[test]
        fn test_render_world_with_camera() {
            let w = World::default();
            let mut c = Camera::new(11, 11, PI / 2.);
            let from = Tuple::point(0., 0., -5.);
            let to = Tuple::point(0., 0., 0.);
            let up = Tuple::vector(0., 1., 0.);
            c.transform = Transform::view(from, to, up);
            let image = c.render(w);
            assert_eq!(image.pixel_at(5, 5), Color::new(0.38066, 0.47583, 0.2855));
        }
    }
}
