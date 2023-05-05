use std::{fs, io::Write};

use rtc_rust::{tuple::Tuple, canvas::{Canvas, Color}, intersection::{Sphere, Ray, hit}};

fn main() -> std::io::Result<()> {
    let ray_origin = Tuple::point(0., 0., -5.);
    let wall_z = 10.;
    let wall_size = 7.;
    let canvas_pixels = 1024;
    let pixel_size = wall_size / canvas_pixels as f64;
    let half = wall_size / 2.;

    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);
    let shape = Sphere::new();

    for y in 0..canvas_pixels {
        let world_y = half - pixel_size * y as f64;
        for x in 0..canvas_pixels {
            let world_x = -half + pixel_size * x as f64;
            let position = Tuple::point(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, (position - ray_origin).normalize());
            let xs = shape.intersect(r);
            if let Some(_) = hit(xs) {
                let color = Color::new(x as f64 / canvas_pixels as f64, 0., y as f64 / canvas_pixels as f64);
                canvas.write_pixel(x, y, color);
            }
        }
    }

    let ppm = canvas.to_ppm();
    let mut file = fs::File::create("out/circle.ppm")?;
    file.write(ppm.as_bytes())?;
    Ok(())
}
