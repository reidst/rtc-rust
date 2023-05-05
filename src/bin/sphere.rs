use std::fs;
use std::io::Write;
use rtc_rust::shading::{PointLight, lighting};
use rtc_rust::tuple::Tuple;
use rtc_rust::canvas::{Canvas, Color};
use rtc_rust::intersection::{Sphere, Ray, Intersection};

fn main() -> std::io::Result<()> {
    let ray_origin = Tuple::point(0., 0., -5.);
    let wall_z = 10.;
    let wall_size = 7.;
    let canvas_pixels = 1024;
    let pixel_size = wall_size / canvas_pixels as f64;
    let half = wall_size / 2.;

    let mut canvas = Canvas::new(canvas_pixels, canvas_pixels);
    let mut sphere = Sphere::new();
    sphere.material.color = Color::new(1., 0.2, 1.);

    let light_position = Tuple::point(-10., 10., -10.);
    let light_color = Color::new(1., 1., 1.);
    let light = PointLight::new(light_position, light_color);

    for y in 0..canvas_pixels {
        let world_y = half - pixel_size * y as f64;
        for x in 0..canvas_pixels {
            let world_x = -half + pixel_size * x as f64;
            let position = Tuple::point(world_x, world_y, wall_z);
            let r = Ray::new(ray_origin, (position - ray_origin).normalize());
            let xs = sphere.intersect(r);
            if let Some(hit) = Intersection::hit(xs) {
                let point = r.position(hit.t);
                let normal = hit.object.normal_at(point);
                let eye = -r.direction;
                let color = lighting(hit.object.material, light, point, eye, normal);
                canvas.write_pixel(x, y, color);
            }
        }
    }

    let ppm = canvas.to_ppm();
    let mut file = fs::File::create("out/sphere.ppm")?;
    file.write(ppm.as_bytes())?;
    Ok(())
}