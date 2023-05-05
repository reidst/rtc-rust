use std::fs;
use std::io::Write;

use rtc_rust::canvas::Color;
use rtc_rust::intersection::Sphere;
use rtc_rust::scene::{World, Camera};
use rtc_rust::shading::{Material, PointLight};
use rtc_rust::transformation::Transform;
use rtc_rust::tuple::Tuple;

const PI: f64 = std::f64::consts::PI;

fn main() -> std::io::Result<()> {
    // the floor and walls are extremely squished matte spheres
    let mut floor = Sphere::new();
    floor.transform = Transform::scaling(10., 0.01, 10.);
    floor.material = Material::new();
    floor.material.color = Color::new(1., 0.9, 0.9);
    floor.material.specular = 0.;

    let mut left_wall = Sphere::new();
    left_wall.transform =
        Transform::translation(0., 0., 5.) *
        Transform::rotation_y(-PI / 4.) *
        Transform::rotation_x(PI / 2.) *
        Transform::scaling(10., 0.01, 10.);
    left_wall.material = floor.material;

    let mut right_wall = Sphere::new();
    right_wall.transform =
        Transform::translation(0., 0., 5.) *
        Transform::rotation_y(PI / 4.) *
        Transform::rotation_x(PI / 2.) *
        Transform::scaling(10., 0.01, 10.);
    // A large green sphere in the center of the view
    let mut middle = Sphere::new();
    middle.transform = Transform::translation(-0.5, 1., 0.5);
    middle.material = Material::new();
    middle.material.color = Color::new(0.1, 1., 0.5);
    middle.material.diffuse = 0.7;
    middle.material.specular = 0.3;
    // a smaller yellow-green sphere to the right
    let mut right = Sphere::new();
    right.transform =
        Transform::translation(1.5, 0.5, -0.5) *
        Transform::scaling(0.5, 0.5, 0.5);
    right.material = Material::new();
    right.material.color = Color::new(0.5, 1., 0.1);
    right.material.diffuse = 0.7;
    right.material.specular = 0.3;
    // a smallest yellow sphere to the left
    let mut left = Sphere::new();
    left.transform =
        Transform::translation(-1.5, 0.33, -0.75) *
        Transform::scaling(0.33, 0.33, 0.33);
    left.material = Material::new();
    left.material.color = Color::new(1., 0.8, 0.1);
    left.material.diffuse = 0.7;
    left.material.specular = 0.3;
    // create world; add a point light and the above objects
    let mut world = World::empty();
    world.light = Some(
        PointLight::new(Tuple::point(-10., 10., -10.), Color::new(1., 1., 1.))
    );
    world.objects.extend(vec![
        floor,
        left_wall,
        right_wall,
        middle,
        right,
        left,
    ]);
    let mut camera = Camera::new(1920, 1080, PI / 3.);
    camera.transform = Transform::view(
        Tuple::point(0., 1.5, -5.),
        Tuple::point(0., 1., 0.),
        Tuple::vector(0., 1., 0.),
    );
    // pulling everything together
    let canvas = camera.render(world);
    let ppm = canvas.to_ppm();
    let mut file = fs::File::create("out/helloworld.ppm")?;
    file.write(ppm.as_bytes())?;
    Ok(())
}