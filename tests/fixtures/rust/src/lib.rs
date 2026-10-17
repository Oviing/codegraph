/// A greeter.
pub trait Greet {
    fn greet(&self) -> String;
}

pub struct En;

impl Greet for En {
    fn greet(&self) -> String {
        helper()
    }
}

pub fn helper() -> String {
    String::from("hi")
}
