fn main() {
    // Link to the system-installed primesieve library
    println!("cargo:rustc-link-lib=primesieve");
}
