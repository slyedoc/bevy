# How to run

## Desktop
cargo run -p android_native_activity --features desktop

## Android


rustup target add aarch64-linux-android x86_64-linux-android
rustup target add 
cargo install cargo-apk

cargo apk run -p android_native_activity
```