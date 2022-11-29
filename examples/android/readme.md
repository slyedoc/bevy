# Cargo APK Build
```
export ANDROID_NDK_HOME="path/to/ndk"
export ANDROID_SDK_HOME="path/to/sdk"

rustup target add aarch64-linux-android
cargo install cargo-apk

cargo apk run
```

# Build Issues


Also make sure NDK llvm tools are in your PATH, oboe requires it

```
$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin
```


https://github.com/katyo/oboe-rs#build-issues