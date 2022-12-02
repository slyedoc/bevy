# Notes

Work in progress, testing for winit 0.8 that will replace ndk-glue with android-activity.

All sound is disabled, will come back to that after resume suspend if working.

# Test commands
(Assuming from bevy root directory)

Run on device or emulator
```
cargo apk run -p android_native_activity
```

Run on desktop
```
cargo run -p android_native_activity --features desktop
```

# Build Issues

Make sure these are set

```
export ANDROID_HOME="path/to/sdk"
```

For oboe build, also make sure NDK llvm tools are in your PATH

```
export PATH="$ANDROID_HOME/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH";
```

https://github.com/katyo/oboe-rs#build-issues