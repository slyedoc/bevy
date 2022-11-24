use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

pub fn bevy_main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    assert!(
        input.sig.ident == "main",
        "`bevy_main` can only be used on a function called 'main'.",
    );

    TokenStream::from(quote! {
    
        // TODO: Remove this
        // use ndk-glue macro to create an activity: https://github.com/rust-windowing/android-ndk-rs/tree/master/ndk-macro
        // #[cfg(target_os = "android")]
        // #[cfg_attr(target_os = "android", bevy::ndk_glue::main(backtrace = "on", ndk_glue = "bevy::ndk_glue"))]
        // fn android_main() {
        //     main()
        // }

        // Notes:
        /// Global static maybe? 
        /// https://www.sitepoint.com/rust-global-variables/

        // #[cfg(target_os = "android")]        
        // #[no_mangle]
        // fn android_main(app: bevy::winit::android::activity::AndroidApp) {
        //     use bevy::winit::android::EventLoopBuilderExtAndroid;

        //     android_logger::init_once(android_logger::Config::default().with_min_level(log::Level::Trace));

        //     let event_loop = EventLoopBuilder::with_user_event()
        //         .with_android_app(app)
        //         .build();
        //     //main(event_loop);
        //     // TODO: figure out how to pass the event loop to main
        //     main()
        // }


        #[no_mangle]
        #[cfg(target_os = "ios")]
        extern "C" fn main_rs() {
            main();
        }

        #[allow(unused)]
        #input
    })
}
