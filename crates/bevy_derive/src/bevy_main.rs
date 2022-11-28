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
    
        #[cfg(target_os = "android")] {
            use bevy::winit::android::activity::AndroidApp;

            // TODO: I don't really like the idea of adding a global, but it keeps 
            // the user code clean
            thread_local!(static GLOBAL_ANDROID_APP: Option<AndroidApp> = None);
            
            #[no_mangle]
            fn android_main(android_app: AndroidApp) {
                
                GLOBAL_ANDROID_APP.with(|global_android_app| {
                    *global_android_app = Some(android_app);
                });
                main()
            }

        }

        #[no_mangle]
        #[cfg(target_os = "ios")]
        extern "C" fn main_rs() {
            main();
        }

        #[allow(unused)]
        #input
    })
}
