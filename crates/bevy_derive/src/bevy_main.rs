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
        
        // #[cfg(target_os = "android")]
        // #[no_mangle]
        // fn android_main(android_app: bevy::android::AndroidApp) {

        //     #[allow(unused)]
        //     let _result = bevy::android::GLOBAL_ANDROID_APP.set(android_app).unwrap();

        //     main(android_app)
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
