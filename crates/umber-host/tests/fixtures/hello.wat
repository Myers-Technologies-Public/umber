;; Host ABI v1 wasm fixture: registers one command and emits fixed text on
;; invoke. Compiled directly from `.wat` by wasmtime's `wat` feature.
(module
  (import "umber" "register" (func $register (param i32 i32 i32 i32)))
  (import "umber" "emit" (func $emit (param i32 i32)))
  (memory (export "memory") 1)
  ;; "hello.greet" @0 (len 11), "Hello: Greet" @16 (len 12),
  ;; "hello from wasm" @32 (len 15).
  (data (i32.const 0) "hello.greet")
  (data (i32.const 16) "Hello: Greet")
  (data (i32.const 32) "hello from wasm")
  (func (export "umber_abi_version") (result i32)
    i32.const 1)
  (func (export "umber_register")
    i32.const 0
    i32.const 11
    i32.const 16
    i32.const 12
    call $register)
  (func (export "umber_invoke") (param i32)
    i32.const 32
    i32.const 15
    call $emit)
)
