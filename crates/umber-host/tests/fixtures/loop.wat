;; Host ABI v1 wasm fixture whose command spins forever. The host's epoch
;; deadline must trap it (HostError::Timeout) instead of freezing.
(module
  (import "umber" "register" (func $register (param i32 i32 i32 i32)))
  (import "umber" "emit" (func $emit (param i32 i32)))
  (memory (export "memory") 1)
  ;; "loop.spin" @0 (len 9), "Loop: Spin" @16 (len 10).
  (data (i32.const 0) "loop.spin")
  (data (i32.const 16) "Loop: Spin")
  (func (export "umber_register")
    i32.const 0
    i32.const 9
    i32.const 16
    i32.const 10
    call $register)
  (func (export "umber_invoke") (param i32)
    (loop $l
      br $l))
)
