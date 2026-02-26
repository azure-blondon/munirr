# munirr

munirr is a rust compiler for the muni programming language that targets WebAssembly.

## goal
My current goal is to bootstrap the language, so, minimal dependencies are used in this implementation.


## roadmap to bootstrapping
Here is a list of every feature that is needed in order to rewrite the compiler in the language itself
needed features:
- [x] functions
- [x] function exports
- [x] variables (local & global)
- [x] an integer type
- [x] loops
- [x] imports (for WASI i/o imports)
- [x] a buffer type (C-style array)
- [ ] structs

minimal stdlib:
- [ ] vector type
- [x] println function

nice to have:
- [x] char literals
- [x] string literals


## how to use

it is possible to compile a muni program using `cargo`

```bash
cargo run <file1.mun> [file2.mun ...] -o <out.wasm>
```

example, using the wasi lib:
```bash
cargo run lib/wasi.mun main.mun -o out.wasm
```
