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
