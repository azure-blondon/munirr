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
- [ ] loops
- [ ] imports (for WASI i/o imports)
- [ ] a buffer type (C-style array)
- [ ] structs

minimal stdlib:
- [ ] vector type
- [ ] println function

nice to have:
- [ ] char literals





## to-do
- [x] start the readme
- [x] compile a first program (yay!)
- [ ] rewrite lexer and parser
- [ ] write tests
- [ ] add loops