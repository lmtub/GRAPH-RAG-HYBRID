static CheckasmFunc *get_func(const char *name, int length)

{

    CheckasmFunc *f, **f_ptr = &state.funcs;



    /* Search the tree for a matching node */

    while ((f = *f_ptr)) {

        int cmp = cmp_func_names(name, f->name);

        if (!cmp)

            return f;



        f_ptr = &f->child[(cmp > 0)];

    }



    /* Allocate and insert a new node into the tree */

    f = *f_ptr = checkasm_malloc(sizeof(CheckasmFunc) + length);

    memcpy(f->name, name, length+1);



    return f;

}
