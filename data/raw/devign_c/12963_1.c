int spapr_rtas_register(const char *name, spapr_rtas_fn fn)

{

    int i;



    for (i = 0; i < (rtas_next - rtas_table); i++) {

        if (strcmp(name, rtas_table[i].name) == 0) {

            fprintf(stderr, "RTAS call \"%s\" registered twice\n", name);

            exit(1);

        }

    }



    assert(rtas_next < (rtas_table + TOKEN_MAX));



    rtas_next->name = name;

    rtas_next->fn = fn;



    return (rtas_next++ - rtas_table) + TOKEN_BASE;

}
