static void dump_op_count(void)

{

    int i;

    FILE *f;

    f = fopen("/tmp/op.log", "w");

    for(i = INDEX_op_end; i < NB_OPS; i++) {

        fprintf(f, "%s %" PRId64 "\n", tcg_op_defs[i].name, tcg_table_op_count[i]);

    }

    fclose(f);

}
