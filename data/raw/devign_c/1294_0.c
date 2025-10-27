processed(OptsVisitor *ov, const char *name)

{

    if (ov->repeated_opts == NULL) {

        g_hash_table_remove(ov->unprocessed_opts, name);

    }

}
