void tcg_register_helper(void *func, const char *name)

{

    TCGContext *s = &tcg_ctx;

    GHashTable *table = s->helpers;



    if (table == NULL) {

        /* Use g_direct_hash/equal for direct pointer comparisons on func.  */

        table = g_hash_table_new(NULL, NULL);

        s->helpers = table;

    }



    g_hash_table_insert(table, (gpointer)func, (gpointer)name);

}
