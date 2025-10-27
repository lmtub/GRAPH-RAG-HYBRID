static void aml_free(gpointer data, gpointer user_data)

{

    Aml *var = data;

    build_free_array(var->buf);


}