static void string_cleanup(void *datap)

{

    StringSerializeData *d = datap;



    visit_free(string_output_get_visitor(d->sov));

    visit_free(d->siv);

    g_free(d->string);

    g_free(d);

}
