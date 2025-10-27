static void qmp_cleanup(void *datap)

{

    QmpSerializeData *d = datap;

    visit_free(qmp_output_get_visitor(d->qov));

    visit_free(d->qiv);



    g_free(d);

}
