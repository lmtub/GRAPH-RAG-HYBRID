static void qmp_serialize(void *native_in, void **datap,

                          VisitorFunc visit, Error **errp)

{

    QmpSerializeData *d = g_malloc0(sizeof(*d));



    d->qov = qmp_output_visitor_new(&d->obj);

    visit(d->qov, &native_in, errp);

    *datap = d;

}
