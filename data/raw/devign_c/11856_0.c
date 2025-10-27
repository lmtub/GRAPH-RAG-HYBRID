static void qmp_output_start_list(Visitor *v, const char *name,

                                  GenericList **listp, size_t size,

                                  Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    QList *list = qlist_new();



    qmp_output_add(qov, name, list);

    qmp_output_push(qov, list, listp);

}
