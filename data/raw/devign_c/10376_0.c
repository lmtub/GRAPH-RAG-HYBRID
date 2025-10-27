static void qmp_output_type_null(Visitor *v, const char *name, Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_add_obj(qov, name, qnull());

}
