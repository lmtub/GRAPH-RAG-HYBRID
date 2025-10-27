static void qmp_output_end_struct(Visitor *v, Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_pop(qov);

}
