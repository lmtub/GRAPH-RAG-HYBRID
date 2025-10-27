static void qmp_output_type_number(Visitor *v, const char *name, double *obj,

                                   Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_add(qov, name, qfloat_from_double(*obj));

}
