static void qmp_output_type_uint64(Visitor *v, const char *name, uint64_t *obj,

                                   Error **errp)

{

    /* FIXME values larger than INT64_MAX become negative */

    QmpOutputVisitor *qov = to_qov(v);

    qmp_output_add(qov, name, qint_from_int(*obj));

}
