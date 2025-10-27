static void qmp_output_type_str(Visitor *v, const char *name, char **obj,

                                Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    if (*obj) {

        qmp_output_add(qov, name, qstring_from_str(*obj));

    } else {

        qmp_output_add(qov, name, qstring_from_str(""));

    }

}
