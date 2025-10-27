static void qmp_output_start_struct(Visitor *v, const char *name, void **obj,

                                    size_t unused, Error **errp)

{

    QmpOutputVisitor *qov = to_qov(v);

    QDict *dict = qdict_new();



    qmp_output_add(qov, name, dict);

    qmp_output_push(qov, dict, obj);

}
