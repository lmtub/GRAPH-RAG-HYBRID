static void qmp_input_end_struct(Visitor *v, Error **errp)

{

    QmpInputVisitor *qiv = to_qiv(v);



    qmp_input_pop(qiv, errp);

}
