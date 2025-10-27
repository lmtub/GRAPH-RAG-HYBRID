static void qmp_input_end_list(Visitor *v)

{

    QmpInputVisitor *qiv = to_qiv(v);



    qmp_input_pop(qiv, &error_abort);

}
