void qapi_copy_SocketAddress(SocketAddress **p_dest,

                             SocketAddress *src)

{

    QmpOutputVisitor *qov;

    Visitor *ov, *iv;

    QObject *obj;



    *p_dest = NULL;



    qov = qmp_output_visitor_new();

    ov = qmp_output_get_visitor(qov);

    visit_type_SocketAddress(ov, NULL, &src, &error_abort);

    obj = qmp_output_get_qobject(qov);

    visit_free(ov);

    if (!obj) {

        return;

    }



    iv = qmp_input_visitor_new(obj, true);

    visit_type_SocketAddress(iv, NULL, p_dest, &error_abort);

    visit_free(iv);

    qobject_decref(obj);

}
