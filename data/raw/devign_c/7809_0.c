void qmp_input_visitor_cleanup(QmpInputVisitor *v)

{

    qobject_decref(v->stack[0].obj);

    g_free(v);

}
