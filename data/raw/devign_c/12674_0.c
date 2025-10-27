static void qbool_destroy_obj(QObject *obj)

{

    assert(obj != NULL);

    g_free(qobject_to_qbool(obj));

}
