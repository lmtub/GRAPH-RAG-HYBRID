QObject *qmp_output_get_qobject(QmpOutputVisitor *qov)

{

    /* FIXME: we should require that a visit occurred, and that it is

     * complete (no starts without a matching end) */

    QObject *obj = qov->root;

    if (obj) {

        qobject_incref(obj);

    } else {

        obj = qnull();

    }

    return obj;

}
