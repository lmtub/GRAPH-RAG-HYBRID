QError *qobject_to_qerror(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QERROR) {

        return NULL;

    }



    return container_of(obj, QError, base);

}
