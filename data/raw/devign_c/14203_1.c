QList *qobject_to_qlist(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QLIST) {

        return NULL;

    }



    return container_of(obj, QList, base);

}
