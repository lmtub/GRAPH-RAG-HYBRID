QString *qobject_to_qstring(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QSTRING)

        return NULL;



    return container_of(obj, QString, base);

}
