QFloat *qobject_to_qfloat(const QObject *obj)

{

    if (qobject_type(obj) != QTYPE_QFLOAT)

        return NULL;



    return container_of(obj, QFloat, base);

}
