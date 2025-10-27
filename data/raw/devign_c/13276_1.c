static QObject *qdict_get_obj(const QDict *qdict, const char *key, QType type)

{

    QObject *obj;



    obj = qdict_get(qdict, key);

    assert(obj != NULL);

    assert(qobject_type(obj) == type);



    return obj;

}
