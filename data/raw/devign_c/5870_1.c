int64_t object_property_get_int(Object *obj, const char *name,

                                Error **errp)

{

    QObject *ret = object_property_get_qobject(obj, name, errp);

    QInt *qint;

    int64_t retval;



    if (!ret) {

        return -1;

    }

    qint = qobject_to_qint(ret);

    if (!qint) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name, "int");

        retval = -1;

    } else {

        retval = qint_get_int(qint);

    }



    QDECREF(qint);

    return retval;

}
