bool object_property_get_bool(Object *obj, const char *name,

                              Error **errp)

{

    QObject *ret = object_property_get_qobject(obj, name, errp);

    QBool *qbool;

    bool retval;



    if (!ret) {

        return false;

    }

    qbool = qobject_to_qbool(ret);

    if (!qbool) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name, "boolean");

        retval = false;

    } else {

        retval = qbool_get_bool(qbool);

    }



    QDECREF(qbool);

    return retval;

}
