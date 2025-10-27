Object *user_creatable_add_type(const char *type, const char *id,

                                const QDict *qdict,

                                Visitor *v, Error **errp)

{

    Object *obj;

    ObjectClass *klass;

    const QDictEntry *e;

    Error *local_err = NULL;



    klass = object_class_by_name(type);

    if (!klass) {

        error_setg(errp, "invalid object type: %s", type);

        return NULL;

    }



    if (!object_class_dynamic_cast(klass, TYPE_USER_CREATABLE)) {

        error_setg(errp, "object type '%s' isn't supported by object-add",

                   type);

        return NULL;

    }



    if (object_class_is_abstract(klass)) {

        error_setg(errp, "object type '%s' is abstract", type);

        return NULL;

    }



    obj = object_new(type);

    if (qdict) {

        for (e = qdict_first(qdict); e; e = qdict_next(qdict, e)) {

            object_property_set(obj, v, e->key, &local_err);

            if (local_err) {

                goto out;

            }

        }

    }



    object_property_add_child(object_get_objects_root(),

                              id, obj, &local_err);

    if (local_err) {

        goto out;

    }



    user_creatable_complete(obj, &local_err);

    if (local_err) {

        object_property_del(object_get_objects_root(),

                            id, &error_abort);

        goto out;

    }

out:

    if (local_err) {

        error_propagate(errp, local_err);

        object_unref(obj);

        return NULL;

    }

    return obj;

}
