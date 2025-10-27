Object *object_new_with_propv(const char *typename,

                              Object *parent,

                              const char *id,

                              Error **errp,

                              va_list vargs)

{

    Object *obj;

    ObjectClass *klass;

    Error *local_err = NULL;



    klass = object_class_by_name(typename);

    if (!klass) {

        error_setg(errp, "invalid object type: %s", typename);

        return NULL;

    }



    if (object_class_is_abstract(klass)) {

        error_setg(errp, "object type '%s' is abstract", typename);

        return NULL;

    }

    obj = object_new(typename);



    if (object_set_propv(obj, &local_err, vargs) < 0) {

        goto error;

    }



    object_property_add_child(parent, id, obj, &local_err);

    if (local_err) {

        goto error;

    }



    if (object_dynamic_cast(obj, TYPE_USER_CREATABLE)) {

        user_creatable_complete(obj, &local_err);

        if (local_err) {

            object_unparent(obj);

            goto error;

        }

    }



    object_unref(OBJECT(obj));

    return obj;



 error:

    if (local_err) {

        error_propagate(errp, local_err);

    }

    object_unref(obj);

    return NULL;

}
