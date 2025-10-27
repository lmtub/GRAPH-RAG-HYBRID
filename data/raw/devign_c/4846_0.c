void object_property_add_child(Object *obj, const char *name,

                               Object *child, Error **errp)

{

    Error *local_err = NULL;

    gchar *type;



    type = g_strdup_printf("child<%s>", object_get_typename(OBJECT(child)));



    object_property_add(obj, name, type, object_get_child_property, NULL,

                        object_finalize_child_property, child, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }

    object_ref(child);

    g_assert(child->parent == NULL);

    child->parent = obj;



out:

    g_free(type);

}
