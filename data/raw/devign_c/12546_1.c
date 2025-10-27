static void object_get_link_property(Object *obj, Visitor *v, void *opaque,

                                     const char *name, Error **errp)

{

    Object **child = opaque;

    gchar *path;



    if (*child) {

        path = object_get_canonical_path(*child);

        visit_type_str(v, &path, name, errp);

        g_free(path);

    } else {

        path = (gchar *)"";

        visit_type_str(v, &path, name, errp);

    }

}
