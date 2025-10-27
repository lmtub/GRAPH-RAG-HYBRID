void object_property_get_uint16List(Object *obj, const char *name,

                                    uint16List **list, Error **errp)

{

    Error *err = NULL;

    StringOutputVisitor *ov;

    Visitor *v;

    char *str;



    ov = string_output_visitor_new(false);

    object_property_get(obj, string_output_get_visitor(ov),

                        name, &err);

    if (err) {

        error_propagate(errp, err);

        goto out;

    }

    str = string_output_get_string(ov);

    v = string_input_visitor_new(str);

    visit_type_uint16List(v, NULL, list, errp);



    g_free(str);

    visit_free(v);

out:

    string_output_visitor_cleanup(ov);

}
