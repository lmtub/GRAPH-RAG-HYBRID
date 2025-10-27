char *object_property_print(Object *obj, const char *name, bool human,

                            Error **errp)

{

    StringOutputVisitor *mo;

    char *string;



    mo = string_output_visitor_new(human);

    object_property_get(obj, string_output_get_visitor(mo), name, errp);

    string = string_output_get_string(mo);

    string_output_visitor_cleanup(mo);

    return string;

}
