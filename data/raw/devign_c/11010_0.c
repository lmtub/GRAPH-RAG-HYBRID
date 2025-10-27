static void qmp_output_type_enum(Visitor *v, int *obj, const char *strings[],

                                 const char *kind, const char *name,

                                 Error **errp)

{

    int i = 0;

    int value = *obj;

    char *enum_str;



    assert(strings);

    while (strings[i++] != NULL);

    if (value >= i - 1) {

        error_set(errp, QERR_INVALID_PARAMETER, name ? name : "null");

        return;

    }



    enum_str = (char *)strings[value];

    qmp_output_type_str(v, &enum_str, name, errp);

}
