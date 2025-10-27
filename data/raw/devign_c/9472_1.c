static void powerpc_set_compat(Object *obj, Visitor *v, const char *name,

                               void *opaque, Error **errp)

{

    Error *error = NULL;

    char *value = NULL;

    Property *prop = opaque;

    uint32_t *max_compat = qdev_get_prop_ptr(DEVICE(obj), prop);



    visit_type_str(v, name, &value, &error);

    if (error) {

        error_propagate(errp, error);

        return;

    }



    if (strcmp(value, "power6") == 0) {

        *max_compat = CPU_POWERPC_LOGICAL_2_05;

    } else if (strcmp(value, "power7") == 0) {

        *max_compat = CPU_POWERPC_LOGICAL_2_06;

    } else if (strcmp(value, "power8") == 0) {

        *max_compat = CPU_POWERPC_LOGICAL_2_07;

    } else {

        error_setg(errp, "Invalid compatibility mode \"%s\"", value);

    }



    g_free(value);

}
