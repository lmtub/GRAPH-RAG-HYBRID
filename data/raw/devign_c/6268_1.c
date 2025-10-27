void visit_type_int32(Visitor *v, int32_t *obj, const char *name, Error **errp)

{

    int64_t value;

    if (!error_is_set(errp)) {

        if (v->type_int32) {

            v->type_int32(v, obj, name, errp);

        } else {

            value = *obj;

            v->type_int(v, &value, name, errp);

            if (value < INT32_MIN || value > INT32_MAX) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

                          "int32_t");

                return;

            }

            *obj = value;

        }

    }

}
