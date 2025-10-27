void visit_type_int8(Visitor *v, int8_t *obj, const char *name, Error **errp)

{

    int64_t value;

    if (!error_is_set(errp)) {

        if (v->type_int8) {

            v->type_int8(v, obj, name, errp);

        } else {

            value = *obj;

            v->type_int(v, &value, name, errp);

            if (value < INT8_MIN || value > INT8_MAX) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

                          "int8_t");

                return;

            }

            *obj = value;

        }

    }

}
