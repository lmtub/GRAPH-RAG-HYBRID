void visit_type_int16(Visitor *v, int16_t *obj, const char *name, Error **errp)

{

    int64_t value;

    if (!error_is_set(errp)) {

        if (v->type_int16) {

            v->type_int16(v, obj, name, errp);

        } else {

            value = *obj;

            v->type_int(v, &value, name, errp);

            if (value < INT16_MIN || value > INT16_MAX) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

                          "int16_t");

                return;

            }

            *obj = value;

        }

    }

}
