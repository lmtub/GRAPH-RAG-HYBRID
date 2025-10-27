void visit_type_uint32(Visitor *v, uint32_t *obj, const char *name, Error **errp)

{

    int64_t value;

    if (!error_is_set(errp)) {

        if (v->type_uint32) {

            v->type_uint32(v, obj, name, errp);

        } else {

            value = *obj;

            v->type_int(v, &value, name, errp);

            if (value < 0 || value > UINT32_MAX) {

                error_set(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

                          "uint32_t");

                return;

            }

            *obj = value;

        }

    }

}
