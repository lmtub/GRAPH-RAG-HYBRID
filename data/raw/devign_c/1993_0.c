void visit_type_uint16(Visitor *v, uint16_t *obj, const char *name, Error **errp)

{

    int64_t value;



    if (v->type_uint16) {

        v->type_uint16(v, obj, name, errp);

    } else {

        value = *obj;

        v->type_int64(v, &value, name, errp);

        if (value < 0 || value > UINT16_MAX) {

            /* FIXME questionable reuse of errp if callback changed

               value on error */

            error_setg(errp, QERR_INVALID_PARAMETER_VALUE,

                       name ? name : "null", "uint16_t");

            return;

        }

        *obj = value;

    }

}
