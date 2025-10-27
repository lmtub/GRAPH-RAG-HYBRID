static void parse_type_int64(Visitor *v, const char *name, int64_t *obj,

                             Error **errp)

{

    StringInputVisitor *siv = to_siv(v);



    if (!siv->string) {

        error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

                   "integer");

        return;

    }



    parse_str(siv, errp);



    if (!siv->ranges) {

        goto error;

    }



    if (!siv->cur_range) {

        Range *r;



        siv->cur_range = g_list_first(siv->ranges);

        if (!siv->cur_range) {

            goto error;

        }



        r = siv->cur_range->data;

        if (!r) {

            goto error;

        }



        siv->cur = r->begin;

    }



    *obj = siv->cur;

    siv->cur++;

    return;



error:

    error_setg(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

               "an int64 value or range");

}
