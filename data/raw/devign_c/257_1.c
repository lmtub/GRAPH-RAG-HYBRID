static void parse_type_bool(Visitor *v, const char *name, bool *obj,

                            Error **errp)

{

    StringInputVisitor *siv = to_siv(v);



    if (siv->string) {

        if (!strcasecmp(siv->string, "on") ||

            !strcasecmp(siv->string, "yes") ||

            !strcasecmp(siv->string, "true")) {

            *obj = true;

            return;

        }

        if (!strcasecmp(siv->string, "off") ||

            !strcasecmp(siv->string, "no") ||

            !strcasecmp(siv->string, "false")) {

            *obj = false;

            return;

        }

    }



    error_setg(errp, QERR_INVALID_PARAMETER_TYPE, name ? name : "null",

               "boolean");

}
