void error_vprepend(Error **errp, const char *fmt, va_list ap)

{

    GString *newmsg;



    if (!errp) {

        return;

    }



    newmsg = g_string_new(NULL);

    g_string_vprintf(newmsg, fmt, ap);

    g_string_append(newmsg, (*errp)->msg);


    (*errp)->msg = g_string_free(newmsg, 0);

}