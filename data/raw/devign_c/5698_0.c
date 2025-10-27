static QDict *do_info_vnc_client(Monitor *mon, VncState *client)

{

    QDict *qdict;



    qdict = qdict_new();

    if (vnc_qdict_remote_addr(qdict, client->csock) < 0) {

        QDECREF(qdict);

        return NULL;

    }



#ifdef CONFIG_VNC_TLS

    if (client->tls.session &&

        client->tls.dname) {

        qdict_put(qdict, "x509_dname", qstring_from_str(client->tls.dname));

    }

#endif

#ifdef CONFIG_VNC_SASL

    if (client->sasl.conn &&

        client->sasl.username) {

        qdict_put(qdict, "sasl_username",

                  qstring_from_str(client->sasl.username));

    }

#endif



    return qdict;

}
