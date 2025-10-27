void migration_tls_channel_connect(MigrationState *s,

                                   QIOChannel *ioc,

                                   const char *hostname,

                                   Error **errp)

{

    QCryptoTLSCreds *creds;

    QIOChannelTLS *tioc;



    creds = migration_tls_get_creds(

        s, QCRYPTO_TLS_CREDS_ENDPOINT_CLIENT, errp);

    if (!creds) {

        return;

    }



    if (s->parameters.tls_hostname) {

        hostname = s->parameters.tls_hostname;

    }

    if (!hostname) {

        error_setg(errp, "No hostname available for TLS");

        return;

    }



    tioc = qio_channel_tls_new_client(

        ioc, creds, hostname, errp);

    if (!tioc) {

        return;

    }



    trace_migration_tls_outgoing_handshake_start(hostname);

    qio_channel_set_name(QIO_CHANNEL(tioc), "migration-tls-outgoing");

    qio_channel_tls_handshake(tioc,

                              migration_tls_outgoing_handshake,

                              s,

                              NULL);

}
