qcrypto_tls_session_check_certificate(QCryptoTLSSession *session,

                                      Error **errp)

{

    int ret;

    unsigned int status;

    const gnutls_datum_t *certs;

    unsigned int nCerts, i;

    time_t now;

    gnutls_x509_crt_t cert = NULL;



    now = time(NULL);

    if (now == ((time_t)-1)) {

        error_setg_errno(errp, errno, "Cannot get current time");

        return -1;

    }



    ret = gnutls_certificate_verify_peers2(session->handle, &status);

    if (ret < 0) {

        error_setg(errp, "Verify failed: %s", gnutls_strerror(ret));

        return -1;

    }



    if (status != 0) {

        const char *reason = "Invalid certificate";



        if (status & GNUTLS_CERT_INVALID) {

            reason = "The certificate is not trusted";

        }



        if (status & GNUTLS_CERT_SIGNER_NOT_FOUND) {

            reason = "The certificate hasn't got a known issuer";

        }



        if (status & GNUTLS_CERT_REVOKED) {

            reason = "The certificate has been revoked";

        }



        if (status & GNUTLS_CERT_INSECURE_ALGORITHM) {

            reason = "The certificate uses an insecure algorithm";

        }



        error_setg(errp, "%s", reason);

        return -1;

    }



    certs = gnutls_certificate_get_peers(session->handle, &nCerts);

    if (!certs) {

        error_setg(errp, "No certificate peers");

        return -1;

    }



    for (i = 0; i < nCerts; i++) {

        ret = gnutls_x509_crt_init(&cert);

        if (ret < 0) {

            error_setg(errp, "Cannot initialize certificate: %s",

                       gnutls_strerror(ret));

            return -1;

        }



        ret = gnutls_x509_crt_import(cert, &certs[i], GNUTLS_X509_FMT_DER);

        if (ret < 0) {

            error_setg(errp, "Cannot import certificate: %s",

                       gnutls_strerror(ret));

            goto error;

        }



        if (gnutls_x509_crt_get_expiration_time(cert) < now) {

            error_setg(errp, "The certificate has expired");

            goto error;

        }



        if (gnutls_x509_crt_get_activation_time(cert) > now) {

            error_setg(errp, "The certificate is not yet activated");

            goto error;

        }



        if (gnutls_x509_crt_get_activation_time(cert) > now) {

            error_setg(errp, "The certificate is not yet activated");

            goto error;

        }



        if (i == 0) {

            size_t dnameSize = 1024;

            session->peername = g_malloc(dnameSize);

        requery:

            ret = gnutls_x509_crt_get_dn(cert, session->peername, &dnameSize);

            if (ret < 0) {

                if (ret == GNUTLS_E_SHORT_MEMORY_BUFFER) {

                    session->peername = g_realloc(session->peername,

                                                  dnameSize);

                    goto requery;

                }

                error_setg(errp, "Cannot get client distinguished name: %s",

                           gnutls_strerror(ret));

                goto error;

            }

            if (session->aclname) {

                qemu_acl *acl = qemu_acl_find(session->aclname);

                int allow;

                if (!acl) {

                    error_setg(errp, "Cannot find ACL %s",

                               session->aclname);

                    goto error;

                }



                allow = qemu_acl_party_is_allowed(acl, session->peername);



                error_setg(errp, "TLS x509 ACL check for %s is %s",

                           session->peername, allow ? "allowed" : "denied");

                if (!allow) {

                    goto error;

                }

            }

            if (session->hostname) {

                if (!gnutls_x509_crt_check_hostname(cert, session->hostname)) {

                    error_setg(errp,

                               "Certificate does not match the hostname %s",

                               session->hostname);

                    goto error;

                }

            }

        }



        gnutls_x509_crt_deinit(cert);

    }



    return 0;



 error:

    gnutls_x509_crt_deinit(cert);

    return -1;

}
