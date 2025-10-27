int vnc_tls_validate_certificate(VncState *vs)

{

    int ret;

    unsigned int status;

    const gnutls_datum_t *certs;

    unsigned int nCerts, i;

    time_t now;



    VNC_DEBUG("Validating client certificate\n");

    if ((ret = gnutls_certificate_verify_peers2 (vs->tls.session, &status)) < 0) {

        VNC_DEBUG("Verify failed %s\n", gnutls_strerror(ret));

        return -1;

    }



    if ((now = time(NULL)) == ((time_t)-1)) {

        return -1;

    }



    if (status != 0) {

        if (status & GNUTLS_CERT_INVALID)

            VNC_DEBUG("The certificate is not trusted.\n");



        if (status & GNUTLS_CERT_SIGNER_NOT_FOUND)

            VNC_DEBUG("The certificate hasn't got a known issuer.\n");



        if (status & GNUTLS_CERT_REVOKED)

            VNC_DEBUG("The certificate has been revoked.\n");



        if (status & GNUTLS_CERT_INSECURE_ALGORITHM)

            VNC_DEBUG("The certificate uses an insecure algorithm\n");



        return -1;

    } else {

        VNC_DEBUG("Certificate is valid!\n");

    }



    /* Only support x509 for now */

    if (gnutls_certificate_type_get(vs->tls.session) != GNUTLS_CRT_X509)

        return -1;



    if (!(certs = gnutls_certificate_get_peers(vs->tls.session, &nCerts)))

        return -1;



    for (i = 0 ; i < nCerts ; i++) {

        gnutls_x509_crt_t cert;

        VNC_DEBUG ("Checking certificate chain %d\n", i);

        if (gnutls_x509_crt_init (&cert) < 0)

            return -1;



        if (gnutls_x509_crt_import(cert, &certs[i], GNUTLS_X509_FMT_DER) < 0) {

            gnutls_x509_crt_deinit (cert);

            return -1;

        }



        if (gnutls_x509_crt_get_expiration_time (cert) < now) {

            VNC_DEBUG("The certificate has expired\n");

            gnutls_x509_crt_deinit (cert);

            return -1;

        }



        if (gnutls_x509_crt_get_activation_time (cert) > now) {

            VNC_DEBUG("The certificate is not yet activated\n");

            gnutls_x509_crt_deinit (cert);

            return -1;

        }



        if (gnutls_x509_crt_get_activation_time (cert) > now) {

            VNC_DEBUG("The certificate is not yet activated\n");

            gnutls_x509_crt_deinit (cert);

            return -1;

        }



        if (i == 0) {

            size_t dnameSize = 1024;

            vs->tls.dname = g_malloc(dnameSize);

        requery:

            if ((ret = gnutls_x509_crt_get_dn (cert, vs->tls.dname, &dnameSize)) != 0) {

                if (ret == GNUTLS_E_SHORT_MEMORY_BUFFER) {

                    vs->tls.dname = g_realloc(vs->tls.dname, dnameSize);

                    goto requery;

                }

                gnutls_x509_crt_deinit (cert);

                VNC_DEBUG("Cannot get client distinguished name: %s",

                          gnutls_strerror (ret));

                return -1;

            }



            if (vs->vd->tls.x509verify) {

                int allow;

                if (!vs->vd->tls.acl) {

                    VNC_DEBUG("no ACL activated, allowing access");

                    gnutls_x509_crt_deinit (cert);

                    continue;

                }



                allow = qemu_acl_party_is_allowed(vs->vd->tls.acl,

                                                  vs->tls.dname);



                VNC_DEBUG("TLS x509 ACL check for %s is %s\n",

                          vs->tls.dname, allow ? "allowed" : "denied");

                if (!allow) {

                    gnutls_x509_crt_deinit (cert);

                    return -1;

                }

            }

        }



        gnutls_x509_crt_deinit (cert);

    }



    return 0;

}
