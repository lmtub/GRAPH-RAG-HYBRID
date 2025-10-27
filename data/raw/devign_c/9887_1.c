static int vnc_tls_initialize(void)

{

    static int tlsinitialized = 0;



    if (tlsinitialized)

        return 1;



    if (gnutls_global_init () < 0)

        return 0;



    /* XXX ought to re-generate diffie-hellman params periodically */

    if (gnutls_dh_params_init (&dh_params) < 0)

        return 0;

    if (gnutls_dh_params_generate2 (dh_params, DH_BITS) < 0)

        return 0;



#if defined(_VNC_DEBUG) && _VNC_DEBUG >= 2

    gnutls_global_set_log_level(10);

    gnutls_global_set_log_function(vnc_debug_gnutls_log);

#endif



    tlsinitialized = 1;



    return 1;

}
