static int nbd_handle_reply_err(uint32_t opt, uint32_t type, Error **errp)

{

    if (!(type & (1 << 31))) {

        return 0;

    }



    switch (type) {

    case NBD_REP_ERR_UNSUP:

        error_setg(errp, "Unsupported option type %x", opt);

        break;



    case NBD_REP_ERR_POLICY:

        error_setg(errp, "Denied by server for option %x", opt);

        break;



    case NBD_REP_ERR_INVALID:

        error_setg(errp, "Invalid data length for option %x", opt);

        break;



    case NBD_REP_ERR_TLS_REQD:

        error_setg(errp, "TLS negotiation required before option %x", opt);

        break;



    default:

        error_setg(errp, "Unknown error code when asking for option %x", opt);

        break;

    }



    return -1;

}
