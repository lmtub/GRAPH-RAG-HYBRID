static int nbd_receive_list(QIOChannel *ioc, char **name, Error **errp)

{

    uint64_t magic;

    uint32_t opt;

    uint32_t type;

    uint32_t len;

    uint32_t namelen;



    *name = NULL;

    if (read_sync(ioc, &magic, sizeof(magic)) != sizeof(magic)) {

        error_setg(errp, "failed to read list option magic");

        return -1;

    }

    magic = be64_to_cpu(magic);

    if (magic != NBD_REP_MAGIC) {

        error_setg(errp, "Unexpected option list magic");

        return -1;

    }

    if (read_sync(ioc, &opt, sizeof(opt)) != sizeof(opt)) {

        error_setg(errp, "failed to read list option");

        return -1;

    }

    opt = be32_to_cpu(opt);

    if (opt != NBD_OPT_LIST) {

        error_setg(errp, "Unexpected option type %x expected %x",

                   opt, NBD_OPT_LIST);

        return -1;

    }



    if (read_sync(ioc, &type, sizeof(type)) != sizeof(type)) {

        error_setg(errp, "failed to read list option type");

        return -1;

    }

    type = be32_to_cpu(type);

    if (type == NBD_REP_ERR_UNSUP) {

        return 0;

    }

    if (nbd_handle_reply_err(opt, type, errp) < 0) {

        return -1;

    }



    if (read_sync(ioc, &len, sizeof(len)) != sizeof(len)) {

        error_setg(errp, "failed to read option length");

        return -1;

    }

    len = be32_to_cpu(len);



    if (type == NBD_REP_ACK) {

        if (len != 0) {

            error_setg(errp, "length too long for option end");

            return -1;

        }

    } else if (type == NBD_REP_SERVER) {

        if (read_sync(ioc, &namelen, sizeof(namelen)) != sizeof(namelen)) {

            error_setg(errp, "failed to read option name length");

            return -1;

        }

        namelen = be32_to_cpu(namelen);

        if (len != (namelen + sizeof(namelen))) {

            error_setg(errp, "incorrect option mame length");

            return -1;

        }

        if (namelen > 255) {

            error_setg(errp, "export name length too long %d", namelen);

            return -1;

        }



        *name = g_new0(char, namelen + 1);

        if (read_sync(ioc, *name, namelen) != namelen) {

            error_setg(errp, "failed to read export name");

            g_free(*name);

            *name = NULL;

            return -1;

        }

        (*name)[namelen] = '\0';

    } else {

        error_setg(errp, "Unexpected reply type %x expected %x",

                   type, NBD_REP_SERVER);

        return -1;

    }

    return 1;

}
