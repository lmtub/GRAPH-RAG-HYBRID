static int nbd_send_rep_list(int csock, NBDExport *exp)

{

    uint64_t magic, name_len;

    uint32_t opt, type, len;



    name_len = strlen(exp->name);

    magic = cpu_to_be64(NBD_REP_MAGIC);

    if (write_sync(csock, &magic, sizeof(magic)) != sizeof(magic)) {

        LOG("write failed (magic)");

        return -EINVAL;

     }

    opt = cpu_to_be32(NBD_OPT_LIST);

    if (write_sync(csock, &opt, sizeof(opt)) != sizeof(opt)) {

        LOG("write failed (opt)");

        return -EINVAL;

    }

    type = cpu_to_be32(NBD_REP_SERVER);

    if (write_sync(csock, &type, sizeof(type)) != sizeof(type)) {

        LOG("write failed (reply type)");

        return -EINVAL;

    }

    len = cpu_to_be32(name_len + sizeof(len));

    if (write_sync(csock, &len, sizeof(len)) != sizeof(len)) {

        LOG("write failed (length)");

        return -EINVAL;

    }

    len = cpu_to_be32(name_len);

    if (write_sync(csock, &len, sizeof(len)) != sizeof(len)) {

        LOG("write failed (length)");

        return -EINVAL;

    }

    if (write_sync(csock, exp->name, name_len) != name_len) {

        LOG("write failed (buffer)");

        return -EINVAL;

    }

    return 0;

}
