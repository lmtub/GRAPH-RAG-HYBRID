int nbd_receive_negotiate(int csock, const char *name, uint32_t *flags,

                          off_t *size, size_t *blocksize)

{

    char buf[256];

    uint64_t magic, s;

    uint16_t tmp;



    TRACE("Receiving negotiation.");



    if (read_sync(csock, buf, 8) != 8) {

        LOG("read failed");

        errno = EINVAL;

        return -1;

    }



    buf[8] = '\0';

    if (strlen(buf) == 0) {

        LOG("server connection closed");

        errno = EINVAL;

        return -1;

    }



    TRACE("Magic is %c%c%c%c%c%c%c%c",

          qemu_isprint(buf[0]) ? buf[0] : '.',

          qemu_isprint(buf[1]) ? buf[1] : '.',

          qemu_isprint(buf[2]) ? buf[2] : '.',

          qemu_isprint(buf[3]) ? buf[3] : '.',

          qemu_isprint(buf[4]) ? buf[4] : '.',

          qemu_isprint(buf[5]) ? buf[5] : '.',

          qemu_isprint(buf[6]) ? buf[6] : '.',

          qemu_isprint(buf[7]) ? buf[7] : '.');



    if (memcmp(buf, "NBDMAGIC", 8) != 0) {

        LOG("Invalid magic received");

        errno = EINVAL;

        return -1;

    }



    if (read_sync(csock, &magic, sizeof(magic)) != sizeof(magic)) {

        LOG("read failed");

        errno = EINVAL;

        return -1;

    }

    magic = be64_to_cpu(magic);

    TRACE("Magic is 0x%" PRIx64, magic);



    if (name) {

        uint32_t reserved = 0;

        uint32_t opt;

        uint32_t namesize;



        TRACE("Checking magic (opts_magic)");

        if (magic != 0x49484156454F5054LL) {

            LOG("Bad magic received");

            errno = EINVAL;

            return -1;

        }

        if (read_sync(csock, &tmp, sizeof(tmp)) != sizeof(tmp)) {

            LOG("flags read failed");

            errno = EINVAL;

            return -1;

        }

        *flags = be16_to_cpu(tmp) << 16;

        /* reserved for future use */

        if (write_sync(csock, &reserved, sizeof(reserved)) !=

            sizeof(reserved)) {

            LOG("write failed (reserved)");

            errno = EINVAL;

            return -1;

        }

        /* write the export name */

        magic = cpu_to_be64(magic);

        if (write_sync(csock, &magic, sizeof(magic)) != sizeof(magic)) {

            LOG("write failed (magic)");

            errno = EINVAL;

            return -1;

        }

        opt = cpu_to_be32(NBD_OPT_EXPORT_NAME);

        if (write_sync(csock, &opt, sizeof(opt)) != sizeof(opt)) {

            LOG("write failed (opt)");

            errno = EINVAL;

            return -1;

        }

        namesize = cpu_to_be32(strlen(name));

        if (write_sync(csock, &namesize, sizeof(namesize)) !=

            sizeof(namesize)) {

            LOG("write failed (namesize)");

            errno = EINVAL;

            return -1;

        }

        if (write_sync(csock, (char*)name, strlen(name)) != strlen(name)) {

            LOG("write failed (name)");

            errno = EINVAL;

            return -1;

        }

    } else {

        TRACE("Checking magic (cli_magic)");



        if (magic != 0x00420281861253LL) {

            LOG("Bad magic received");

            errno = EINVAL;

            return -1;

        }

    }



    if (read_sync(csock, &s, sizeof(s)) != sizeof(s)) {

        LOG("read failed");

        errno = EINVAL;

        return -1;

    }

    *size = be64_to_cpu(s);

    *blocksize = 1024;

    TRACE("Size is %" PRIu64, *size);



    if (!name) {

        if (read_sync(csock, flags, sizeof(*flags)) != sizeof(*flags)) {

            LOG("read failed (flags)");

            errno = EINVAL;

            return -1;

        }

        *flags = be32_to_cpup(flags);

    } else {

        if (read_sync(csock, &tmp, sizeof(tmp)) != sizeof(tmp)) {

            LOG("read failed (tmp)");

            errno = EINVAL;

            return -1;

        }

        *flags |= be32_to_cpu(tmp);

    }

    if (read_sync(csock, &buf, 124) != 124) {

        LOG("read failed (buf)");

        errno = EINVAL;

        return -1;

    }

        return 0;

}
