SocketAddressLegacy *socket_parse(const char *str, Error **errp)

{

    SocketAddressLegacy *addr;



    addr = g_new0(SocketAddressLegacy, 1);

    if (strstart(str, "unix:", NULL)) {

        if (str[5] == '\0') {

            error_setg(errp, "invalid Unix socket address");

            goto fail;

        } else {

            addr->type = SOCKET_ADDRESS_LEGACY_KIND_UNIX;

            addr->u.q_unix.data = g_new(UnixSocketAddress, 1);

            addr->u.q_unix.data->path = g_strdup(str + 5);

        }

    } else if (strstart(str, "fd:", NULL)) {

        if (str[3] == '\0') {

            error_setg(errp, "invalid file descriptor address");

            goto fail;

        } else {

            addr->type = SOCKET_ADDRESS_LEGACY_KIND_FD;

            addr->u.fd.data = g_new(String, 1);

            addr->u.fd.data->str = g_strdup(str + 3);

        }

    } else if (strstart(str, "vsock:", NULL)) {

        addr->type = SOCKET_ADDRESS_LEGACY_KIND_VSOCK;

        addr->u.vsock.data = g_new(VsockSocketAddress, 1);

        if (vsock_parse(addr->u.vsock.data, str + strlen("vsock:"), errp)) {

            goto fail;

        }

    } else {

        addr->type = SOCKET_ADDRESS_LEGACY_KIND_INET;

        addr->u.inet.data = g_new(InetSocketAddress, 1);

        if (inet_parse(addr->u.inet.data, str, errp)) {

            goto fail;

        }

    }

    return addr;



fail:

    qapi_free_SocketAddressLegacy(addr);

    return NULL;

}
