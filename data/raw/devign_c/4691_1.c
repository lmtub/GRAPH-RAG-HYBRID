static int connect_to_ssh(BDRVSSHState *s, QDict *options,

                          int ssh_flags, int creat_mode, Error **errp)

{

    int r, ret;

    const char *host, *user, *path, *host_key_check;

    int port;



    if (!qdict_haskey(options, "host")) {

        ret = -EINVAL;

        error_setg(errp, "No hostname was specified");

        goto err;

    }

    host = qdict_get_str(options, "host");



    if (qdict_haskey(options, "port")) {

        port = qdict_get_int(options, "port");

    } else {

        port = 22;

    }



    if (!qdict_haskey(options, "path")) {

        ret = -EINVAL;

        error_setg(errp, "No path was specified");

        goto err;

    }

    path = qdict_get_str(options, "path");



    if (qdict_haskey(options, "user")) {

        user = qdict_get_str(options, "user");

    } else {

        user = g_get_user_name();

        if (!user) {

            error_setg_errno(errp, errno, "Can't get user name");

            ret = -errno;

            goto err;

        }

    }



    if (qdict_haskey(options, "host_key_check")) {

        host_key_check = qdict_get_str(options, "host_key_check");

    } else {

        host_key_check = "yes";

    }



    /* Construct the host:port name for inet_connect. */

    g_free(s->hostport);

    s->hostport = g_strdup_printf("%s:%d", host, port);



    /* Open the socket and connect. */

    s->sock = inet_connect(s->hostport, errp);

    if (s->sock < 0) {

        ret = -EIO;

        goto err;

    }



    /* Create SSH session. */

    s->session = libssh2_session_init();

    if (!s->session) {

        ret = -EINVAL;

        session_error_setg(errp, s, "failed to initialize libssh2 session");

        goto err;

    }



#if TRACE_LIBSSH2 != 0

    libssh2_trace(s->session, TRACE_LIBSSH2);

#endif



    r = libssh2_session_handshake(s->session, s->sock);

    if (r != 0) {

        ret = -EINVAL;

        session_error_setg(errp, s, "failed to establish SSH session");

        goto err;

    }



    /* Check the remote host's key against known_hosts. */

    ret = check_host_key(s, host, port, host_key_check, errp);

    if (ret < 0) {

        goto err;

    }



    /* Authenticate. */

    ret = authenticate(s, user, errp);

    if (ret < 0) {

        goto err;

    }



    /* Start SFTP. */

    s->sftp = libssh2_sftp_init(s->session);

    if (!s->sftp) {

        session_error_setg(errp, s, "failed to initialize sftp handle");

        ret = -EINVAL;

        goto err;

    }



    /* Open the remote file. */

    DPRINTF("opening file %s flags=0x%x creat_mode=0%o",

            path, ssh_flags, creat_mode);

    s->sftp_handle = libssh2_sftp_open(s->sftp, path, ssh_flags, creat_mode);

    if (!s->sftp_handle) {

        session_error_setg(errp, s, "failed to open remote file '%s'", path);

        ret = -EINVAL;

        goto err;

    }



    r = libssh2_sftp_fstat(s->sftp_handle, &s->attrs);

    if (r < 0) {

        sftp_error_setg(errp, s, "failed to read file attributes");

        return -EINVAL;

    }



    /* Delete the options we've used; any not deleted will cause the

     * block layer to give an error about unused options.

     */

    qdict_del(options, "host");

    qdict_del(options, "port");

    qdict_del(options, "user");

    qdict_del(options, "path");

    qdict_del(options, "host_key_check");



    return 0;



 err:

    if (s->sftp_handle) {

        libssh2_sftp_close(s->sftp_handle);

    }

    s->sftp_handle = NULL;

    if (s->sftp) {

        libssh2_sftp_shutdown(s->sftp);

    }

    s->sftp = NULL;

    if (s->session) {

        libssh2_session_disconnect(s->session,

                                   "from qemu ssh client: "

                                   "error opening connection");

        libssh2_session_free(s->session);

    }

    s->session = NULL;



    return ret;

}
