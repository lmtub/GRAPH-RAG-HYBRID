static int check_host_key_knownhosts(BDRVSSHState *s,

                                     const char *host, int port)

{

    const char *home;

    char *knh_file = NULL;

    LIBSSH2_KNOWNHOSTS *knh = NULL;

    struct libssh2_knownhost *found;

    int ret, r;

    const char *hostkey;

    size_t len;

    int type;



    hostkey = libssh2_session_hostkey(s->session, &len, &type);

    if (!hostkey) {

        ret = -EINVAL;

        session_error_report(s, "failed to read remote host key");

        goto out;

    }



    knh = libssh2_knownhost_init(s->session);

    if (!knh) {

        ret = -EINVAL;

        session_error_report(s, "failed to initialize known hosts support");

        goto out;

    }



    home = getenv("HOME");

    if (home) {

        knh_file = g_strdup_printf("%s/.ssh/known_hosts", home);

    } else {

        knh_file = g_strdup_printf("/root/.ssh/known_hosts");

    }



    /* Read all known hosts from OpenSSH-style known_hosts file. */

    libssh2_knownhost_readfile(knh, knh_file, LIBSSH2_KNOWNHOST_FILE_OPENSSH);



    r = libssh2_knownhost_checkp(knh, host, port, hostkey, len,

                                 LIBSSH2_KNOWNHOST_TYPE_PLAIN|

                                 LIBSSH2_KNOWNHOST_KEYENC_RAW,

                                 &found);

    switch (r) {

    case LIBSSH2_KNOWNHOST_CHECK_MATCH:

        /* OK */

        DPRINTF("host key OK: %s", found->key);

        break;

    case LIBSSH2_KNOWNHOST_CHECK_MISMATCH:

        ret = -EINVAL;

        session_error_report(s, "host key does not match the one in known_hosts (found key %s)",

                             found->key);

        goto out;

    case LIBSSH2_KNOWNHOST_CHECK_NOTFOUND:

        ret = -EINVAL;

        session_error_report(s, "no host key was found in known_hosts");

        goto out;

    case LIBSSH2_KNOWNHOST_CHECK_FAILURE:

        ret = -EINVAL;

        session_error_report(s, "failure matching the host key with known_hosts");

        goto out;

    default:

        ret = -EINVAL;

        session_error_report(s, "unknown error matching the host key with known_hosts (%d)",

                             r);

        goto out;

    }



    /* known_hosts checking successful. */

    ret = 0;



 out:

    if (knh != NULL) {

        libssh2_knownhost_free(knh);

    }

    g_free(knh_file);

    return ret;

}
