int net_init_slirp(const NetClientOptions *opts, const char *name,

                   NetClientState *peer, Error **errp)

{

    /* FIXME error_setg(errp, ...) on failure */

    struct slirp_config_str *config;

    char *vnet;

    int ret;

    const NetdevUserOptions *user;

    const char **dnssearch;



    assert(opts->kind == NET_CLIENT_OPTIONS_KIND_USER);

    user = opts->user;



    vnet = user->has_net ? g_strdup(user->net) :

           user->has_ip  ? g_strdup_printf("%s/24", user->ip) :

           NULL;



    dnssearch = slirp_dnssearch(user->dnssearch);



    /* all optional fields are initialized to "all bits zero" */



    net_init_slirp_configs(user->hostfwd, SLIRP_CFG_HOSTFWD);

    net_init_slirp_configs(user->guestfwd, 0);



    ret = net_slirp_init(peer, "user", name, user->q_restrict, vnet,

                         user->host, user->hostname, user->tftp,

                         user->bootfile, user->dhcpstart, user->dns, user->smb,

                         user->smbserver, dnssearch);



    while (slirp_configs) {

        config = slirp_configs;

        slirp_configs = config->next;

        g_free(config);

    }



    g_free(vnet);

    g_free(dnssearch);



    return ret;

}
