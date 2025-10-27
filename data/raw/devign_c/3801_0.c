static int qio_dns_resolver_lookup_sync_nop(QIODNSResolver *resolver,

                                            SocketAddress *addr,

                                            size_t *naddrs,

                                            SocketAddress ***addrs,

                                            Error **errp)

{

    *naddrs = 1;

    *addrs = g_new0(SocketAddress *, 1);

    (*addrs)[0] = QAPI_CLONE(SocketAddress, addr);



    return 0;

}
