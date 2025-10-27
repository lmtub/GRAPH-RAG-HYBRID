static int net_slirp_init(NetClientState *peer, const char *model,

                          const char *name, int restricted,

                          bool ipv4, const char *vnetwork, const char *vhost,

                          bool ipv6, const char *vprefix6, int vprefix6_len,

                          const char *vhost6,

                          const char *vhostname, const char *tftp_export,

                          const char *bootfile, const char *vdhcp_start,

                          const char *vnameserver, const char *vnameserver6,

                          const char *smb_export, const char *vsmbserver,

                          const char **dnssearch)

{

    /* default settings according to historic slirp */

    struct in_addr net  = { .s_addr = htonl(0x0a000200) }; /* 10.0.2.0 */

    struct in_addr mask = { .s_addr = htonl(0xffffff00) }; /* 255.255.255.0 */

    struct in_addr host = { .s_addr = htonl(0x0a000202) }; /* 10.0.2.2 */

    struct in_addr dhcp = { .s_addr = htonl(0x0a00020f) }; /* 10.0.2.15 */

    struct in_addr dns  = { .s_addr = htonl(0x0a000203) }; /* 10.0.2.3 */

    struct in6_addr ip6_prefix;

    struct in6_addr ip6_host;

    struct in6_addr ip6_dns;

#ifndef _WIN32

    struct in_addr smbsrv = { .s_addr = 0 };

#endif

    NetClientState *nc;

    SlirpState *s;

    char buf[20];

    uint32_t addr;

    int shift;

    char *end;

    struct slirp_config_str *config;



    if (!ipv4 && (vnetwork || vhost || vnameserver)) {

        return -1;

    }



    if (!ipv6 && (vprefix6 || vhost6 || vnameserver6)) {

        return -1;

    }



    if (!ipv4 && !ipv6) {

        /* It doesn't make sense to disable both */

        return -1;

    }



    if (!tftp_export) {

        tftp_export = legacy_tftp_prefix;

    }

    if (!bootfile) {

        bootfile = legacy_bootp_filename;

    }



    if (vnetwork) {

        if (get_str_sep(buf, sizeof(buf), &vnetwork, '/') < 0) {

            if (!inet_aton(vnetwork, &net)) {

                return -1;

            }

            addr = ntohl(net.s_addr);

            if (!(addr & 0x80000000)) {

                mask.s_addr = htonl(0xff000000); /* class A */

            } else if ((addr & 0xfff00000) == 0xac100000) {

                mask.s_addr = htonl(0xfff00000); /* priv. 172.16.0.0/12 */

            } else if ((addr & 0xc0000000) == 0x80000000) {

                mask.s_addr = htonl(0xffff0000); /* class B */

            } else if ((addr & 0xffff0000) == 0xc0a80000) {

                mask.s_addr = htonl(0xffff0000); /* priv. 192.168.0.0/16 */

            } else if ((addr & 0xffff0000) == 0xc6120000) {

                mask.s_addr = htonl(0xfffe0000); /* tests 198.18.0.0/15 */

            } else if ((addr & 0xe0000000) == 0xe0000000) {

                mask.s_addr = htonl(0xffffff00); /* class C */

            } else {

                mask.s_addr = htonl(0xfffffff0); /* multicast/reserved */

            }

        } else {

            if (!inet_aton(buf, &net)) {

                return -1;

            }

            shift = strtol(vnetwork, &end, 10);

            if (*end != '\0') {

                if (!inet_aton(vnetwork, &mask)) {

                    return -1;

                }

            } else if (shift < 4 || shift > 32) {

                return -1;

            } else {

                mask.s_addr = htonl(0xffffffff << (32 - shift));

            }

        }

        net.s_addr &= mask.s_addr;

        host.s_addr = net.s_addr | (htonl(0x0202) & ~mask.s_addr);

        dhcp.s_addr = net.s_addr | (htonl(0x020f) & ~mask.s_addr);

        dns.s_addr  = net.s_addr | (htonl(0x0203) & ~mask.s_addr);

    }



    if (vhost && !inet_aton(vhost, &host)) {

        return -1;

    }

    if ((host.s_addr & mask.s_addr) != net.s_addr) {

        return -1;

    }



    if (vnameserver && !inet_aton(vnameserver, &dns)) {

        return -1;

    }

    if ((dns.s_addr & mask.s_addr) != net.s_addr ||

        dns.s_addr == host.s_addr) {

        return -1;

    }



    if (vdhcp_start && !inet_aton(vdhcp_start, &dhcp)) {

        return -1;

    }

    if ((dhcp.s_addr & mask.s_addr) != net.s_addr ||

        dhcp.s_addr == host.s_addr || dhcp.s_addr == dns.s_addr) {

        return -1;

    }



#ifndef _WIN32

    if (vsmbserver && !inet_aton(vsmbserver, &smbsrv)) {

        return -1;

    }

#endif



#if defined(_WIN32) && (_WIN32_WINNT < 0x0600)

    /* No inet_pton helper before Vista... */

    if (vprefix6) {

        /* Unsupported */

        return -1;

    }

    memset(&ip6_prefix, 0, sizeof(ip6_prefix));

    ip6_prefix.s6_addr[0] = 0xfe;

    ip6_prefix.s6_addr[1] = 0xc0;

#else

    if (!vprefix6) {

        vprefix6 = "fec0::";

    }

    if (!inet_pton(AF_INET6, vprefix6, &ip6_prefix)) {

        return -1;

    }

#endif



    if (!vprefix6_len) {

        vprefix6_len = 64;

    }

    if (vprefix6_len < 0 || vprefix6_len > 126) {

        return -1;

    }



    if (vhost6) {

#if defined(_WIN32) && (_WIN32_WINNT < 0x0600)

        return -1;

#else

        if (!inet_pton(AF_INET6, vhost6, &ip6_host)) {

            return -1;

        }

        if (!in6_equal_net(&ip6_prefix, &ip6_host, vprefix6_len)) {

            return -1;

        }

#endif

    } else {

        ip6_host = ip6_prefix;

        ip6_host.s6_addr[15] |= 2;

    }



    if (vnameserver6) {

#if defined(_WIN32) && (_WIN32_WINNT < 0x0600)

        return -1;

#else

        if (!inet_pton(AF_INET6, vnameserver6, &ip6_dns)) {

            return -1;

        }

        if (!in6_equal_net(&ip6_prefix, &ip6_dns, vprefix6_len)) {

            return -1;

        }

#endif

    } else {

        ip6_dns = ip6_prefix;

        ip6_dns.s6_addr[15] |= 3;

    }





    nc = qemu_new_net_client(&net_slirp_info, peer, model, name);



    snprintf(nc->info_str, sizeof(nc->info_str),

             "net=%s,restrict=%s", inet_ntoa(net),

             restricted ? "on" : "off");



    s = DO_UPCAST(SlirpState, nc, nc);



    s->slirp = slirp_init(restricted, ipv4, net, mask, host,

                          ipv6, ip6_prefix, vprefix6_len, ip6_host,

                          vhostname, tftp_export, bootfile, dhcp,

                          dns, ip6_dns, dnssearch, s);

    QTAILQ_INSERT_TAIL(&slirp_stacks, s, entry);



    for (config = slirp_configs; config; config = config->next) {

        if (config->flags & SLIRP_CFG_HOSTFWD) {

            if (slirp_hostfwd(s, config->str,

                              config->flags & SLIRP_CFG_LEGACY) < 0)

                goto error;

        } else {

            if (slirp_guestfwd(s, config->str,

                               config->flags & SLIRP_CFG_LEGACY) < 0)

                goto error;

        }

    }

#ifndef _WIN32

    if (!smb_export) {

        smb_export = legacy_smb_export;

    }

    if (smb_export) {

        if (slirp_smb(s, smb_export, smbsrv) < 0)

            goto error;

    }

#endif



    s->exit_notifier.notify = slirp_smb_exit;

    qemu_add_exit_notifier(&s->exit_notifier);

    return 0;



error:

    qemu_del_net_client(nc);

    return -1;

}
