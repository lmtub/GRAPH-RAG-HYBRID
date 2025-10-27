int net_client_init(Monitor *mon, const char *device, const char *p)

{

    char buf[1024];

    int vlan_id, ret;

    VLANState *vlan;

    char *name = NULL;



    vlan_id = 0;

    if (get_param_value(buf, sizeof(buf), "vlan", p)) {

        vlan_id = strtol(buf, NULL, 0);

    }

    vlan = qemu_find_vlan(vlan_id, 1);



    if (get_param_value(buf, sizeof(buf), "name", p)) {

        name = qemu_strdup(buf);

    }

    if (!strcmp(device, "nic")) {

        static const char * const nic_params[] = {

            "vlan", "name", "macaddr", "model", "addr", "id", "vectors", NULL

        };

        NICInfo *nd;

        uint8_t *macaddr;

        int idx = nic_get_free_idx();



        if (check_params(buf, sizeof(buf), nic_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (idx == -1 || nb_nics >= MAX_NICS) {

            config_error(mon, "Too Many NICs\n");

            ret = -1;

            goto out;

        }

        nd = &nd_table[idx];

        memset(nd, 0, sizeof(*nd));

        macaddr = nd->macaddr;

        macaddr[0] = 0x52;

        macaddr[1] = 0x54;

        macaddr[2] = 0x00;

        macaddr[3] = 0x12;

        macaddr[4] = 0x34;

        macaddr[5] = 0x56 + idx;



        if (get_param_value(buf, sizeof(buf), "macaddr", p)) {

            if (parse_macaddr(macaddr, buf) < 0) {

                config_error(mon, "invalid syntax for ethernet address\n");

                ret = -1;

                goto out;

            }

        }

        if (get_param_value(buf, sizeof(buf), "model", p)) {

            nd->model = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "addr", p)) {

            nd->devaddr = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "id", p)) {

            nd->id = qemu_strdup(buf);

        }

        nd->nvectors = NIC_NVECTORS_UNSPECIFIED;

        if (get_param_value(buf, sizeof(buf), "vectors", p)) {

            char *endptr;

            long vectors = strtol(buf, &endptr, 0);

            if (*endptr) {

                config_error(mon, "invalid syntax for # of vectors\n");

                ret = -1;

                goto out;

            }

            if (vectors < 0 || vectors > 0x7ffffff) {

                config_error(mon, "invalid # of vectors\n");

                ret = -1;

                goto out;

            }

            nd->nvectors = vectors;

        }

        nd->vlan = vlan;

        nd->name = name;

        nd->used = 1;

        name = NULL;

        nb_nics++;

        vlan->nb_guest_devs++;

        ret = idx;

    } else

    if (!strcmp(device, "none")) {

        if (*p != '\0') {

            config_error(mon, "'none' takes no parameters\n");

            ret = -1;

            goto out;

        }

        /* does nothing. It is needed to signal that no network cards

           are wanted */

        ret = 0;

    } else

#ifdef CONFIG_SLIRP

    if (!strcmp(device, "user")) {

        static const char * const slirp_params[] = {

            "vlan", "name", "hostname", "restrict", "ip", "net", "host",

            "tftp", "bootfile", "dhcpstart", "dns", "smb", "smbserver",

            "hostfwd", "guestfwd", NULL

        };

        struct slirp_config_str *config;

        int restricted = 0;

        char *vnet = NULL;

        char *vhost = NULL;

        char *vhostname = NULL;

        char *tftp_export = NULL;

        char *bootfile = NULL;

        char *vdhcp_start = NULL;

        char *vnamesrv = NULL;

        char *smb_export = NULL;

        char *vsmbsrv = NULL;

        const char *q;



        if (check_params(buf, sizeof(buf), slirp_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (get_param_value(buf, sizeof(buf), "ip", p)) {

            int vnet_buflen = strlen(buf) + strlen("/24") + 1;

            /* emulate legacy parameter */

            vnet = qemu_malloc(vnet_buflen);

            pstrcpy(vnet, vnet_buflen, buf);

            pstrcat(vnet, vnet_buflen, "/24");

        }

        if (get_param_value(buf, sizeof(buf), "net", p)) {

            vnet = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "host", p)) {

            vhost = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "hostname", p)) {

            vhostname = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "restrict", p)) {

            restricted = (buf[0] == 'y') ? 1 : 0;

        }

        if (get_param_value(buf, sizeof(buf), "dhcpstart", p)) {

            vdhcp_start = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "dns", p)) {

            vnamesrv = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "tftp", p)) {

            tftp_export = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "bootfile", p)) {

            bootfile = qemu_strdup(buf);

        }

        if (get_param_value(buf, sizeof(buf), "smb", p)) {

            smb_export = qemu_strdup(buf);

            if (get_param_value(buf, sizeof(buf), "smbserver", p)) {

                vsmbsrv = qemu_strdup(buf);

            }

        }

        q = p;

        while (1) {

            config = qemu_malloc(sizeof(*config));

            if (!get_next_param_value(config->str, sizeof(config->str),

                                      "hostfwd", &q)) {

                break;

            }

            config->flags = SLIRP_CFG_HOSTFWD;

            config->next = slirp_configs;

            slirp_configs = config;

            config = NULL;

        }

        q = p;

        while (1) {

            config = qemu_malloc(sizeof(*config));

            if (!get_next_param_value(config->str, sizeof(config->str),

                                      "guestfwd", &q)) {

                break;

            }

            config->flags = 0;

            config->next = slirp_configs;

            slirp_configs = config;

            config = NULL;

        }

        qemu_free(config);

        vlan->nb_host_devs++;

        ret = net_slirp_init(mon, vlan, device, name, restricted, vnet, vhost,

                             vhostname, tftp_export, bootfile, vdhcp_start,

                             vnamesrv, smb_export, vsmbsrv);

        while (slirp_configs) {

            config = slirp_configs;

            slirp_configs = config->next;

            qemu_free(config);

        }

        qemu_free(vnet);

        qemu_free(vhost);

        qemu_free(vhostname);

        qemu_free(tftp_export);

        qemu_free(bootfile);

        qemu_free(vdhcp_start);

        qemu_free(vnamesrv);

        qemu_free(smb_export);

        qemu_free(vsmbsrv);

    } else if (!strcmp(device, "channel")) {

        if (QTAILQ_EMPTY(&slirp_stacks)) {

            struct slirp_config_str *config;



            config = qemu_malloc(sizeof(*config));

            pstrcpy(config->str, sizeof(config->str), p);

            config->flags = SLIRP_CFG_LEGACY;

            config->next = slirp_configs;

            slirp_configs = config;

        } else {

            slirp_guestfwd(QTAILQ_FIRST(&slirp_stacks), mon, p, 1);

        }

        ret = 0;

    } else

#endif

#ifdef _WIN32

    if (!strcmp(device, "tap")) {

        static const char * const tap_params[] = {

            "vlan", "name", "ifname", NULL

        };

        char ifname[64];



        if (check_params(buf, sizeof(buf), tap_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

            config_error(mon, "tap: no interface name\n");

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

        ret = tap_win32_init(vlan, device, name, ifname);

    } else

#elif defined (_AIX)

#else

    if (!strcmp(device, "tap")) {

        char ifname[64], chkbuf[64];

        char setup_script[1024], down_script[1024];

        TAPState *s;

        int fd;

        vlan->nb_host_devs++;

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            static const char * const fd_params[] = {

                "vlan", "name", "fd", "sndbuf", NULL

            };

            ret = -1;

            if (check_params(chkbuf, sizeof(chkbuf), fd_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                goto out;

            }

            fd = net_handle_fd_param(mon, buf);

            if (fd == -1) {

                goto out;

            }

            fcntl(fd, F_SETFL, O_NONBLOCK);

            s = net_tap_fd_init(vlan, device, name, fd);

            if (!s) {

                close(fd);

            }

        } else {

            static const char * const tap_params[] = {

                "vlan", "name", "ifname", "script", "downscript", "sndbuf", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), tap_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            if (get_param_value(ifname, sizeof(ifname), "ifname", p) <= 0) {

                ifname[0] = '\0';

            }

            if (get_param_value(setup_script, sizeof(setup_script), "script", p) == 0) {

                pstrcpy(setup_script, sizeof(setup_script), DEFAULT_NETWORK_SCRIPT);

            }

            if (get_param_value(down_script, sizeof(down_script), "downscript", p) == 0) {

                pstrcpy(down_script, sizeof(down_script), DEFAULT_NETWORK_DOWN_SCRIPT);

            }

            s = net_tap_init(vlan, device, name, ifname, setup_script, down_script);

        }

        if (s != NULL) {

            const char *sndbuf_str = NULL;

            if (get_param_value(buf, sizeof(buf), "sndbuf", p)) {

                sndbuf_str = buf;

            }

            tap_set_sndbuf(s, sndbuf_str, mon);

            ret = 0;

        } else {

            ret = -1;

        }

    } else

#endif

    if (!strcmp(device, "socket")) {

        char chkbuf[64];

        if (get_param_value(buf, sizeof(buf), "fd", p) > 0) {

            static const char * const fd_params[] = {

                "vlan", "name", "fd", NULL

            };

            int fd;

            ret = -1;

            if (check_params(chkbuf, sizeof(chkbuf), fd_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                goto out;

            }

            fd = net_handle_fd_param(mon, buf);

            if (fd == -1) {

                goto out;

            }

            if (!net_socket_fd_init(vlan, device, name, fd, 1)) {

                close(fd);

                goto out;

            }

            ret = 0;

        } else if (get_param_value(buf, sizeof(buf), "listen", p) > 0) {

            static const char * const listen_params[] = {

                "vlan", "name", "listen", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), listen_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_listen_init(vlan, device, name, buf);

        } else if (get_param_value(buf, sizeof(buf), "connect", p) > 0) {

            static const char * const connect_params[] = {

                "vlan", "name", "connect", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), connect_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_connect_init(vlan, device, name, buf);

        } else if (get_param_value(buf, sizeof(buf), "mcast", p) > 0) {

            static const char * const mcast_params[] = {

                "vlan", "name", "mcast", NULL

            };

            if (check_params(chkbuf, sizeof(chkbuf), mcast_params, p) < 0) {

                config_error(mon, "invalid parameter '%s' in '%s'\n", chkbuf, p);

                ret = -1;

                goto out;

            }

            ret = net_socket_mcast_init(vlan, device, name, buf);

        } else {

            config_error(mon, "Unknown socket options: %s\n", p);

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

    } else

#ifdef CONFIG_VDE

    if (!strcmp(device, "vde")) {

        static const char * const vde_params[] = {

            "vlan", "name", "sock", "port", "group", "mode", NULL

        };

        char vde_sock[1024], vde_group[512];

	int vde_port, vde_mode;



        if (check_params(buf, sizeof(buf), vde_params, p) < 0) {

            config_error(mon, "invalid parameter '%s' in '%s'\n", buf, p);

            ret = -1;

            goto out;

        }

        vlan->nb_host_devs++;

        if (get_param_value(vde_sock, sizeof(vde_sock), "sock", p) <= 0) {

	    vde_sock[0] = '\0';

	}

	if (get_param_value(buf, sizeof(buf), "port", p) > 0) {

	    vde_port = strtol(buf, NULL, 10);

	} else {

	    vde_port = 0;

	}

	if (get_param_value(vde_group, sizeof(vde_group), "group", p) <= 0) {

	    vde_group[0] = '\0';

	}

	if (get_param_value(buf, sizeof(buf), "mode", p) > 0) {

	    vde_mode = strtol(buf, NULL, 8);

	} else {

	    vde_mode = 0700;

	}

	ret = net_vde_init(vlan, device, name, vde_sock, vde_port, vde_group, vde_mode);

    } else

#endif

    if (!strcmp(device, "dump")) {

        int len = 65536;



        if (get_param_value(buf, sizeof(buf), "len", p) > 0) {

            len = strtol(buf, NULL, 0);

        }

        if (!get_param_value(buf, sizeof(buf), "file", p)) {

            snprintf(buf, sizeof(buf), "qemu-vlan%d.pcap", vlan_id);

        }

        ret = net_dump_init(mon, vlan, device, name, buf, len);

    } else {

        config_error(mon, "Unknown network device: %s\n", device);

        ret = -1;

        goto out;

    }

    if (ret < 0) {

        config_error(mon, "Could not initialize device '%s'\n", device);

    }

out:

    qemu_free(name);

    return ret;

}
