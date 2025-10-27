int net_init_dump(const NetClientOptions *opts, const char *name,

                  NetClientState *peer, Error **errp)

{

    int len, rc;

    const char *file;

    char def_file[128];

    const NetdevDumpOptions *dump;

    NetClientState *nc;

    DumpNetClient *dnc;



    assert(opts->type == NET_CLIENT_OPTIONS_KIND_DUMP);

    dump = opts->u.dump;



    assert(peer);



    if (dump->has_file) {

        file = dump->file;

    } else {

        int id;

        int ret;



        ret = net_hub_id_for_client(peer, &id);

        assert(ret == 0); /* peer must be on a hub */



        snprintf(def_file, sizeof(def_file), "qemu-vlan%d.pcap", id);

        file = def_file;

    }



    if (dump->has_len) {

        if (dump->len > INT_MAX) {

            error_setg(errp, "invalid length: %"PRIu64, dump->len);

            return -1;

        }

        len = dump->len;

    } else {

        len = 65536;

    }



    nc = qemu_new_net_client(&net_dump_info, peer, "dump", name);

    snprintf(nc->info_str, sizeof(nc->info_str),

             "dump to %s (len=%d)", file, len);



    dnc = DO_UPCAST(DumpNetClient, nc, nc);

    rc = net_dump_state_init(&dnc->ds, file, len, errp);

    if (rc) {

        qemu_del_net_client(nc);

    }

    return rc;

}
