static int net_dump_init(VLANState *vlan, const char *device,

                         const char *name, const char *filename, int len)

{

    struct pcap_file_hdr hdr;

    DumpState *s;



    s = qemu_malloc(sizeof(DumpState));



    s->fd = open(filename, O_CREAT | O_WRONLY | O_BINARY, 0644);

    if (s->fd < 0) {

        qemu_error("-net dump: can't open %s\n", filename);

        return -1;

    }



    s->pcap_caplen = len;



    hdr.magic = PCAP_MAGIC;

    hdr.version_major = 2;

    hdr.version_minor = 4;

    hdr.thiszone = 0;

    hdr.sigfigs = 0;

    hdr.snaplen = s->pcap_caplen;

    hdr.linktype = 1;



    if (write(s->fd, &hdr, sizeof(hdr)) < sizeof(hdr)) {

        qemu_error("-net dump write error: %s\n", strerror(errno));

        close(s->fd);

        qemu_free(s);

        return -1;

    }



    s->pcap_vc = qemu_new_vlan_client(NET_CLIENT_TYPE_DUMP,

                                      vlan, NULL, device, name, NULL,

                                      dump_receive, NULL, NULL,

                                      net_dump_cleanup, s);

    snprintf(s->pcap_vc->info_str, sizeof(s->pcap_vc->info_str),

             "dump to %s (len=%d)", filename, len);

    return 0;

}
