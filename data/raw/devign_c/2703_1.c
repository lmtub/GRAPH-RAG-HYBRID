static int net_dump_init(NetClientState *peer, const char *device,

                         const char *name, const char *filename, int len)

{

    struct pcap_file_hdr hdr;

    NetClientState *nc;

    DumpState *s;

    struct tm tm;

    int fd;



    fd = open(filename, O_CREAT | O_TRUNC | O_WRONLY | O_BINARY, 0644);

    if (fd < 0) {

        error_report("-net dump: can't open %s", filename);

        return -1;

    }



    hdr.magic = PCAP_MAGIC;

    hdr.version_major = 2;

    hdr.version_minor = 4;

    hdr.thiszone = 0;

    hdr.sigfigs = 0;

    hdr.snaplen = len;

    hdr.linktype = 1;



    if (write(fd, &hdr, sizeof(hdr)) < sizeof(hdr)) {

        error_report("-net dump write error: %s", strerror(errno));

        close(fd);

        return -1;

    }



    nc = qemu_new_net_client(&net_dump_info, peer, device, name);



    snprintf(nc->info_str, sizeof(nc->info_str),

             "dump to %s (len=%d)", filename, len);



    s = DO_UPCAST(DumpState, nc, nc);



    s->fd = fd;

    s->pcap_caplen = len;



    qemu_get_timedate(&tm, 0);

    s->start_ts = mktime(&tm);



    return 0;

}
