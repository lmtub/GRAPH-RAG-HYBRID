int net_slirp_smb(const char *exported_dir)

{

    struct in_addr vserver_addr = { .s_addr = 0 };



    if (legacy_smb_export) {

        fprintf(stderr, "-smb given twice\n");

        return -1;

    }

    legacy_smb_export = exported_dir;

    if (!QTAILQ_EMPTY(&slirp_stacks)) {

        return slirp_smb(QTAILQ_FIRST(&slirp_stacks), exported_dir,

                         vserver_addr);

    }

    return 0;

}
