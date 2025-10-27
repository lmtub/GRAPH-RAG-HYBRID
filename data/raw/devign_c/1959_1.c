int read_targphys(const char *name,

                  int fd, target_phys_addr_t dst_addr, size_t nbytes)

{

    uint8_t *buf;

    size_t did;



    buf = g_malloc(nbytes);

    did = read(fd, buf, nbytes);

    if (did > 0)

        rom_add_blob_fixed("read", buf, did, dst_addr);

    g_free(buf);

    return did;

}
