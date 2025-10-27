static int sd_create_branch(BDRVSheepdogState *s)

{

    int ret, fd;

    uint32_t vid;

    char *buf;



    dprintf("%" PRIx32 " is snapshot.\n", s->inode.vdi_id);



    buf = g_malloc(SD_INODE_SIZE);



    ret = do_sd_create(s->name, s->inode.vdi_size, s->inode.vdi_id, &vid, 1,

                       s->addr, s->port);

    if (ret) {

        goto out;

    }



    dprintf("%" PRIx32 " is created.\n", vid);



    fd = connect_to_sdog(s->addr, s->port);

    if (fd < 0) {

        error_report("failed to connect");

        ret = fd;

        goto out;

    }



    ret = read_object(fd, buf, vid_to_vdi_oid(vid), s->inode.nr_copies,

                      SD_INODE_SIZE, 0, s->cache_enabled);



    closesocket(fd);



    if (ret < 0) {

        goto out;

    }



    memcpy(&s->inode, buf, sizeof(s->inode));



    s->is_snapshot = false;

    ret = 0;

    dprintf("%" PRIx32 " was newly created.\n", s->inode.vdi_id);



out:

    g_free(buf);



    return ret;

}
