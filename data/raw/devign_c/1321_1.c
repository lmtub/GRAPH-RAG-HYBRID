static int sd_snapshot_create(BlockDriverState *bs, QEMUSnapshotInfo *sn_info)

{

    Error *local_err = NULL;

    BDRVSheepdogState *s = bs->opaque;

    int ret, fd;

    uint32_t new_vid;

    SheepdogInode *inode;

    unsigned int datalen;



    DPRINTF("sn_info: name %s id_str %s s: name %s vm_state_size %" PRId64 " "

            "is_snapshot %d\n", sn_info->name, sn_info->id_str,

            s->name, sn_info->vm_state_size, s->is_snapshot);



    if (s->is_snapshot) {

        error_report("You can't create a snapshot of a snapshot VDI, "

                     "%s (%" PRIu32 ").", s->name, s->inode.vdi_id);



        return -EINVAL;

    }



    DPRINTF("%s %s\n", sn_info->name, sn_info->id_str);



    s->inode.vm_state_size = sn_info->vm_state_size;

    s->inode.vm_clock_nsec = sn_info->vm_clock_nsec;

    /* It appears that inode.tag does not require a NUL terminator,

     * which means this use of strncpy is ok.

     */

    strncpy(s->inode.tag, sn_info->name, sizeof(s->inode.tag));

    /* we don't need to update entire object */

    datalen = SD_INODE_SIZE - sizeof(s->inode.data_vdi_id);



    /* refresh inode. */

    fd = connect_to_sdog(s, &local_err);

    if (fd < 0) {

        error_report("%s", error_get_pretty(local_err));;

        error_free(local_err);

        ret = fd;

        goto cleanup;

    }



    ret = write_object(fd, (char *)&s->inode, vid_to_vdi_oid(s->inode.vdi_id),

                       s->inode.nr_copies, datalen, 0, false, s->cache_flags);

    if (ret < 0) {

        error_report("failed to write snapshot's inode.");

        goto cleanup;

    }



    ret = do_sd_create(s, &new_vid, 1, &local_err);

    if (ret < 0) {

        error_report("%s", error_get_pretty(local_err));;

        error_free(local_err);

        error_report("failed to create inode for snapshot. %s",

                     strerror(errno));

        goto cleanup;

    }



    inode = (SheepdogInode *)g_malloc(datalen);



    ret = read_object(fd, (char *)inode, vid_to_vdi_oid(new_vid),

                      s->inode.nr_copies, datalen, 0, s->cache_flags);



    if (ret < 0) {

        error_report("failed to read new inode info. %s", strerror(errno));

        goto cleanup;

    }



    memcpy(&s->inode, inode, datalen);

    DPRINTF("s->inode: name %s snap_id %x oid %x\n",

            s->inode.name, s->inode.snap_id, s->inode.vdi_id);



cleanup:

    closesocket(fd);

    return ret;

}
