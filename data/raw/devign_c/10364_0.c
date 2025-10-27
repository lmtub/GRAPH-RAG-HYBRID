static int virtio_ccw_handle_set_vq(SubchDev *sch, CCW1 ccw, bool check_len,

                                    bool is_legacy)

{

    int ret;

    VqInfoBlock info;

    VqInfoBlockLegacy linfo;

    size_t info_len = is_legacy ? sizeof(linfo) : sizeof(info);



    if (check_len) {

        if (ccw.count != info_len) {

            return -EINVAL;

        }

    } else if (ccw.count < info_len) {

        /* Can't execute command. */

        return -EINVAL;

    }

    if (!ccw.cda) {

        return -EFAULT;

    }

    if (is_legacy) {

        linfo.queue = address_space_ldq_be(&address_space_memory, ccw.cda,

                                           MEMTXATTRS_UNSPECIFIED, NULL);

        linfo.align = address_space_ldl_be(&address_space_memory,

                                           ccw.cda + sizeof(linfo.queue),

                                           MEMTXATTRS_UNSPECIFIED,

                                           NULL);

        linfo.index = address_space_lduw_be(&address_space_memory,

                                            ccw.cda + sizeof(linfo.queue)

                                            + sizeof(linfo.align),

                                            MEMTXATTRS_UNSPECIFIED,

                                            NULL);

        linfo.num = address_space_lduw_be(&address_space_memory,

                                          ccw.cda + sizeof(linfo.queue)

                                          + sizeof(linfo.align)

                                          + sizeof(linfo.index),

                                          MEMTXATTRS_UNSPECIFIED,

                                          NULL);

        ret = virtio_ccw_set_vqs(sch, NULL, &linfo);

    } else {

        info.desc = address_space_ldq_be(&address_space_memory, ccw.cda,

                                           MEMTXATTRS_UNSPECIFIED, NULL);

        info.index = address_space_lduw_be(&address_space_memory,

                                           ccw.cda + sizeof(info.desc)

                                           + sizeof(info.res0),

                                           MEMTXATTRS_UNSPECIFIED, NULL);

        info.num = address_space_lduw_be(&address_space_memory,

                                         ccw.cda + sizeof(info.desc)

                                         + sizeof(info.res0)

                                         + sizeof(info.index),

                                         MEMTXATTRS_UNSPECIFIED, NULL);

        info.avail = address_space_ldq_be(&address_space_memory,

                                          ccw.cda + sizeof(info.desc)

                                          + sizeof(info.res0)

                                          + sizeof(info.index)

                                          + sizeof(info.num),

                                          MEMTXATTRS_UNSPECIFIED, NULL);

        info.used = address_space_ldq_be(&address_space_memory,

                                         ccw.cda + sizeof(info.desc)

                                         + sizeof(info.res0)

                                         + sizeof(info.index)

                                         + sizeof(info.num)

                                         + sizeof(info.avail),

                                         MEMTXATTRS_UNSPECIFIED, NULL);

        ret = virtio_ccw_set_vqs(sch, &info, NULL);

    }

    sch->curr_status.scsw.count = 0;

    return ret;

}
