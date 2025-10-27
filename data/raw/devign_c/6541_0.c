static int v9fs_receive_response(V9fsProxy *proxy, int type,

                                 int *status, void *response)

{

    int retval;

    ProxyHeader header;

    struct iovec *reply = &proxy->in_iovec;



    *status = 0;

    reply->iov_len = 0;

    retval = socket_read(proxy->sockfd, reply->iov_base, PROXY_HDR_SZ);

    if (retval < 0) {

        return retval;

    }

    reply->iov_len = PROXY_HDR_SZ;

    proxy_unmarshal(reply, 0, "dd", &header.type, &header.size);

    /*

     * if response size > PROXY_MAX_IO_SZ, read the response but ignore it and

     * return -ENOBUFS

     */

    if (header.size > PROXY_MAX_IO_SZ) {

        int count;

        while (header.size > 0) {

            count = MIN(PROXY_MAX_IO_SZ, header.size);

            count = socket_read(proxy->sockfd, reply->iov_base, count);

            if (count < 0) {

                return count;

            }

            header.size -= count;

        }

        *status = -ENOBUFS;

        return 0;

    }



    retval = socket_read(proxy->sockfd,

                         reply->iov_base + PROXY_HDR_SZ, header.size);

    if (retval < 0) {

        return retval;

    }

    reply->iov_len += header.size;

    /* there was an error during processing request */

    if (header.type == T_ERROR) {

        int ret;

        ret = proxy_unmarshal(reply, PROXY_HDR_SZ, "d", status);

        if (ret < 0) {

            *status = ret;

        }

        return 0;

    }



    switch (type) {

    case T_LSTAT: {

        ProxyStat prstat;

        retval = proxy_unmarshal(reply, PROXY_HDR_SZ,

                                 "qqqdddqqqqqqqqqq", &prstat.st_dev,

                                 &prstat.st_ino, &prstat.st_nlink,

                                 &prstat.st_mode, &prstat.st_uid,

                                 &prstat.st_gid, &prstat.st_rdev,

                                 &prstat.st_size, &prstat.st_blksize,

                                 &prstat.st_blocks,

                                 &prstat.st_atim_sec, &prstat.st_atim_nsec,

                                 &prstat.st_mtim_sec, &prstat.st_mtim_nsec,

                                 &prstat.st_ctim_sec, &prstat.st_ctim_nsec);

        prstat_to_stat(response, &prstat);

        break;

    }

    case T_STATFS: {

        ProxyStatFS prstfs;

        retval = proxy_unmarshal(reply, PROXY_HDR_SZ,

                                 "qqqqqqqqqqq", &prstfs.f_type,

                                 &prstfs.f_bsize, &prstfs.f_blocks,

                                 &prstfs.f_bfree, &prstfs.f_bavail,

                                 &prstfs.f_files, &prstfs.f_ffree,

                                 &prstfs.f_fsid[0], &prstfs.f_fsid[1],

                                 &prstfs.f_namelen, &prstfs.f_frsize);

        prstatfs_to_statfs(response, &prstfs);

        break;

    }

    case T_READLINK: {

        V9fsString target;

        v9fs_string_init(&target);

        retval = proxy_unmarshal(reply, PROXY_HDR_SZ, "s", &target);

        strcpy(response, target.data);

        v9fs_string_free(&target);

        break;

    }

    case T_LGETXATTR:

    case T_LLISTXATTR: {

        V9fsString xattr;

        v9fs_string_init(&xattr);

        retval = proxy_unmarshal(reply, PROXY_HDR_SZ, "s", &xattr);

        memcpy(response, xattr.data, xattr.size);

        v9fs_string_free(&xattr);

        break;

    }

    case T_GETVERSION:

        proxy_unmarshal(reply, PROXY_HDR_SZ, "q", response);

        break;

    default:

        return -1;

    }

    if (retval < 0) {

        *status  = retval;

    }

    return 0;

}
