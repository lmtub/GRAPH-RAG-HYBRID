static ssize_t v9fs_synth_preadv(FsContext *ctx, V9fsFidOpenState *fs,

                                 const struct iovec *iov,

                                 int iovcnt, off_t offset)

{

    int i, count = 0, rcount;

    V9fsSynthOpenState *synth_open = fs->private;

    V9fsSynthNode *node = synth_open->node;

    if (!node->attr->read) {

        errno = EPERM;

        return -1;

    }

    for (i = 0; i < iovcnt; i++) {

        rcount = node->attr->read(iov[i].iov_base, iov[i].iov_len,

                                  offset, node->private);

        offset += rcount;

        count  += rcount;

        /* If we read less than requested. we are done */

        if (rcount < iov[i].iov_len) {

            break;

        }

    }

    return count;

}
