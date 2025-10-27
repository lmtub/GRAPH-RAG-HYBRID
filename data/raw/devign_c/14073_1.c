bool bdrv_qiov_is_aligned(BlockDriverState *bs, QEMUIOVector *qiov)

{

    int i;



    for (i = 0; i < qiov->niov; i++) {

        if ((uintptr_t) qiov->iov[i].iov_base % bs->buffer_alignment) {

            return false;

        }

        if (qiov->iov[i].iov_len % bs->buffer_alignment) {

            return false;

        }

    }



    return true;

}
