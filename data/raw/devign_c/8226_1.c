void qemu_iovec_destroy(QEMUIOVector *qiov)

{

    assert(qiov->nalloc != -1);




    g_free(qiov->iov);



}