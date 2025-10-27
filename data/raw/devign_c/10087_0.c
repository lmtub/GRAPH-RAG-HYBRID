static size_t handle_aiocb_rw(struct qemu_paiocb *aiocb)

{

    size_t nbytes;

    char *buf;



    if (!(aiocb->aio_type & QEMU_AIO_MISALIGNED)) {

        /*

         * If there is just a single buffer, and it is properly aligned

         * we can just use plain pread/pwrite without any problems.

         */

        if (aiocb->aio_niov == 1)

             return handle_aiocb_rw_linear(aiocb, aiocb->aio_iov->iov_base);



        /*

         * We have more than one iovec, and all are properly aligned.

         *

         * Try preadv/pwritev first and fall back to linearizing the

         * buffer if it's not supported.

         */

	if (preadv_present) {

            nbytes = handle_aiocb_rw_vector(aiocb);

            if (nbytes == aiocb->aio_nbytes)

	        return nbytes;

            if (nbytes < 0 && nbytes != -ENOSYS)

                return nbytes;

            preadv_present = 0;

        }



        /*

         * XXX(hch): short read/write.  no easy way to handle the reminder

         * using these interfaces.  For now retry using plain

         * pread/pwrite?

         */

    }



    /*

     * Ok, we have to do it the hard way, copy all segments into

     * a single aligned buffer.

     */

    buf = qemu_memalign(512, aiocb->aio_nbytes);

    if (aiocb->aio_type & QEMU_AIO_WRITE) {

        char *p = buf;

        int i;



        for (i = 0; i < aiocb->aio_niov; ++i) {

            memcpy(p, aiocb->aio_iov[i].iov_base, aiocb->aio_iov[i].iov_len);

            p += aiocb->aio_iov[i].iov_len;

        }

    }



    nbytes = handle_aiocb_rw_linear(aiocb, buf);

    if (!(aiocb->aio_type & QEMU_AIO_WRITE)) {

        char *p = buf;

        size_t count = aiocb->aio_nbytes, copy;

        int i;



        for (i = 0; i < aiocb->aio_niov && count; ++i) {

            copy = count;

            if (copy > aiocb->aio_iov[i].iov_len)

                copy = aiocb->aio_iov[i].iov_len;

            memcpy(aiocb->aio_iov[i].iov_base, p, copy);

            p     += copy;

            count -= copy;

        }

    }

    qemu_vfree(buf);



    return nbytes;

}
