static struct mmsghdr *build_l2tpv3_vector(NetL2TPV3State *s, int count)

{

    int i;

    struct iovec *iov;

    struct mmsghdr *msgvec, *result;



    msgvec = g_malloc(sizeof(struct mmsghdr) * count);

    result = msgvec;

    for (i = 0; i < count ; i++) {

        msgvec->msg_hdr.msg_name = NULL;

        msgvec->msg_hdr.msg_namelen = 0;

        iov =  g_malloc(sizeof(struct iovec) * IOVSIZE);

        msgvec->msg_hdr.msg_iov = iov;

        iov->iov_base = g_malloc(s->header_size);

        iov->iov_len = s->header_size;

        iov++ ;

        iov->iov_base = qemu_memalign(BUFFER_ALIGN, BUFFER_SIZE);

        iov->iov_len = BUFFER_SIZE;

        msgvec->msg_hdr.msg_iovlen = 2;

        msgvec->msg_hdr.msg_control = NULL;

        msgvec->msg_hdr.msg_controllen = 0;

        msgvec->msg_hdr.msg_flags = 0;

        msgvec++;

    }

    return result;

}
