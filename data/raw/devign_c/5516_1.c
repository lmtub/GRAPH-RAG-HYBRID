int qemu_sendv(int sockfd, struct iovec *iov, int len, int iov_offset)

{

    return do_sendv_recvv(sockfd, iov, len, iov_offset, 1);

}
