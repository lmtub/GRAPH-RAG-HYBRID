static ssize_t socket_read(int sockfd, void *buff, size_t size)

{

    ssize_t retval, total = 0;



    while (size) {

        retval = read(sockfd, buff, size);

        if (retval == 0) {

            return -EIO;

        }

        if (retval < 0) {

            if (errno == EINTR) {

                continue;

            }

            return -errno;

        }

        size -= retval;

        buff += retval;

        total += retval;

    }

    return total;

}
