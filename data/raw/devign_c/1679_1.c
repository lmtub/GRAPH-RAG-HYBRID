static struct iovec *cap_sg(struct iovec *sg, int cap, int *cnt)

{

    int i;

    int total = 0;



    for (i = 0; i < *cnt; i++) {

        if ((total + sg[i].iov_len) > cap) {

            sg[i].iov_len -= ((total + sg[i].iov_len) - cap);

            i++;

            break;

        }

        total += sg[i].iov_len;

    }



    *cnt = i;



    return sg;

}
