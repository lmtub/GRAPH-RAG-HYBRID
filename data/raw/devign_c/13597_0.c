static bool virtqueue_map_desc(VirtIODevice *vdev, unsigned int *p_num_sg,

                               hwaddr *addr, struct iovec *iov,

                               unsigned int max_num_sg, bool is_write,

                               hwaddr pa, size_t sz)

{

    bool ok = false;

    unsigned num_sg = *p_num_sg;

    assert(num_sg <= max_num_sg);



    if (!sz) {

        virtio_error(vdev, "virtio: zero sized buffers are not allowed");

        goto out;

    }



    while (sz) {

        hwaddr len = sz;



        if (num_sg == max_num_sg) {

            virtio_error(vdev, "virtio: too many write descriptors in "

                               "indirect table");

            goto out;

        }



        iov[num_sg].iov_base = cpu_physical_memory_map(pa, &len, is_write);

        if (!iov[num_sg].iov_base) {

            virtio_error(vdev, "virtio: bogus descriptor or out of resources");

            goto out;

        }



        iov[num_sg].iov_len = len;

        addr[num_sg] = pa;



        sz -= len;

        pa += len;

        num_sg++;

    }

    ok = true;



out:

    *p_num_sg = num_sg;

    return ok;

}
