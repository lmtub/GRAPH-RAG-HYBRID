vu_queue_notify(VuDev *dev, VuVirtq *vq)

{

    if (unlikely(dev->broken)) {

        return;

    }



    if (!vring_notify(dev, vq)) {

        DPRINT("skipped notify...\n");

        return;

    }



    if (eventfd_write(vq->call_fd, 1) < 0) {

        vu_panic(dev, "Error writing eventfd: %s", strerror(errno));

    }

}
