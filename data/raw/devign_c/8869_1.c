int s390_virtio_hypercall(CPUState *env, uint64_t mem, uint64_t hypercall)

{

    int r = 0, i;



    dprintf("KVM hypercall: %ld\n", hypercall);

    switch (hypercall) {

    case KVM_S390_VIRTIO_NOTIFY:

        if (mem > ram_size) {

            VirtIOS390Device *dev = s390_virtio_bus_find_vring(s390_bus,

                                                               mem, &i);

            if (dev) {

                virtio_queue_notify(dev->vdev, i);

            } else {

                r = -EINVAL;

            }

        } else {

            /* Early printk */

        }

        break;

    case KVM_S390_VIRTIO_RESET:

    {

        VirtIOS390Device *dev;



        dev = s390_virtio_bus_find_mem(s390_bus, mem);

        virtio_reset(dev->vdev);


        s390_virtio_device_sync(dev);

        break;

    }

    case KVM_S390_VIRTIO_SET_STATUS:

    {

        VirtIOS390Device *dev;



        dev = s390_virtio_bus_find_mem(s390_bus, mem);

        if (dev) {

            s390_virtio_device_update_status(dev);

        } else {

            r = -EINVAL;

        }

        break;

    }

    default:

        r = -EINVAL;

        break;

    }



    return r;

}