static int vhost_verify_ring_mappings(struct vhost_dev *dev,

                                      uint64_t start_addr,

                                      uint64_t size)

{

    int i, j;

    int r = 0;

    const char *part_name[] = {

        "descriptor table",

        "available ring",

        "used ring"

    };



    for (i = 0; i < dev->nvqs; ++i) {

        struct vhost_virtqueue *vq = dev->vqs + i;



        j = 0;

        r = vhost_verify_ring_part_mapping(vq->desc, vq->desc_phys,

                                           vq->desc_size, start_addr, size);

        if (!r) {

            break;

        }



        j++;

        r = vhost_verify_ring_part_mapping(vq->avail, vq->avail_phys,

                                           vq->avail_size, start_addr, size);

        if (!r) {

            break;

        }



        j++;

        r = vhost_verify_ring_part_mapping(vq->used, vq->used_phys,

                                           vq->used_size, start_addr, size);

        if (!r) {

            break;

        }

    }



    if (r == -ENOMEM) {

        error_report("Unable to map %s for ring %d", part_name[j], i);

    } else if (r == -EBUSY) {

        error_report("%s relocated for ring %d", part_name[j], i);

    }

    return r;

}
