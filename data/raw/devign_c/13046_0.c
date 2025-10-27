static unsigned virtqueue_read_next_desc(VirtIODevice *vdev, VRingDesc *desc,

                                         hwaddr desc_pa, unsigned int max)

{

    unsigned int next;



    /* If this descriptor says it doesn't chain, we're done. */

    if (!(desc->flags & VRING_DESC_F_NEXT)) {

        return max;

    }



    /* Check they're not leading us off end of descriptors. */

    next = desc->next;

    /* Make sure compiler knows to grab that: we don't want it changing! */

    smp_wmb();



    if (next >= max) {

        error_report("Desc next is %u", next);

        exit(1);

    }



    vring_desc_read(vdev, desc, desc_pa, next);

    return next;

}
