static void virtio_ccw_notify(DeviceState *d, uint16_t vector)

{

    VirtioCcwDevice *dev = to_virtio_ccw_dev_fast(d);

    SubchDev *sch = dev->sch;

    uint64_t indicators;



    if (vector >= 128) {

        return;

    }



    if (vector < VIRTIO_PCI_QUEUE_MAX) {

        if (!dev->indicators) {

            return;

        }

        if (sch->thinint_active) {

            /*

             * In the adapter interrupt case, indicators points to a

             * memory area that may be (way) larger than 64 bit and

             * ind_bit indicates the start of the indicators in a big

             * endian notation.

             */

            uint64_t ind_bit = dev->routes.adapter.ind_offset;



            virtio_set_ind_atomic(sch, dev->indicators->addr +

                                  (ind_bit + vector) / 8,

                                  0x80 >> ((ind_bit + vector) % 8));

            if (!virtio_set_ind_atomic(sch, dev->summary_indicator->addr,

                                       0x01)) {

                css_adapter_interrupt(dev->thinint_isc);

            }

        } else {

            indicators = address_space_ldq(&address_space_memory,

                                           dev->indicators->addr,

                                           MEMTXATTRS_UNSPECIFIED,

                                           NULL);

            indicators |= 1ULL << vector;

            address_space_stq(&address_space_memory, dev->indicators->addr,

                              indicators, MEMTXATTRS_UNSPECIFIED, NULL);

            css_conditional_io_interrupt(sch);

        }

    } else {

        if (!dev->indicators2) {

            return;

        }

        vector = 0;

        indicators = address_space_ldq(&address_space_memory,

                                       dev->indicators2->addr,

                                       MEMTXATTRS_UNSPECIFIED,

                                       NULL);

        indicators |= 1ULL << vector;

        address_space_stq(&address_space_memory, dev->indicators2->addr,

                          indicators, MEMTXATTRS_UNSPECIFIED, NULL);

        css_conditional_io_interrupt(sch);

    }

}
