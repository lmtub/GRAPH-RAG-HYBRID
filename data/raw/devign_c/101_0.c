static void vring_desc_read(VirtIODevice *vdev, VRingDesc *desc,

                            hwaddr desc_pa, int i)

{

    address_space_read(&address_space_memory, desc_pa + i * sizeof(VRingDesc),

                       MEMTXATTRS_UNSPECIFIED, (void *)desc, sizeof(VRingDesc));

    virtio_tswap64s(vdev, &desc->addr);

    virtio_tswap32s(vdev, &desc->len);

    virtio_tswap16s(vdev, &desc->flags);

    virtio_tswap16s(vdev, &desc->next);

}
