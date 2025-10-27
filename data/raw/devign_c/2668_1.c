static void vmxnet3_update_features(VMXNET3State *s)

{

    uint32_t guest_features;

    int rxcso_supported;



    guest_features = VMXNET3_READ_DRV_SHARED32(s->drv_shmem,

                                               devRead.misc.uptFeatures);



    rxcso_supported = VMXNET_FLAG_IS_SET(guest_features, UPT1_F_RXCSUM);

    s->rx_vlan_stripping = VMXNET_FLAG_IS_SET(guest_features, UPT1_F_RXVLAN);

    s->lro_supported = VMXNET_FLAG_IS_SET(guest_features, UPT1_F_LRO);



    VMW_CFPRN("Features configuration: LRO: %d, RXCSUM: %d, VLANSTRIP: %d",

              s->lro_supported, rxcso_supported,

              s->rx_vlan_stripping);

    if (s->peer_has_vhdr) {

        qemu_set_offload(qemu_get_queue(s->nic)->peer,

                         rxcso_supported,

                         s->lro_supported,

                         s->lro_supported,

                         0,

                         0);

    }

}
