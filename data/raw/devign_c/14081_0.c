static void virtio_net_cleanup(NetClientState *nc)

{

    VirtIONet *n = qemu_get_nic_opaque(nc);



    n->nic = NULL;

}
