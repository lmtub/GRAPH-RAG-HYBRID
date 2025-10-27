void virtio_net_exit(VirtIODevice *vdev)

{

    VirtIONet *n = DO_UPCAST(VirtIONet, vdev, vdev);

    qemu_del_vm_change_state_handler(n->vmstate);



    if (n->vhost_started) {

        vhost_net_stop(tap_get_vhost_net(n->nic->nc.peer), vdev);

    }



    qemu_purge_queued_packets(&n->nic->nc);



    unregister_savevm(n->qdev, "virtio-net", n);



    qemu_free(n->mac_table.macs);

    qemu_free(n->vlans);



    qemu_del_timer(n->tx_timer);

    qemu_free_timer(n->tx_timer);



    virtio_cleanup(&n->vdev);

    qemu_del_vlan_client(&n->nic->nc);

}
