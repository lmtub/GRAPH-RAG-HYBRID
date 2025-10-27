VLANClientState *qdev_get_vlan_client(DeviceState *dev,

                                      NetCanReceive *can_receive,

                                      NetReceive *receive,

                                      NetReceiveIOV *receive_iov,

                                      NetCleanup *cleanup,

                                      void *opaque)

{

    NICInfo *nd = dev->nd;

    assert(nd);

    return qemu_new_vlan_client(nd->vlan, nd->model, nd->name, can_receive,

                                receive, receive_iov, cleanup, opaque);

}
