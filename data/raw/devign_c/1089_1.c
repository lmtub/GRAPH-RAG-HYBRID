int vhost_backend_invalidate_device_iotlb(struct vhost_dev *dev,

                                                 uint64_t iova, uint64_t len)

{

    struct vhost_iotlb_msg imsg;



    imsg.iova = iova;

    imsg.size = len;

    imsg.type = VHOST_IOTLB_INVALIDATE;



    return dev->vhost_ops->vhost_send_device_iotlb_msg(dev, &imsg);

}
